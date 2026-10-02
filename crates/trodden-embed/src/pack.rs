use std::{
    cmp::Ordering,
    collections::HashMap,
    fs::{self, File},
    io::{BufWriter, Read, Seek, SeekFrom, Write},
    path::Path,
};

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;

use crate::{
    DIMS, Embedding,
    tokenizer::{Tokenizer, Vocabulary},
};

const MAGIC: &[u8; 8] = b"TRDEMB\x01\0";
const HEADER_LEN: usize = 24;
const INDEX_ENTRY_LEN: usize = 12;
const ROW_LEN: usize = 4 + DIMS;

const MAX_TOKENS: usize = 512;

#[derive(Debug)]
pub struct ModelPack;

impl ModelPack {
    pub const MODEL_REPO: &str = "minishlab/potion-code-16M-v2";

    pub const MODEL_FILES: [&str; 2] = ["tokenizer.json", "model.safetensors"];

    pub fn import(model_dir: &Path, out: &Path) -> Result<()> {
        let vocab = TokenizerFile::load(&model_dir.join("tokenizer.json"))?;
        let weights = Safetensors::load(&model_dir.join("model.safetensors"))?;
        ensure!(
            weights.rows == vocab.len(),
            "model has {} rows but the tokenizer has {} tokens",
            weights.rows,
            vocab.len()
        );

        let mut tokens: Vec<(&str, u32)> = vocab
            .iter()
            .map(|(token, id)| (token.as_str(), *id))
            .collect();
        tokens.sort_unstable_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
        let strings_len: usize = tokens.iter().map(|(token, _)| token.len()).sum();

        let temp = out.with_extension("tmp");
        let file = File::create(&temp).with_context(|| format!("create {}", temp.display()))?;
        let mut writer = BufWriter::new(file);
        let write = |writer: &mut BufWriter<File>, bytes: &[u8]| {
            writer
                .write_all(bytes)
                .with_context(|| format!("write {}", temp.display()))
        };

        write(&mut writer, MAGIC)?;
        for value in [DIMS, tokens.len(), strings_len, 0] {
            write(&mut writer, &Self::to_u32(value)?.to_le_bytes())?;
        }
        let mut offset = 0;
        for (token, id) in &tokens {
            write(&mut writer, &Self::to_u32(offset)?.to_le_bytes())?;
            write(&mut writer, &Self::to_u32(token.len())?.to_le_bytes())?;
            write(&mut writer, &id.to_le_bytes())?;
            offset += token.len();
        }
        for (token, _) in &tokens {
            write(&mut writer, token.as_bytes())?;
        }
        for row in 0..weights.rows {
            let values = weights.row(row);
            let max = values
                .iter()
                .fold(0.0_f32, |max, value| max.max(value.abs()));
            let scale = if max > 0.0 { max / 127.0 } else { 1.0 };
            write(&mut writer, &scale.to_le_bytes())?;
            let quantized: Vec<u8> = values
                .iter()
                .map(|value| {
                    #[expect(clippy::cast_possible_truncation, reason = "clamped to i8 range")]
                    let q = (value / scale).round().clamp(-127.0, 127.0) as i8;
                    q.to_le_bytes()[0]
                })
                .collect();
            write(&mut writer, &quantized)?;
        }
        writer
            .flush()
            .with_context(|| format!("flush {}", temp.display()))?;
        drop(writer);
        fs::rename(&temp, out).with_context(|| format!("move pack into place at {}", out.display()))
    }

    fn to_u32(value: usize) -> Result<u32> {
        u32::try_from(value).context("fit a pack field in 32 bits")
    }
}

#[derive(Deserialize)]
struct TokenizerFile {
    model: TokenizerModel,
    normalizer: Option<TokenizerNormalizer>,
}

#[derive(Deserialize)]
struct TokenizerModel {
    #[serde(rename = "type")]
    kind: String,
    vocab: HashMap<String, u32>,
}

#[derive(Deserialize)]
struct TokenizerNormalizer {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    lowercase: bool,
}

impl TokenizerFile {
    fn load(path: &Path) -> Result<HashMap<String, u32>> {
        let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let file: Self =
            serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
        let normalizer = file.normalizer.as_ref();
        if file.model.kind != "WordPiece"
            || normalizer.is_none_or(|normalizer| {
                normalizer.kind != "BertNormalizer" || !normalizer.lowercase
            })
        {
            bail!("only lowercasing BERT WordPiece tokenizers are supported");
        }
        Ok(file.model.vocab)
    }
}

struct Safetensors {
    data: Vec<u8>,
    rows: usize,
    half: bool,
}

#[derive(Deserialize)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

impl Safetensors {
    fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let header_len = bytes
            .get(..8)
            .and_then(|len| len.try_into().ok())
            .map(u64::from_le_bytes)
            .and_then(|len| usize::try_from(len).ok())
            .context("read the safetensors header length")?;
        let header = bytes
            .get(8..8 + header_len)
            .context("read the safetensors header")?;
        let tensors: HashMap<String, serde_json::Value> =
            serde_json::from_slice(header).context("parse the safetensors header")?;
        let info: TensorInfo = tensors
            .get("embeddings")
            .cloned()
            .map(serde_json::from_value)
            .context("find the `embeddings` tensor")?
            .context("parse the `embeddings` tensor description")?;

        let half = match info.dtype.as_str() {
            "F16" => true,
            "F32" => false,
            other => bail!("unsupported embedding dtype {other}"),
        };
        let [rows, dims] = info.shape[..] else {
            bail!("embeddings are not a matrix")
        };
        ensure!(
            dims == DIMS,
            "embeddings have {dims} dimensions, expected {DIMS}"
        );
        let start = 8 + header_len + info.data_offsets[0];
        let end = 8 + header_len + info.data_offsets[1];
        let data = bytes
            .get(start..end)
            .context("read the embedding data")?
            .to_vec();
        ensure!(
            data.len() == rows * dims * if half { 2 } else { 4 },
            "embedding data has the wrong size"
        );
        Ok(Self { data, rows, half })
    }

    fn row(&self, row: usize) -> Vec<f32> {
        let width = if self.half { 2 } else { 4 };
        let bytes = &self.data[row * DIMS * width..(row + 1) * DIMS * width];
        bytes
            .chunks_exact(width)
            .map(|chunk| {
                if self.half {
                    Self::f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]))
                } else {
                    f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                }
            })
            .collect()
    }

    fn f16_to_f32(bits: u16) -> f32 {
        let sign = u32::from(bits >> 15) << 31;
        let exponent = u32::from((bits >> 10) & 0x1f);
        let mantissa = u32::from(bits & 0x3ff);
        let magnitude = match (exponent, mantissa) {
            (0, 0) => 0,
            (0, _) => {
                let shift = mantissa.leading_zeros() - 21;
                ((113 - shift) << 23) | ((mantissa << shift) & 0x3ff) << 13
            }
            (0x1f, _) => (0xff << 23) | (mantissa << 13),
            _ => ((exponent + 112) << 23) | (mantissa << 13),
        };
        f32::from_bits(sign | magnitude)
    }
}

#[derive(Debug)]
pub struct Embedder {
    file: File,
    index: Vec<u8>,
    strings: Vec<u8>,
    rows_offset: u64,
}

impl Embedder {
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mut header = [0; HEADER_LEN];
        file.read_exact(&mut header)
            .context("read the pack header")?;
        ensure!(
            &header[..8] == MAGIC,
            "{} is not an embedding pack",
            path.display()
        );
        let field = |at: usize| {
            u32::from_le_bytes([header[at], header[at + 1], header[at + 2], header[at + 3]])
                as usize
        };
        let (dims, vocab, strings_len) = (field(8), field(12), field(16));
        ensure!(dims == DIMS, "pack has {dims} dimensions, expected {DIMS}");

        let mut index = vec![0; vocab * INDEX_ENTRY_LEN];
        file.read_exact(&mut index).context("read the pack index")?;
        let mut strings = vec![0; strings_len];
        file.read_exact(&mut strings)
            .context("read the pack vocabulary")?;
        let rows_offset =
            u64::try_from(HEADER_LEN + index.len() + strings.len()).context("locate pack rows")?;
        Ok(Self {
            file,
            index,
            strings,
            rows_offset,
        })
    }

    pub fn embed(&mut self, text: &str) -> Result<Option<Embedding>> {
        let mut ids = self.token_ids(text);
        if ids.is_empty() {
            return Ok(None);
        }

        let count = ids.len();
        ids.sort_unstable();
        let mut sum = [0.0_f32; DIMS];
        let mut row = [0_u8; ROW_LEN];
        for run in ids.chunk_by(|a, b| a == b) {
            let offset = self.rows_offset + u64::from(run[0]) * ROW_LEN as u64;
            self.file
                .seek(SeekFrom::Start(offset))
                .context("seek to a pack row")?;
            self.file.read_exact(&mut row).context("read a pack row")?;
            let scale = f32::from_le_bytes([row[0], row[1], row[2], row[3]]);
            let weight = scale * run.len() as f32;
            for (total, byte) in sum.iter_mut().zip(&row[4..]) {
                *total += f32::from(i8::from_le_bytes([*byte])) * weight;
            }
        }

        let count = count as f32;
        let norm = sum
            .iter()
            .map(|value| (value / count).powi(2))
            .sum::<f32>()
            .sqrt();
        if norm == 0.0 {
            return Ok(None);
        }
        Ok(Some(sum.map(|value| value / count / norm)))
    }

    pub fn token_ids(&self, text: &str) -> Vec<u32> {
        let mut ids = Tokenizer::encode(text, self);
        ids.truncate(MAX_TOKENS);
        ids
    }

    fn entry(&self, position: usize) -> (&[u8], u32) {
        let entry = &self.index[position * INDEX_ENTRY_LEN..(position + 1) * INDEX_ENTRY_LEN];
        let read = |at: usize| {
            u32::from_le_bytes([entry[at], entry[at + 1], entry[at + 2], entry[at + 3]])
        };
        let (offset, len) = (read(0) as usize, read(4) as usize);
        (&self.strings[offset..offset + len], read(8))
    }
}

impl Vocabulary for Embedder {
    fn id(&self, token: &str) -> Option<u32> {
        let (mut low, mut high) = (0, self.index.len() / INDEX_ENTRY_LEN);
        while low < high {
            let middle = low + (high - low) / 2;
            let (bytes, id) = self.entry(middle);
            match bytes.cmp(token.as_bytes()) {
                Ordering::Less => low = middle + 1,
                Ordering::Greater => high = middle,
                Ordering::Equal => return Some(id),
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widens_half_precision() {
        assert_eq!(Safetensors::f16_to_f32(0x3c00), 1.0);
        assert_eq!(Safetensors::f16_to_f32(0xc000), -2.0);
        assert_eq!(Safetensors::f16_to_f32(0x3555), 0.333_251_95);
        assert_eq!(Safetensors::f16_to_f32(0x0001), 5.960_464_5e-8);
        assert!(Safetensors::f16_to_f32(0x7c00).is_infinite());
    }
}
