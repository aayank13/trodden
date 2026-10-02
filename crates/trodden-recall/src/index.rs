use std::{
    collections::{BTreeMap, HashSet},
    fs::{self, File},
    io::{BufWriter, Read, Seek, SeekFrom, Write},
    path::Path,
};

use anyhow::{Context, Result, ensure};
use trodden_embed::{CODE_LEN, Embedding, QUANTIZED_LEN, Quantized};

const MAGIC: &[u8; 8] = b"TRDIDX\x01\0";
const HEADER_LEN: usize = 16;
const GROUP_LEN: usize = 16;

const RESCORE: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbor {
    pub rowid: i64,
    pub cosine: f32,
}

#[derive(Debug)]
pub struct VectorIndex {
    file: File,
    count: usize,
    groups: BTreeMap<u64, (usize, usize)>,
}

impl VectorIndex {
    pub fn build(rows: &[(i64, String, Vec<u8>)], path: &Path) -> Result<()> {
        let mut grouped: BTreeMap<u64, Vec<(i64, Quantized)>> = BTreeMap::new();
        for (rowid, repo, bytes) in rows {
            let vector = Quantized::from_bytes(bytes).context("decode a stored embedding")?;
            grouped
                .entry(Self::repo_hash(repo))
                .or_default()
                .push((*rowid, vector));
        }

        let temp = path.with_extension("tmp");
        let file = File::create(&temp).with_context(|| format!("create {}", temp.display()))?;
        let mut out = BufWriter::new(file);
        let mut write = |bytes: &[u8]| out.write_all(bytes).context("write the vector index");
        let count = u32::try_from(rows.len()).context("count indexed procedures")?;
        let groups = u32::try_from(grouped.len()).context("count indexed repositories")?;
        write(MAGIC)?;
        write(&count.to_le_bytes())?;
        write(&groups.to_le_bytes())?;
        let mut start = 0_u32;
        for (hash, members) in &grouped {
            let len = u32::try_from(members.len()).context("count a repository's procedures")?;
            write(&hash.to_le_bytes())?;
            write(&start.to_le_bytes())?;
            write(&len.to_le_bytes())?;
            start += len;
        }
        let members: Vec<&(i64, Quantized)> = grouped.values().flatten().collect();
        for (rowid, _) in &members {
            write(&rowid.to_le_bytes())?;
        }
        for (_, vector) in &members {
            write(&vector.code())?;
        }
        for (_, vector) in &members {
            write(&vector.to_bytes())?;
        }
        out.flush().context("flush the vector index")?;
        drop(out);
        fs::rename(&temp, path)
            .with_context(|| format!("move the vector index into place at {}", path.display()))
    }

    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mut header = [0; HEADER_LEN];
        file.read_exact(&mut header)
            .context("read the vector index header")?;
        ensure!(
            &header[..8] == MAGIC,
            "{} is not a vector index",
            path.display()
        );
        let count = u32::from_le_bytes([header[8], header[9], header[10], header[11]]) as usize;
        let group_count =
            u32::from_le_bytes([header[12], header[13], header[14], header[15]]) as usize;

        let mut table = vec![0; group_count * GROUP_LEN];
        file.read_exact(&mut table)
            .context("read the vector index groups")?;
        let groups = table
            .as_chunks::<GROUP_LEN>()
            .0
            .iter()
            .map(|group| {
                let hash = u64::from_le_bytes(std::array::from_fn(|i| group[i]));
                let start = u32::from_le_bytes(std::array::from_fn(|i| group[8 + i]));
                let len = u32::from_le_bytes(std::array::from_fn(|i| group[12 + i]));
                (hash, (start as usize, len as usize))
            })
            .collect();
        Ok(Self {
            file,
            count,
            groups,
        })
    }

    pub fn search(&mut self, query: &Embedding, repo: &str, limit: usize) -> Result<Vec<Neighbor>> {
        let query_words = Self::words(&Quantized::query_code(query));
        let mut ranges: Vec<(usize, usize)> = [repo, ""]
            .iter()
            .filter_map(|repo| self.groups.get(&Self::repo_hash(repo)).copied())
            .collect();
        ranges.dedup();

        let total = ranges.iter().map(|(_, len)| len).sum();
        let mut closest: Vec<(u32, usize)> = Vec::with_capacity(total);
        for (start, len) in ranges {
            let mut codes = vec![0; len * CODE_LEN];
            self.read_at(self.codes_offset() + start * CODE_LEN, &mut codes)?;
            for (offset, code) in codes.as_chunks::<CODE_LEN>().0.iter().enumerate() {
                let distance = Self::words(code)
                    .iter()
                    .zip(&query_words)
                    .map(|(a, b)| (a ^ b).count_ones())
                    .sum();
                closest.push((distance, start + offset));
            }
        }
        if closest.len() > RESCORE {
            closest.select_nth_unstable(RESCORE);
            closest.truncate(RESCORE);
        }

        let mut neighbors = Vec::with_capacity(closest.len());
        let mut vector = [0; QUANTIZED_LEN];
        let mut rowid = [0; 8];
        for (_, position) in closest {
            self.read_at(
                self.vectors_offset() + position * QUANTIZED_LEN,
                &mut vector,
            )?;
            self.read_at(self.rowids_offset() + position * 8, &mut rowid)?;
            let quantized = Quantized::from_array(&vector);
            neighbors.push(Neighbor {
                rowid: i64::from_le_bytes(rowid),
                cosine: quantized.cosine(query),
            });
        }
        neighbors.sort_by(|a, b| b.cosine.total_cmp(&a.cosine));
        let mut seen = HashSet::with_capacity(neighbors.len());
        neighbors.retain(|neighbor| seen.insert(neighbor.rowid));
        neighbors.truncate(limit);
        Ok(neighbors)
    }

    fn words(code: &[u8; CODE_LEN]) -> [u64; CODE_LEN / 8] {
        std::array::from_fn(|word| {
            u64::from_le_bytes(std::array::from_fn(|byte| code[word * 8 + byte]))
        })
    }

    fn rowids_offset(&self) -> usize {
        HEADER_LEN + self.groups.len() * GROUP_LEN
    }

    fn codes_offset(&self) -> usize {
        self.rowids_offset() + self.count * 8
    }

    fn vectors_offset(&self) -> usize {
        self.codes_offset() + self.count * CODE_LEN
    }

    fn read_at(&mut self, offset: usize, buffer: &mut [u8]) -> Result<()> {
        self.file
            .seek(SeekFrom::Start(offset as u64))
            .context("seek in the vector index")?;
        self.file
            .read_exact(buffer)
            .context("read the vector index")
    }

    fn repo_hash(repo: &str) -> u64 {
        repo.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
        })
    }
}
