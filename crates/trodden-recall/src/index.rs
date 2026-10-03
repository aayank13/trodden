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
        let file_len = file
            .metadata()
            .with_context(|| format!("read the size of {}", path.display()))?
            .len();
        let mut header = [0; HEADER_LEN];
        file.read_exact(&mut header)
            .context("read the vector index header")?;
        ensure!(
            &header[..8] == MAGIC,
            "{} is not a vector index",
            path.display()
        );
        let count = u32::from_le_bytes([header[8], header[9], header[10], header[11]]);
        let group_count = u32::from_le_bytes([header[12], header[13], header[14], header[15]]);

        let table_len = u64::from(group_count) * GROUP_LEN as u64;
        let index_len = HEADER_LEN as u64
            + table_len
            + u64::from(count) * (8 + CODE_LEN + QUANTIZED_LEN) as u64;
        ensure!(
            index_len <= file_len,
            "{} is truncated: its header describes {index_len} bytes but the file has {file_len}",
            path.display()
        );
        let mut table =
            vec![0; usize::try_from(table_len).context("size the vector index groups")?];
        file.read_exact(&mut table)
            .context("read the vector index groups")?;
        let count = count as usize;
        let groups: BTreeMap<u64, (usize, usize)> = table
            .as_chunks::<GROUP_LEN>()
            .0
            .iter()
            .map(|group| {
                let hash = u64::from_le_bytes(std::array::from_fn(|i| group[i]));
                let start = u32::from_le_bytes(std::array::from_fn(|i| group[8 + i])) as usize;
                let len = u32::from_le_bytes(std::array::from_fn(|i| group[12 + i])) as usize;
                ensure!(
                    start.checked_add(len).is_some_and(|end| end <= count),
                    "{} has a repository group outside its {count} procedures",
                    path.display()
                );
                Ok((hash, (start, len)))
            })
            .collect::<Result<_>>()?;
        ensure!(
            groups.len() == group_count as usize,
            "{} lists a repository group twice",
            path.display()
        );
        Ok(Self {
            file,
            count,
            groups,
        })
    }

    pub fn search(&mut self, query: &Embedding, repo: &str, limit: usize) -> Result<Vec<Neighbor>> {
        let query_words = Self::words(&Quantized::query_code(query));
        let ranges = self.ranges(repo);

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

    pub fn cosines(
        &mut self,
        query: &Embedding,
        repo: &str,
        rowids: &[i64],
    ) -> Result<Vec<Neighbor>> {
        let mut neighbors: Vec<Neighbor> = Vec::with_capacity(rowids.len());
        if rowids.is_empty() {
            return Ok(neighbors);
        }
        let mut vector = [0; QUANTIZED_LEN];
        for (start, len) in self.ranges(repo) {
            let mut ids = vec![0; len * 8];
            self.read_at(self.rowids_offset() + start * 8, &mut ids)?;
            for (offset, id) in ids.as_chunks::<8>().0.iter().enumerate() {
                let rowid = i64::from_le_bytes(*id);
                if !rowids.contains(&rowid) {
                    continue;
                }
                self.read_at(
                    self.vectors_offset() + (start + offset) * QUANTIZED_LEN,
                    &mut vector,
                )?;
                let cosine = Quantized::from_array(&vector).cosine(query);
                match neighbors
                    .iter_mut()
                    .find(|neighbor| neighbor.rowid == rowid)
                {
                    Some(neighbor) => neighbor.cosine = neighbor.cosine.max(cosine),
                    None => neighbors.push(Neighbor { rowid, cosine }),
                }
            }
        }
        Ok(neighbors)
    }

    fn ranges(&self, repo: &str) -> Vec<(usize, usize)> {
        let mut ranges: Vec<(usize, usize)> = [repo, ""]
            .iter()
            .filter_map(|repo| self.groups.get(&Self::repo_hash(repo)).copied())
            .collect();
        ranges.dedup();
        ranges
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

#[cfg(test)]
mod tests {
    use trodden_embed::DIMS;

    use super::*;

    fn direction(weights: &[(usize, f32)]) -> Embedding {
        let mut embedding = [0.0; DIMS];
        for (axis, weight) in weights {
            embedding[*axis] = *weight;
        }
        embedding
    }

    #[derive(Debug)]
    struct Fixture;

    impl Fixture {
        fn index(name: &str) -> Vec<u8> {
            let rows: Vec<(i64, String, Vec<u8>)> = [(1, "repo-a"), (2, "repo-b"), (3, "")]
                .into_iter()
                .map(|(rowid, repo)| {
                    (
                        rowid,
                        repo.to_owned(),
                        Quantized::new(&direction(&[(0, 1.0)])).to_bytes(),
                    )
                })
                .collect();
            let path = Self::path(&format!("{name}-built"));
            VectorIndex::build(&rows, &path).expect("index builds");
            let index = fs::read(&path).expect("index is readable");
            fs::remove_file(&path).expect("index is removable");
            index
        }

        fn path(name: &str) -> std::path::PathBuf {
            std::env::temp_dir().join(format!("trodden-index-{name}-{}.index", std::process::id()))
        }

        fn patch(index: &mut [u8], at: usize, value: u32) {
            index[at..at + 4].copy_from_slice(&value.to_le_bytes());
        }

        fn open(name: &str, index: &[u8]) -> Result<VectorIndex> {
            let path = Self::path(name);
            fs::write(&path, index).expect("index is writable");
            let opened = VectorIndex::open(&path);
            fs::remove_file(&path).expect("index is removable");
            opened
        }

        fn rejects(name: &str, index: &[u8], reason: &str) {
            let error = Self::open(name, index).expect_err("corrupt index is rejected");
            let message = format!("{error:#}");
            assert!(message.contains(reason), "{message}");
        }
    }

    #[test]
    fn searches_a_valid_index() {
        let mut index = Fixture::open("search", &Fixture::index("search")).expect("index opens");

        let neighbors = index
            .search(&direction(&[(0, 1.0)]), "repo-a", 10)
            .expect("index searches");

        let mut rowids: Vec<i64> = neighbors.iter().map(|neighbor| neighbor.rowid).collect();
        rowids.sort_unstable();
        assert_eq!(rowids, [1, 3]);
    }

    #[test]
    fn rejects_counts_larger_than_the_file_before_allocating() {
        for (name, at) in [("groups", 12), ("count", 8)] {
            let mut index = Fixture::index(name);
            Fixture::patch(&mut index, at, u32::MAX);
            Fixture::rejects(name, &index, "is truncated");
        }
    }

    #[test]
    fn rejects_a_group_outside_the_procedures() {
        for (name, at) in [("start", HEADER_LEN + 8), ("len", HEADER_LEN + 12)] {
            let mut index = Fixture::index(name);
            Fixture::patch(&mut index, at, 0x7fff_ffff);
            Fixture::rejects(name, &index, "repository group outside");
        }
    }

    #[test]
    fn rejects_a_repeated_group() {
        let mut index = Fixture::index("repeated");
        let first: Vec<u8> = index[HEADER_LEN..HEADER_LEN + 8].to_vec();
        index[HEADER_LEN + GROUP_LEN..HEADER_LEN + GROUP_LEN + 8].copy_from_slice(&first);
        Fixture::rejects("repeated", &index, "repository group twice");
    }

    #[test]
    fn rejects_a_truncated_index() {
        let index = Fixture::index("truncated");
        Fixture::rejects("truncated", &index[..index.len() - 1], "is truncated");
        Fixture::rejects(
            "header",
            &index[..HEADER_LEN - 1],
            "read the vector index header",
        );
        Fixture::rejects("empty", &[], "read the vector index header");
    }

    #[test]
    fn looks_up_the_best_cosine_of_given_procedures_in_a_repository() {
        let row = |rowid: i64, repo: &str, weights: &[(usize, f32)]| {
            (
                rowid,
                repo.to_owned(),
                Quantized::new(&direction(weights)).to_bytes(),
            )
        };
        let rows = vec![
            row(1, "repo-a", &[(1, 1.0)]),
            row(1, "repo-a", &[(0, 1.0)]),
            row(2, "repo-a", &[(0, 1.0)]),
            row(3, "repo-b", &[(0, 1.0)]),
            row(4, "", &[(0, 0.6), (1, 0.8)]),
        ];
        let path =
            std::env::temp_dir().join(format!("trodden-cosines-{}.index", std::process::id()));
        VectorIndex::build(&rows, &path).expect("index builds");
        let mut index = VectorIndex::open(&path).expect("index opens");

        let mut cosines = index
            .cosines(&direction(&[(0, 1.0)]), "repo-a", &[1, 3, 4, 9])
            .expect("index looks up cosines");
        fs::remove_file(&path).expect("index is removable");

        cosines.sort_by_key(|neighbor| neighbor.rowid);
        let rowids: Vec<i64> = cosines.iter().map(|neighbor| neighbor.rowid).collect();
        assert_eq!(
            rowids,
            [1, 4],
            "repo-b's and unindexed procedures are excluded"
        );
        assert!(
            (cosines[0].cosine - 1.0).abs() < 0.01,
            "best of a procedure's vectors"
        );
        assert!((cosines[1].cosine - 0.6).abs() < 0.01);
    }
}
