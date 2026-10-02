use std::{collections::BTreeSet, path::PathBuf};

use trodden_embed::{DIMS, Embedding, Quantized};
use trodden_recall::VectorIndex;

fn axis(index: usize) -> Embedding {
    let mut embedding = [0.0; DIMS];
    embedding[index] = 1.0;
    embedding
}

#[test]
fn finds_nearest_procedures_within_a_repository() {
    let rows = vec![
        (1, "repo-a".to_owned(), Quantized::new(&axis(0)).to_bytes()),
        (2, "repo-a".to_owned(), Quantized::new(&axis(1)).to_bytes()),
        (3, "repo-b".to_owned(), Quantized::new(&axis(0)).to_bytes()),
        (4, String::new(), Quantized::new(&axis(0)).to_bytes()),
    ];
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("nearest.index");
    VectorIndex::build(&rows, &path).expect("index builds");
    let mut index = VectorIndex::open(&path).expect("index opens");

    let neighbors = index
        .search(&axis(0), "repo-a", 10)
        .expect("index searches");

    let nearest: BTreeSet<i64> = neighbors[..2]
        .iter()
        .map(|neighbor| neighbor.rowid)
        .collect();
    assert_eq!(
        nearest,
        BTreeSet::from([1, 4]),
        "repo-a's and global procedures"
    );
    assert_eq!(neighbors.len(), 3, "repo-b's procedure is excluded");
    assert!(neighbors[0].cosine > 0.99 && neighbors[2].cosine.abs() < 0.01);
}
