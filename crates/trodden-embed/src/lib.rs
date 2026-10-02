mod pack;
mod quantized;
mod tokenizer;

pub use pack::{Embedder, ModelPack};
pub use quantized::{CODE_LEN, QUANTIZED_LEN, Quantized};

pub const DIMS: usize = 256;

pub type Embedding = [f32; DIMS];
