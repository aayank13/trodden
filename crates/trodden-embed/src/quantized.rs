use crate::{DIMS, Embedding};

pub const QUANTIZED_LEN: usize = 4 + DIMS;

pub const CODE_LEN: usize = DIMS / 8;

#[derive(Debug, Clone, PartialEq)]
pub struct Quantized {
    scale: f32,
    values: [i8; DIMS],
}

impl Quantized {
    pub fn new(embedding: &Embedding) -> Self {
        let max = embedding
            .iter()
            .fold(0.0_f32, |max, value| max.max(value.abs()));
        let scale = if max > 0.0 { max / 127.0 } else { 1.0 };
        #[expect(clippy::cast_possible_truncation, reason = "clamped to i8 range")]
        let values = embedding.map(|value| (value / scale).round().clamp(-127.0, 127.0) as i8);
        Self { scale, values }
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        bytes.try_into().ok().map(Self::from_array)
    }

    pub fn from_array(bytes: &[u8; QUANTIZED_LEN]) -> Self {
        let scale = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let mut values = [0_i8; DIMS];
        for (value, byte) in values.iter_mut().zip(&bytes[4..]) {
            *value = i8::from_le_bytes([*byte]);
        }
        Self { scale, values }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(QUANTIZED_LEN);
        bytes.extend_from_slice(&self.scale.to_le_bytes());
        bytes.extend(self.values.iter().map(|value| value.to_le_bytes()[0]));
        bytes
    }

    pub fn cosine(&self, query: &Embedding) -> f32 {
        self.values
            .iter()
            .zip(query)
            .map(|(value, q)| f32::from(*value) * q)
            .sum::<f32>()
            * self.scale
    }

    pub fn code(&self) -> [u8; CODE_LEN] {
        Self::code_of(self.values.iter().map(|value| *value > 0))
    }

    pub fn query_code(query: &Embedding) -> [u8; CODE_LEN] {
        Self::code_of(query.iter().map(|value| *value > 0.0))
    }

    fn code_of(positive: impl Iterator<Item = bool>) -> [u8; CODE_LEN] {
        let mut code = [0; CODE_LEN];
        for (index, bit) in positive.enumerate() {
            if bit {
                code[index / 8] |= 1 << (index % 8);
            }
        }
        code
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(seed: usize) -> Embedding {
        let raw: Embedding = std::array::from_fn(|i| (((i * 7 + seed * 13) % 17) as f32) - 8.0);
        let norm = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
        raw.map(|v| v / norm)
    }

    #[test]
    fn round_trips_through_bytes() {
        let quantized = Quantized::new(&unit(3));

        assert_eq!(
            Quantized::from_bytes(&quantized.to_bytes()),
            Some(quantized)
        );
        assert_eq!(Quantized::from_bytes(&[0; 3]), None);
    }

    #[test]
    fn preserves_cosine_similarity() {
        let (a, b) = (unit(1), unit(2));
        let exact: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();

        let approximate = Quantized::new(&a).cosine(&b);

        assert!(
            (exact - approximate).abs() < 0.01,
            "{exact} vs {approximate}"
        );
        assert!((Quantized::new(&a).cosine(&a) - 1.0).abs() < 0.01);
    }
}
