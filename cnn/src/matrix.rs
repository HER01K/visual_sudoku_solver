#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    pub(crate) data: Vec<f32>,
    pub m: usize,
    pub n: usize,
}

impl Matrix {
    pub fn new(value: impl Fn() -> f32, m: usize, n: usize) -> Self {
        Self {
            data: (0..m * n).into_iter().map(|_| value()).collect(),
            m,
            n
        }
    }
}