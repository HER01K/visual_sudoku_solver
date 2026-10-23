use crate::vector::Vector;

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

    pub fn outer_product_add(&mut self, vector_m: &Vector, vector_n: &Vector, scalar: f32) {
        for i in 0..self.m {
            for j in 0..self.n {
                self.data[i * self.n + j] += vector_m.data[i] * vector_n.data[j] * scalar;
            }
        }
    }
}