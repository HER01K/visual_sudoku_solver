use crate::matrix::Matrix;

#[derive(Clone, Debug, PartialEq)]
pub struct Vector {
    pub(crate) data: Vec<f32>,
}

impl Vector {
    pub fn new(value: impl Fn() -> f32, size: usize) -> Self {
        Self {
            data: (0..size).into_iter().map(|_| value()).collect()
        }
    }

    pub fn add(&mut self, other: &Vector) {
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += b
        }
    }

    pub fn relu(&mut self) {
        for x in &mut self.data {
            if *x < 0.0 {
                *x = 0.0;
            }
        }
    }

    pub fn softmax(&mut self) {
        let denominator: f32 = self.data.iter().map(|x| x.exp()).sum();

        for x in &mut self.data {
            *x = x.exp() / denominator;
        }
    }

    pub fn matrix_vector_mult(&mut self, matrix: &Matrix, vector: &Vector) {
        for (i, x) in self.data.iter_mut().enumerate() {
            *x = 0.0;

            for j in 0..matrix.n {
                *x += matrix.data[i * matrix.n + j] * vector.data[j];
            }
        }
    }


}
