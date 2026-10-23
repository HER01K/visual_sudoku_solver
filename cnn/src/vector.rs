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

    pub fn add_scaled(&mut self, other: &Vector, scalar: f32) {
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += b * scalar;
        }
    }

    pub fn sub(&mut self, other: &Vector) {
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a -= b;
        }
    }

    pub fn relu(&mut self) {
        for x in &mut self.data {
            if *x < 0.0 {
                *x = 0.0;
            }
        }
    }

    pub fn relu_d(&mut self) {
        for x in &mut self.data {
            if *x <= 0.0 {
                *x = 0.0;
            } else {
                *x = 1.0;
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

    pub fn matrix_transpose_vector_mult(&mut self, matrix: &Matrix, vector: &Vector) {
        for x in &mut self.data {
            *x = 0.0;
        }

        for i in 0..matrix.m {
            for j in 0..matrix.n {
                self.data[j] += matrix.data[i * matrix.n + j] * vector.data[i];
            }
        }
    }

    pub fn hadamard_product(&mut self, other: &Vector) {
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a *= b;
        }
    }
}
