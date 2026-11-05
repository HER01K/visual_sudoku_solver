use rand::prelude::SliceRandom;
use crate::io::{load_dataset, Dataset};
use crate::matrix::Matrix;
use crate::vector::Vector;

mod io;
mod matrix;
mod vector;

const INPUT_SIZE_X: usize = 28;
const INPUT_SIZE_Y: usize = 28;
const NUM_INPUTS: usize = INPUT_SIZE_X * INPUT_SIZE_Y;
const NUM_OUTPUTS: usize = 10;
const NUM_EPOCHS: usize = 10;
const LEARNING_RATE: f32 = 0.01;

fn he_uniform_init(fan_in: usize) -> f32 {
    let limit = (6.0 / fan_in as f32).sqrt();
    (rand::random::<f32>() * 2.0 * limit) - limit
}

struct NNLayer {
    size: usize,
    prev_size: usize,
    weights: Matrix,
    biases: Vector,
    activation: Vector,
}

impl NNLayer {
    fn new(size: usize, prev_size: usize) -> Self {
        Self {
            size,
            prev_size,
            weights: Matrix::new(|| he_uniform_init(prev_size), size, prev_size),
            biases: Vector::new(|| 0.0, size),
            activation: Vector::new(|| 0.0, size),
        }
    }
}

fn create_layers(layer_sizes: &[usize]) -> Vec<NNLayer> {
    layer_sizes
        .windows(2)
        .map(|sizes| NNLayer::new(sizes[1], sizes[0]))
        .collect()
}

fn forward_propagation(layers: &mut [NNLayer], input: &Vector) {
    let len = layers.len();

    for i in 0..layers.len() {
        let (before, after) = layers.split_at_mut(i);
        let layer_input = before.last().map_or(input, |prev| &prev.activation);

        // Compute activation
        after[0].activation.matrix_vector_mult(&after[0].weights, layer_input);

        // Add bias
        after[0].activation.add(&after[0].biases);

        // Apply ReLU
        if i < len - 1 {
            after[0].activation.relu();
        } else {
            after[0].activation.softmax();
        }
    }
}

fn backward_propagation(layers: &mut [NNLayer], input: &Vector, expected: &Vector, learning_rate: f32) {
    let len = layers.len();
    let mut layer_errors: Vec<Vector> = Vec::with_capacity(len);

    for layer in layers.iter() {
        layer_errors.push(layer.activation.clone());
    }

    layer_errors[len - 1].sub(expected);

    for i in (0..=len - 2).into_iter().rev() {
        let (before, after) = layer_errors.split_at_mut(i + 1);
        let mut activation_func_error = layers[i].activation.clone();

        before[i].matrix_transpose_vector_mult(&layers[i + 1].weights, &after[0]);
        activation_func_error.relu_d();
        before[i].hadamard_product(&activation_func_error);

        layers[i + 1].weights.outer_product_add(&after[0], &layers[i].activation, -learning_rate);
        layers[i + 1].biases.add_scaled(&after[0], -learning_rate);
    }

    layers[0].weights.outer_product_add(&layer_errors[0], input, -learning_rate);
    layers[0].biases.add_scaled(&layer_errors[0], -learning_rate);
}

fn evaluate(layers: &mut [NNLayer], data: &Dataset) -> f32 {
    let mut correct = 0;

    for (img, &label) in data.images.iter().zip(&data.labels) {
        forward_propagation(
            layers,
            &Vector { data: img.iter().map(|&p| p as f32 / 255.0).collect() }
        );

        let output = layers
            .last()
            .unwrap()
            .activation
            .data
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap();

        if output == label as usize {
            correct += 1;
        }
    }

    correct as f32 / data.images.len() as f32
}

fn main() {
    let train = load_dataset("data/train").expect("failed to load train dataset");
    let test = load_dataset("data/test").expect("failed to load test dataset");
    println!("{} train and {} test examples", train.images.len(), test.images.len());

    let mut layers = create_layers(&[NUM_INPUTS, 128, 64, NUM_OUTPUTS]);

    // Train model and eval at each epoch
    for epoch in 1..=NUM_EPOCHS {
        let mut order: Vec<usize> = (0..train.images.len()).collect();
        order.shuffle(&mut rand::rng());

        for i in order {
            let input = Vector { data: (&train.images[i]).iter().map(|&p| p as f32 / 255.0).collect() };
            let mut expected = Vector::new(|| 0.0, NUM_OUTPUTS);
            expected.data[train.labels[i] as usize] = 1.0;

            forward_propagation(&mut layers, &input);
            backward_propagation(&mut layers, &input, &expected, LEARNING_RATE);
        }

        let accuracy = evaluate(&mut layers, &test);
        println!("epoch {epoch}, test accuracy: {:.2}%", accuracy * 100.0);
    }
}

#[cfg(test)]
mod tests {
    use crate::{create_layers, forward_propagation, he_uniform_init, NUM_INPUTS, NUM_OUTPUTS, backward_propagation};
    use crate::vector::Vector;

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-6, "expected {expected:?}, got {actual:?}");
        }
    }

    #[test]
    fn forward_propagation_small_network() {
        let mut layers = create_layers(&[3, 2, 2]);

        layers[0].weights.data = vec![
            1.0, 1.0, -0.5,
            -3.0, 1.0, 0.0
        ];
        layers[0].biases = Vector::new(|| 0.5, 2);

        layers[1].weights.data = vec![
            1.0, 3.0,
            0.5, 2.0
        ];
        layers[1].biases = Vector::new(|| 0.0, 2);

        forward_propagation(&mut layers, &Vector::new(|| 1.0, 3));

        assert_close(&layers[0].activation.data, &[2.0, 0.0]);

        let e = std::f32::consts::E;
        assert_close(&layers[1].activation.data, &[e / (e + 1.0), 1.0 / (e + 1.0)]);
    }

    #[test]
    fn he_uniform_init_stays_within_limit() {
        let fan_in = 784;
        let limit = (6.0 / fan_in as f32).sqrt();

        for _ in 0..10_000 {
            let x = he_uniform_init(fan_in);
            assert!(x.abs() <= limit, "{x} is outside +-{limit}");
        }
    }

    #[test]
    fn forward_propagation_large_size_outputs_probabilities() {
        let mut layers = create_layers(&[NUM_INPUTS, 128, 64, NUM_OUTPUTS]);

        forward_propagation(&mut layers, &Vector::new(|| rand::random::<f32>(), NUM_INPUTS));

        let output = &layers.last().unwrap().activation.data;
        assert_eq!(output.len(), NUM_OUTPUTS);
        for &p in output {
            assert!(p.is_finite() && (0.0..=1.0).contains(&p), "bad output {output:?}");
        }

        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "outputs sum to {sum}");
    }

    #[test]
    fn backward_propagation_small_network() {
        let mut layers = create_layers(&[3, 2, 2]);

        layers[0].weights.data = vec![
            1.0, 1.0, -0.5,
            -3.0, 1.0, 0.0
        ];
        layers[0].biases = Vector::new(|| 0.5, 2);

        layers[1].weights.data = vec![
            1.0, 3.0,
            0.5, 2.0
        ];
        layers[1].biases = Vector::new(|| 0.0, 2);

        let input = Vector::new(|| 1.0, 3);
        let expected = Vector { data: vec![1.0, 0.0] };

        forward_propagation(&mut layers, &input);
        backward_propagation(&mut layers, &input, &expected, 1.0);

        let q = 1.0 / (std::f32::consts::E + 1.0);

        assert_close(&layers[1].weights.data, &[
            1.0 + 2.0 * q, 3.0,
            0.5 - 2.0 * q, 2.0
        ]);
        assert_close(&layers[1].biases.data, &[q, -q]);

        assert_close(&layers[0].weights.data, &[
            1.0 + 0.5 * q, 1.0 + 0.5 * q, -0.5 + 0.5 * q,
            -3.0, 1.0, 0.0
        ]);
        assert_close(&layers[0].biases.data, &[0.5 + 0.5 * q, 0.5]);
    }
}
