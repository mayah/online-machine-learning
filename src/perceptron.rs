//! perceptron

use std::vec::Vec;

pub struct Perceptron {
    weight: Vec<f32>,
}

impl Perceptron {
    pub fn new(size: usize) -> Perceptron {
        Perceptron {
            weight: vec![0.0; size],
        }
    }

    /// Learn data with label.
    /// If the current Perceptron can predict the result correctly, true is returned.
    /// Otherwise, false is returned.
    pub fn learn(&mut self, data: &[f32], label: bool, eta: f32) -> bool {
        assert_eq!(self.weight.len(), data.len());

        let lb = if label { 1.0 } else { -1.0 };

        let mut ip = 0.0;
        for i in 0..self.weight.len() {
            ip += self.weight[i] * data[i];
        }

        if ip * lb > 0.0 {
            return true;
        }

        for i in 0..self.weight.len() {
            self.weight[i] += data[i] * lb * eta;
        }

        false
    }

    /// Returns how much the data is considered as positive class.
    pub fn predict(&self, data: &[f32]) -> f32 {
        assert_eq!(self.weight.len(), data.len());

        let mut ip = 0.0;
        for i in 0..self.weight.len() {
            ip += self.weight[i] * data[i];
        }

        ip
    }
}
