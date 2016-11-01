//! Logistic Regression

use linear_classifier::LinearClassifier;
use std::vec::Vec;

pub struct LogisticRegression {
    weight: Vec<f32>,
}

impl LogisticRegression {
    pub fn new(size: usize) -> LogisticRegression {
        LogisticRegression {
            weight: vec![0.0; size],
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl LinearClassifier for LogisticRegression {
    fn learn(&mut self, data: &[f32], label: bool, eta: f32) -> bool {
        assert_eq!(self.weight.len(), data.len());

        let c = 0.0001 / (data.len() as f32);
        let lb = if label { 1.0 } else { -1.0 };

        let mut ip = 0.0;
        for i in 0..self.weight.len() {
            ip += self.weight[i] * data[i];
        }

        let delta = (1.0 - sigmoid(lb * ip)) * (-lb);
        for i in 0..self.weight.len() {
            self.weight[i] = self.weight[i] - delta * data[i] - 2.0 * eta * c * self.weight[i];
        }

        ip * lb > 0.0
    }

    fn margin(&self, data: &[f32]) -> f32 {
        assert_eq!(self.weight.len(), data.len());

        let mut ip = 0.0;
        for i in 0..self.weight.len() {
            ip += self.weight[i] * data[i];
        }

        ip
    }
}
