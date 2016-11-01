//! arow

use linear_classifier::LinearClassifier;
use std::vec::Vec;

pub struct Arow {
    mean: Vec<f32>,
    cov: Vec<f32>,
}

impl Arow {
    pub fn new(size: usize) -> Arow {
        Arow {
            mean: vec![0.0; size],
            cov: vec![1.0; size],
        }
    }

    fn confidence(&self, data: &[f32]) -> f32 {
        let mut m = 0.0;
        for i in 0..self.cov.len() {
            m += self.cov[i] * data[i] * data[i];
        }
        m
    }
}

impl LinearClassifier for Arow {
    fn learn(&mut self, data: &[f32], label: bool, eta: f32) -> bool {
        assert_eq!(self.mean.len(), data.len());

        let lb = if label { 1.0 } else { -1.0 };
        let m = self.margin(data);
        if m * lb >= 1.0 {
            return true;
        }

        let v = self.confidence(data);
        let beta = 1.0 / (v + eta);
        let alpha = (1.0 - lb * m) * beta;

        for i in 0..self.mean.len() {
            self.mean[i] += alpha * lb * self.cov[i] * data[i];
            self.cov[i] = 1.0 / ((1.0 / self.cov[i]) + data[i] * data[i] / eta);
        }

        m * lb > 0.0
    }

    fn margin(&self, data: &[f32]) -> f32 {
        assert_eq!(self.mean.len(), data.len());

        let mut ip = 0.0;
        for i in 0..self.mean.len() {
            ip += self.mean[i] * data[i];
        }

        ip
    }
}
