//! support vector machine

use linear_classifier::LinearClassifier;
use std::vec::Vec;

/// Support vector machine. This is for Linear classifier.
pub struct SVM {
    weight: Vec<f32>,
}

impl SVM {
    pub fn new(size: usize) -> SVM {
        SVM {
            weight: vec![0.0; size],
        }
    }
}

impl LinearClassifier for SVM {
    fn learn(&mut self, data: &[f32], label: bool, eta: f32) -> bool {
        assert_eq!(self.weight.len(), data.len());

        let c = 0.0001 / (data.len() as f32);
        let lb = if label { 1.0 } else { -1.0 };

        let mut ip = 0.0;
        for i in 0..self.weight.len() {
            ip += self.weight[i] * data[i];
        }

        if lb * ip <= 1.0 {
            for i in 0..self.weight.len() {
                self.weight[i] = self.weight[i] + eta * lb * data[i] - 2.0 * eta * c * self.weight[i];
            }
        } else {
            for i in 0..self.weight.len() {
                self.weight[i] = self.weight[i] - 2.0 * eta * c * self.weight[i];
            }
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
