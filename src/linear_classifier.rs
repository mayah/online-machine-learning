pub trait LinearClassifier {

    /// Learn data with label.
    /// If the current Perceptron can predict the result correctly, true is returned.
    /// Otherwise, false is returned.
    fn learn(&mut self, data: &[f32], label: bool, eta: f32) -> bool;

    /// Returns how much the data is considered as positive class.
    fn margin(&self, data: &[f32]) -> f32;

    fn predict(&self, data: &[f32]) -> bool {
        self.margin(data) > 0.0
    }
}
