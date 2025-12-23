/// Machine Learning Module
/// 
/// Handles training, aggregation, and validation for federated learning.

pub mod training;
pub mod aggregation;
pub mod validation;

pub use training::{PyTorchTrainer, TrainingResult};
pub use aggregation::FedAvgAggregator;
pub use validation::{Validator, ValidationResult};
