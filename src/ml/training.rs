use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use color_eyre::eyre::{Result, eyre};

use crate::protocol::{ModelParams, TrainingMetrics};

/// Result from Python training
#[derive(Debug, Clone)]
pub struct TrainingResult {
    pub model_params: ModelParams,
    pub metrics: TrainingMetrics,
    pub training_samples: usize,
}

/// Bridge to Python PyTorch training
pub struct PyTorchTrainer {
    _py_initialized: bool,
}

impl PyTorchTrainer {
    /// Create a new PyTorch trainer
    pub fn new() -> Result<Self> {
        // PyO3 auto-initializes Python with the auto-initialize feature
        Ok(Self {
            _py_initialized: true,
        })
    }
    
    /// Train a tiny model on a data subset
    /// 
    /// # Arguments
    /// * `dataset` - Dataset name ("mnist" or "fashion-mnist")
    /// * `start_idx` - Starting index for data subset
    /// * `end_idx` - Ending index for data subset
    /// * `epochs` - Number of training epochs
    /// * `batch_size` - Batch size for training
    /// * `initial_params` - Optional base64-encoded initial parameters
    pub fn train_tiny_model(
        &self,
        algorithm: &str,
        dataset: &str,
        start_idx: usize,
        end_idx: usize,
        epochs: u32,
        batch_size: u32,
        initial_params: Option<String>,
        round_num: u32,
    ) -> Result<TrainingResult> {
        Python::with_gil(|py| {
            // Import the training module
            let sys = py.import("sys")?;
            let path = sys.getattr("path")?;
            let algo_path = format!("/opt/fedstr/algorithms/{}", algorithm);
            path.call_method1("insert", (0, algo_path))?;

            let train_module = PyModule::import(py, "train")?;
            
            // Call train_model function
            let kwargs = PyDict::new(py);
            kwargs.set_item("dataset_name", dataset)?;
            kwargs.set_item("start_idx", start_idx)?;
            kwargs.set_item("end_idx", end_idx)?;
            kwargs.set_item("epochs", epochs)?;
            kwargs.set_item("batch_size", batch_size)?;
            kwargs.set_item("round_num", round_num)?;
            
            if let Some(params) = initial_params {
                kwargs.set_item("initial_params", params)?;
            }
            
            let result = train_module.call_method("train_model", (), Some(kwargs))?;
            
            // Extract results
            let model_base64: String = result.get_item("model_base64")?.extract()?;
            let model_hash: String = result.get_item("model_hash")?.extract()?;
            let size_bytes: usize = result.get_item("size_bytes")?.extract()?;
            
            let final_loss: f64 = result.get_item("final_loss")?.extract()?;
            let final_accuracy: f64 = result.get_item("final_accuracy")?.extract()?;
            let loss_history: Vec<f64> = result.get_item("loss_history")?.extract()?;
            let accuracy_history: Vec<f64> = result.get_item("accuracy_history")?.extract()?;
            let epochs_completed: u32 = result.get_item("epochs_completed")?.extract()?;
            let training_samples: usize = result.get_item("training_samples")?.extract()?;
            
            // Create ModelParams
            let model_params = ModelParams::from_base64(
                model_base64,
                model_hash,
                size_bytes,
            );
            
            let val_loss: f64 = result.get_item("val_loss")
                .ok().and_then(|v| v.extract().ok()).unwrap_or(final_loss);
            let val_perplexity: f64 = result.get_item("val_perplexity")
                .ok().and_then(|v| v.extract().ok()).unwrap_or(final_loss.exp());

            // Create TrainingMetrics
            let metrics = TrainingMetrics::new(
                final_loss,
                final_accuracy,
                loss_history,
                accuracy_history,
                epochs_completed,
                val_loss,
                val_perplexity,
            );
            
            Ok(TrainingResult {
                model_params,
                metrics,
                training_samples,
            })
        })
        .map_err(|e: PyErr| eyre!("Python training error: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    #[ignore] // Requires Python environment
    fn test_tiny_model_training() {
        let trainer = PyTorchTrainer::new().unwrap();
        let result = trainer.train_tiny_model(
            &dataset,
            0,
            1000,  // Small subset for testing
            2,     // 2 epochs
            32,
            None,
        ).unwrap();
        
        assert!(result.model_params.is_embedded());
        assert_eq!(result.metrics.epochs_completed, 2);
        assert!(result.metrics.final_loss > 0.0);
    }
}
