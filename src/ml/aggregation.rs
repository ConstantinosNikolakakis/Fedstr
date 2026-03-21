use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use color_eyre::eyre::{Result, eyre};
use base64::{Engine as _, engine::general_purpose::STANDARD as base64};
use sha2::Digest;

use crate::protocol::{JobResult, ModelParams};

/// FedAvg Aggregator
/// 
/// Implements Federated Averaging (McMahan et al., 2017)
/// Simple parameter averaging across multiple workers.
pub struct FedAvgAggregator {
    _py_initialized: bool,
}

impl FedAvgAggregator {
    /// Create a new FedAvg aggregator
    pub fn new() -> Result<Self> {
        Ok(Self {
            _py_initialized: true,
        })
    }
    
    /// Aggregate model parameters from multiple DVMs using FedAvg
    /// 
    /// Handles both embedded and external storage:
    /// - Embedded models: Use base64 data directly  
    /// - External models: Download from URL first
    /// 
    /// # Arguments
    /// * `job_results` - Results from all DVMs in this round
    /// * `storage` - Storage handler for downloading external models
    /// 
    /// # Returns
    /// Aggregated model parameters (always embedded after aggregation)
    pub async fn aggregate(
        &self, 
        job_results: &[JobResult],
        storage: &crate::storage::ModelStorage,
    ) -> Result<ModelParams> {
        if job_results.is_empty() {
            return Err(eyre!("Cannot aggregate empty results"));
        }
        
        println!("  Preparing models for aggregation...");
        
        // Download external models and convert all to base64
        let mut model_params_base64 = Vec::new();
        let mut sample_counts = Vec::new();
        
        for (i, result) in job_results.iter().enumerate() {
            let base64_data = if let Some(ref params_b64) = result.model_params.params_base64 {
                // Embedded model - use directly
                println!("    DVM {}: Using embedded model ({}KB)", 
                    i, result.model_params.size_bytes / 1024);
                params_b64.clone()
            } else if let Some(ref url) = result.model_params.params_url {
                // External model - download first
                println!("    📥 DVM {}: Downloading model from {}", i, url);
                let model_bytes = storage.download_model(url).await?;
                
                println!("       Downloaded: {} bytes ({:.2} KB)", 
                    model_bytes.len(), 
                    model_bytes.len() as f64 / 1024.0
                );
                
                // Verify hash
                println!("       🔐 Verifying hash...");
                println!("          Expected: {}", result.model_params.params_hash);
                
                let mut hasher = sha2::Sha256::new();
                hasher.update(&model_bytes);
                let downloaded_hash = format!("{:x}", hasher.finalize());
                
                println!("          Computed: {}", downloaded_hash);
                
                if downloaded_hash != result.model_params.params_hash {
                    println!("       ❌ Hash mismatch!");
                    return Err(eyre!("Hash mismatch for model from {}", url));
                }
                
                println!("       ✅ Hash verified - Model integrity confirmed!");
                base64.encode(&model_bytes)
            } else {
                return Err(eyre!("Model params has neither base64 nor URL"));
            };
            
            model_params_base64.push(base64_data);
            sample_counts.push(result.training_samples);
        }
        
        // If only one result, just return it
        if model_params_base64.len() == 1 {
            println!("  Only one model, no aggregation needed");
            return Ok(ModelParams::from_base64(
                model_params_base64[0].clone(),
                job_results[0].model_params.params_hash.clone(),
                job_results[0].model_params.size_bytes,
            ));
        }
        
        // Perform weighted averaging in Python
        println!("  Running FedAvg on {} models...", model_params_base64.len());
        let aggregated_base64 = Python::with_gil(|py| {
            self.aggregate_python(py, &model_params_base64, &sample_counts)
        })?;
        
        // Calculate hash of aggregated model
        let aggregated_bytes = base64.decode(&aggregated_base64)?;
        let mut hasher = sha2::Sha256::new();
        hasher.update(&aggregated_bytes);
        let hash = hasher.finalize();
        let hash_hex = format!("{:x}", hash);
        
        println!("  ✓ Aggregation complete ({}KB)", aggregated_bytes.len() / 1024);
        
        Ok(ModelParams::from_base64(
            aggregated_base64,
            hash_hex,
            aggregated_bytes.len(),
        ))
    }
    
    /// Python implementation of weighted parameter averaging
    fn aggregate_python(
        &self,
        py: Python,
        model_params: &[String],
        sample_counts: &[usize],
    ) -> Result<String> {
        // Add python directory to path
        let sys = py.import("sys")?;
        let path = sys.getattr("path")?;
        path.call_method1("insert", (0, "python"))?;
        
        // Create Python aggregation script inline
        let code = r#"
def aggregate_models(model_params_b64_list, sample_counts):
    """
    Weighted averaging of PyTorch model parameters.
    
    Args:
        model_params_b64_list: List of base64-encoded model state dicts
        sample_counts: List of sample counts for each model
    
    Returns:
        Base64-encoded aggregated model
    """
    # Decode all models
    import torch
    import io
    import base64
    models = []
    for params_b64 in model_params_b64_list:
        params_bytes = base64.b64decode(params_b64)
        buffer = io.BytesIO(params_bytes)
        state_dict = torch.load(buffer, weights_only=True, map_location='cpu')
        models.append(state_dict)
    
    # Calculate total samples for weighted averaging
    total_samples = sum(sample_counts)
    weights = [count / total_samples for count in sample_counts]
    
    # Initialize aggregated state dict with first model structure
    aggregated = {}
    for key in models[0].keys():
        aggregated[key] = torch.zeros_like(models[0][key])
    
    # Weighted averaging
    for model, weight in zip(models, weights):
        for key in model.keys():
            aggregated[key] += weight * model[key]
    
    # Serialize aggregated model
    buffer = io.BytesIO()
    torch.save(aggregated, buffer)
    aggregated_bytes = buffer.getvalue()
    
    # Encode to base64 (no compression needed)
    return base64.b64encode(aggregated_bytes).decode('utf-8')
"#;
        
        // Execute the code
        // Get the aggregate_models function
        let locals = PyDict::new(py);
        py.run(code, None, Some(locals))?;
        let aggregate_fn = locals.get_item("aggregate_models")
            .ok_or_else(|| eyre!("Failed to get aggregate_models function"))?;
        
        // Convert Rust data to Python
        let py_params = PyList::new(py, model_params);
        let py_counts = PyList::new(py, sample_counts);
        
        // Call aggregation function
        let result = aggregate_fn.call1((py_params, py_counts))?;
        let aggregated_base64: String = result.extract()?;
        
        Ok(aggregated_base64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{TrainingMetrics, JobResult};
    
    #[test]
    #[ignore] // Requires Python environment and trained models
    fn test_aggregation_single_model() {
        let aggregator = FedAvgAggregator::new().unwrap();
        
        // Create a dummy result
        let model_params = ModelParams::from_base64(
            "dummy_base64".to_string(),
            "dummy_hash".to_string(),
            100,
        );
        
        let metrics = TrainingMetrics::new(0.5, 0.9, vec![0.5], vec![0.9], 1);
        
        let result = JobResult::new(
            "req1".to_string(),
            "job1".to_string(),
            "dvm1".to_string(),
            1,
            model_params.clone(),
            metrics,
            1000,
        );
        
        let aggregated = aggregator.aggregate(&[result]).unwrap();
        
        // Single model should return itself
        assert_eq!(aggregated.params_base64, model_params.params_base64);
    }
}
