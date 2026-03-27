use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use color_eyre::eyre::{Result, eyre};
use base64::{Engine as _, engine::general_purpose::STANDARD as base64};
use sha2::Digest;

use crate::protocol::{JobResult, ModelParams};


/// For DiLoCo, includes the momentum state v_t to persist across rounds.
/// For FedAvg, v_t is always None.
pub struct AggregateResult {
    pub theta_global: ModelParams,
    pub v_t: Option<ModelParams>,  // DiLoCo momentum state
}

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
        theta_global: Option<String>,
        v_t: Option<String>,       
    ) -> Result<AggregateResult> {
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
            return Ok(AggregateResult {
              theta_global: ModelParams::from_base64(
                  model_params_base64[0].clone(),
                  job_results[0].model_params.params_hash.clone(),
                  job_results[0].model_params.size_bytes,
              ),
              v_t: None,
            });
        }
        
        // Perform weighted averaging in Python
        // For DiLoCo: resolve θ_global to base64
        // current_params is a URL — download and encode it
        let algorithm = std::env::var("ALGORITHM")
            .unwrap_or_else(|_| "fedavg".to_string());
        let theta_global_b64 = match (&theta_global, algorithm.as_str()) {
            (Some(url), "diloco") => {
                println!("  📥 Downloading θ_global for DiLoCo outer step: {}", url);
                let bytes = storage.download_model(url).await?;
                println!("     ✓ Downloaded θ_global ({} bytes)", bytes.len());
                Some(base64.encode(&bytes))
            },
            _ => None,
        };

        let v_t_b64 = match (&v_t, algorithm.as_str()) {
            (Some(url), "diloco") => {
                println!("  📥 Downloading v_t momentum state: {}", url);
                let bytes = storage.download_model(url).await?;
                println!("     ✓ Downloaded v_t ({} bytes)", bytes.len());
                Some(base64.encode(&bytes))
            },
            _ => None,
        };

        println!("  Running {} aggregation on {} models...", algorithm, model_params_base64.len());
        let aggregated_json = Python::with_gil(|py| {
            self.aggregate_python(py, &model_params_base64, &sample_counts, &theta_global_b64, &v_t_b64)
        })?;
        
        // Parse JSON response — both FedAvg and DiLoCo return JSON now
        let parsed: serde_json::Value = serde_json::from_str(&aggregated_json)
            .map_err(|e| eyre!("Failed to parse aggregate JSON: {}", e))?;

        let theta_b64 = parsed["theta_global"].as_str()
            .ok_or_else(|| eyre!("Missing theta_global in aggregate result"))?
            .to_string();

        let aggregated_bytes = base64.decode(&theta_b64)?;
        let mut hasher = sha2::Sha256::new();
        hasher.update(&aggregated_bytes);
        let hash_hex = format!("{:x}", hasher.finalize());
        println!("  ✓ Aggregation complete ({}KB)", aggregated_bytes.len() / 1024);

        let theta_global_out = ModelParams::from_base64(theta_b64, hash_hex, aggregated_bytes.len());

        // Extract v_t if present (DiLoCo only)
        let v_t_out = if let Some(vt_str) = parsed["v_t"].as_str() {
            let vt_bytes = base64.decode(vt_str)?;
            let mut h = sha2::Sha256::new();
            h.update(&vt_bytes);
            let vt_hash = format!("{:x}", h.finalize());
            Some(ModelParams::from_base64(vt_str.to_string(), vt_hash, vt_bytes.len()))
        } else {
            None
        };

        Ok(AggregateResult { theta_global: theta_global_out, v_t: v_t_out })
    }
    
    /// Python implementation of weighted parameter averaging
    fn aggregate_python(
        &self,
        py: Python,
        model_params: &[String],
        sample_counts: &[usize],
        theta_global: &Option<String>,
        v_t: &Option<String>,      
    ) -> Result<String> {
        // Dispatch to the correct algorithm's aggregate.py
        let algorithm = std::env::var("ALGORITHM")
            .unwrap_or_else(|_| "fedavg".to_string());
        let script_path = format!("/opt/fedstr/algorithms/{}/aggregate.py", algorithm);
        let code = std::fs::read_to_string(&script_path)
            .map_err(|e| eyre!("Failed to read {}: {}", script_path, e))?;

        // Add algorithm directory to Python path
        let sys = py.import("sys")?;
        let path = sys.getattr("path")?;
        let algo_path = format!("/opt/fedstr/algorithms/{}", algorithm);
        path.call_method1("insert", (0, algo_path))?;

        // Load and execute the aggregation script
        let locals = PyDict::new(py);
        py.run(&code, None, Some(locals))?;
        let aggregate_fn = locals.get_item("aggregate_models")
            .ok_or_else(|| eyre!("Failed to get aggregate_models function"))?;
        
        // Convert Rust data to Python
        let py_params = PyList::new(py, model_params);
        let py_counts = PyList::new(py, sample_counts);

        // Call aggregation function — pass theta_global for DiLoCo, None for FedAvg
        let result = match (theta_global, v_t) {
        (Some(tg), Some(vt)) => aggregate_fn.call1((py_params, py_counts, tg.as_str(), vt.as_str()))?,
        (Some(tg), None)     => aggregate_fn.call1((py_params, py_counts, tg.as_str()))?,
        _                    => aggregate_fn.call1((py_params, py_counts))?,
        };
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
        
        let metrics = TrainingMetrics::new(0.5, 0.9, vec![0.5], vec![0.9], 1, 0.5, 1.65);
        
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
        assert_eq!(aggregated.theta_global.params_base64, model_params.params_base64);
    }
}
