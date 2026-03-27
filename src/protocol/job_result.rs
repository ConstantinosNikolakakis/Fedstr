use nostr::{Event, EventBuilder, Keys, Kind, Tag};
use serde::{Deserialize, Serialize};
use color_eyre::eyre::Result;

use super::KIND_JOB_RESULT;

/// Job Result Event (Kind 6000)
/// 
/// DVM publishes this after completing training to return the trained model.
/// Corresponds to Event Type 3 in the FEDSTR paper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobResult {
    /// Reference to the original job request
    pub job_request_id: String,
    
    /// Job ID from the request
    pub job_id: String,
    
    /// DVM's public key
    pub service_provider: String,
    
    /// Round number
    pub round: u32,
    
    /// Trained model parameters
    pub model_params: ModelParams,
    
    /// Training metrics for validation
    pub metrics: TrainingMetrics,
    
    /// Number of samples used in training
    pub training_samples: usize,
    
    /// Payment request (dummy for Phase 1)
    pub payment_invoice: Option<String>,
}

/// Container for model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParams {
    /// Phase 1: Direct embedding for tiny models (<1MB)
    /// Base64-encoded serialized model parameters
    pub params_base64: Option<String>,
    
    /// Phase 2: External storage for large models
    /// URL to download model parameters (NIP-94 compatible)
    pub params_url: Option<String>,
    
    /// SHA256 hash of parameters for verification
    pub params_hash: String,
    
    /// Size of model in bytes
    pub size_bytes: usize,
}

impl ModelParams {
    /// Create from base64-encoded parameters (embedded mode - small models)
    pub fn from_base64(params_base64: String, params_hash: String, size_bytes: usize) -> Self {
        Self {
            params_base64: Some(params_base64),
            params_url: None,
            params_hash,
            size_bytes,
        }
    }
    
    /// Create from external URL (external storage mode - large models)
    pub fn from_url(params_url: String, params_hash: String, size_bytes: usize) -> Self {
        Self {
            params_base64: None,
            params_url: Some(params_url),
            params_hash,
            size_bytes,
        }
    }
    
    /// Check if model is embedded or uses external storage
    pub fn is_embedded(&self) -> bool {
        self.params_base64.is_some()
    }
    
    /// Get the storage location description (for logging)
    pub fn storage_info(&self) -> String {
        if self.is_embedded() {
            format!("Embedded ({}KB)", self.size_bytes / 1024)
        } else if let Some(ref url) = self.params_url {
            format!("External: {}", url)
        } else {
            "Unknown".to_string()
        }
    }
}

/// Training metrics for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub final_loss: f64,
    pub final_accuracy: f64,
    pub loss_history: Vec<f64>,
    pub accuracy_history: Vec<f64>,
    pub epochs_completed: u32,
    pub val_loss: f64,           // validation loss
    pub val_perplexity: f64,     // exp(val_loss)
}

impl JobResult {
    /// Create a new job result
    pub fn new(
        job_request_id: String,
        job_id: String,
        service_provider: String,
        round: u32,
        model_params: ModelParams,
        metrics: TrainingMetrics,
        training_samples: usize,
    ) -> Self {
        Self {
            job_request_id,
            job_id,
            service_provider,
            round,
            model_params,
            metrics,
            training_samples,
            payment_invoice: None,
        }
    }
    
    /// Convert to NOSTR event
    pub fn to_event(&self, keys: &Keys) -> Result<Event> {
        let content = serde_json::to_string(self)?;
        
        let tags = vec![
            Tag::parse(&["e", &self.job_request_id])?,  // Reference to request
            Tag::parse(&["job_id", &self.job_id])?,
            Tag::parse(&["round", &self.round.to_string()])?,
            Tag::parse(&["status", "success"])?,
        ];
        
        Ok(EventBuilder::new(Kind::Custom(KIND_JOB_RESULT), content, tags)
            .to_event(keys)?)
    }
    
    /// Parse from NOSTR event
    pub fn from_event(event: &Event) -> Result<Self> {
        if event.kind.as_u16() != KIND_JOB_RESULT {
            return Err(color_eyre::eyre::eyre!(
                "Invalid event kind: expected {}, got {}",
                KIND_JOB_RESULT,
                event.kind.as_u16()
            ));
        }
        
        let job_result: JobResult = serde_json::from_str(&event.content)?;
        Ok(job_result)
    }
}

impl TrainingMetrics {
    /// Create new training metrics
    pub fn new(
        final_loss: f64,
        final_accuracy: f64,
        loss_history: Vec<f64>,
        accuracy_history: Vec<f64>,
        epochs_completed: u32,
        val_loss: f64,
        val_perplexity: f64,
    ) -> Self {
        Self {
            final_loss,
            final_accuracy,
            loss_history,
            accuracy_history,
            epochs_completed,
            val_loss,
            val_perplexity,
        }
    }
    
    /// Check if training improved (loss decreased)
    pub fn has_improved(&self, previous_loss: f64, tolerance: f64) -> bool {
        self.final_loss < previous_loss * (1.0 + tolerance)
    }
    
    /// Get average loss
    pub fn average_loss(&self) -> f64 {
        if self.loss_history.is_empty() {
            self.final_loss
        } else {
            self.loss_history.iter().sum::<f64>() / self.loss_history.len() as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_model_params_base64() {
        let params = ModelParams::from_base64(
            "abc123".to_string(),
            "hash123".to_string(),
            1024,
        );
        
        assert!(params.is_embedded());
        assert!(!params.is_external());
        assert_eq!(params.size_bytes, 1024);
    }
    
    #[test]
    fn test_model_params_url() {
        let params = ModelParams::from_url(
            "https://example.com/model.pth".to_string(),
            "hash456".to_string(),
            1024000,
        );
        
        assert!(!params.is_embedded());
        assert!(params.is_external());
    }
    
    #[test]
    fn test_training_metrics() {
        let metrics = TrainingMetrics::new(
            0.25,
            0.95,
            vec![0.5, 0.35, 0.25],
            vec![0.85, 0.90, 0.95],
            3,
            0.25,   // val_loss
            1.28,   // val_perplexity
        );
        
        assert!(metrics.has_improved(0.5, 0.1));
        assert!(!metrics.has_improved(0.2, 0.1));
        assert_eq!(metrics.epochs_completed, 3);
    }
}
