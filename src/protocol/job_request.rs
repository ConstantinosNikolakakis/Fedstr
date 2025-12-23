use nostr::{Event, EventBuilder, Keys, Kind, Tag};
use serde::{Deserialize, Serialize};
use color_eyre::eyre::Result;

use super::KIND_JOB_REQUEST;

/// Job Request Event (Kind 8000)
/// 
/// Customer publishes this to request federated learning training from DVMs.
/// Corresponds to Event Type 2 in the FEDSTR paper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRequest {
    /// Unique job ID (UUID)
    pub job_id: String,
    
    /// Dataset name (e.g., "mnist", "fashion-mnist")
    pub dataset: String,
    
    /// Data split for this specific DVM
    pub data_split: DataSplit,
    
    /// Model configuration
    pub model_type: String,          // "tiny_linear" or "cnn"
    pub epochs: u32,
    pub batch_size: u32,
    
    /// Current model parameters (None for first round, Some for subsequent rounds)
    /// Phase 1: base64 encoded bytes
    /// Phase 2: URL to download
    pub current_model_params: Option<String>,
    
    /// SHA256 hash of current model parameters for verification
    /// DVMs MUST verify this hash matches the downloaded/decoded model
    /// None for first round (no previous model), Some for rounds 2+
    pub current_model_params_hash: Option<String>,
    
    /// Federated learning round info
    pub round: u32,
    pub total_rounds: u32,
    
    /// Target DVM public key (npub or hex)
    pub service_provider: String,
    
    /// Execution constraints
    pub max_execution_time_secs: u64,
    pub timeout_secs: u64,
    
    /// Payment amount (dummy for Phase 1)
    pub payment_msats: u64,
}

/// Specifies which portion of the dataset this DVM should use
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSplit {
    /// Starting index in the dataset
    pub start_idx: usize,
    
    /// Ending index in the dataset (exclusive)
    pub end_idx: usize,
    
    /// Total dataset size (for validation)
    pub total_size: usize,
}

impl JobRequest {
    /// Create a new job request
    pub fn new(
        job_id: String,
        dataset: String,
        data_split: DataSplit,
        model_type: String,
        epochs: u32,
        batch_size: u32,
        service_provider: String,
        round: u32,
        total_rounds: u32,
    ) -> Self {
        Self {
            job_id,
            dataset,
            data_split,
            model_type,
            epochs,
            batch_size,
            current_model_params: None,
            current_model_params_hash: None,
            round,
            total_rounds,
            service_provider,
            max_execution_time_secs: 600, // 10 minutes default
            timeout_secs: 900,             // 15 minutes timeout
            payment_msats: 1000,           // Dummy payment
        }
    }
    
    /// Set the current model parameters with hash for subsequent rounds
    pub fn with_model_params(mut self, params: String, hash: String) -> Self {
        self.current_model_params = Some(params);
        self.current_model_params_hash = Some(hash);
        self
    }
    
    /// Convert to NOSTR event
    pub fn to_event(&self, keys: &Keys) -> Result<Event> {
        let content = serde_json::to_string(self)?;
        
        let tags = vec![
            Tag::parse(&["p", &self.service_provider])?,
            Tag::parse(&["task", "federated_learning"])?,
            Tag::parse(&["round", &self.round.to_string()])?,
            Tag::parse(&["job_id", &self.job_id])?,
            Tag::parse(&["dataset", &self.dataset])?,
            Tag::parse(&["model", &self.model_type])?,
        ];
        
        Ok(EventBuilder::new(Kind::Custom(KIND_JOB_REQUEST), content, tags)
            .to_event(keys)?)
    }
    
    /// Parse from NOSTR event
    pub fn from_event(event: &Event) -> Result<Self> {
        if event.kind.as_u16() != KIND_JOB_REQUEST {
            return Err(color_eyre::eyre::eyre!(
                "Invalid event kind: expected {}, got {}",
                KIND_JOB_REQUEST,
                event.kind.as_u16()
            ));
        }
        
        let job_request: JobRequest = serde_json::from_str(&event.content)?;
        Ok(job_request)
    }
    
    /// Get the event ID of this job request (if it's been published)
    pub fn event_id(&self, keys: &Keys) -> Result<String> {
        let event = self.to_event(keys)?;
        Ok(event.id.to_hex())
    }
    
    /// Verify that model parameters match the expected hash
    /// 
    /// DVMs should call this after downloading/decoding the model to ensure integrity.
    /// 
    /// # Arguments
    /// * `model_bytes` - The raw bytes of the model (after base64 decode or download)
    /// 
    /// # Returns
    /// * `Ok(true)` - Hash matches, model is valid
    /// * `Ok(false)` - No hash to verify (round 1)
    /// * `Err` - Hash mismatch, model is corrupted or tampered with
    pub fn verify_model_hash(&self, model_bytes: &[u8]) -> Result<bool> {
        use sha2::{Sha256, Digest};
        
        // Round 1 has no previous model to verify
        if self.current_model_params_hash.is_none() {
            println!("  ℹ️  Round {} - No previous model to verify (initial round)", self.round);
            return Ok(false);
        }
        
        let expected_hash = self.current_model_params_hash.as_ref().unwrap();
        
        println!("  🔐 Verifying model integrity...");
        println!("     Model size: {} bytes ({:.2} KB)", 
            model_bytes.len(), 
            model_bytes.len() as f64 / 1024.0
        );
        println!("     Expected hash: {}", expected_hash);
        
        // Calculate actual hash
        let mut hasher = Sha256::new();
        hasher.update(model_bytes);
        let actual_hash = format!("{:x}", hasher.finalize());
        
        println!("     Computed hash: {}", actual_hash);
        
        if actual_hash != *expected_hash {
            println!("  ❌ HASH MISMATCH - Model verification FAILED!");
            println!("     This indicates the model was tampered with or corrupted.");
            return Err(color_eyre::eyre::eyre!(
                "Model hash mismatch! Expected: {}, Got: {}. Model may be corrupted or tampered with.",
                expected_hash,
                actual_hash
            ));
        }
        
        println!("  ✅ Hash verified - Model integrity confirmed!");
        Ok(true)
    }
}

impl DataSplit {
    /// Create a data split
    pub fn new(start_idx: usize, end_idx: usize, total_size: usize) -> Self {
        Self {
            start_idx,
            end_idx,
            total_size,
        }
    }
    
    /// Calculate split size
    pub fn size(&self) -> usize {
        self.end_idx - self.start_idx
    }
    
    /// Check if split is valid
    pub fn is_valid(&self) -> bool {
        self.start_idx < self.end_idx && self.end_idx <= self.total_size
    }
    
    /// Create N equal splits from total dataset size
    pub fn create_splits(total_size: usize, num_splits: usize) -> Vec<DataSplit> {
        let split_size = total_size / num_splits;
        let remainder = total_size % num_splits;
        
        let mut splits = Vec::new();
        let mut current_start = 0;
        
        for i in 0..num_splits {
            // Last split gets the remainder
            let current_size = if i == num_splits - 1 {
                split_size + remainder
            } else {
                split_size
            };
            
            let current_end = current_start + current_size;
            
            splits.push(DataSplit::new(current_start, current_end, total_size));
            current_start = current_end;
        }
        
        splits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_data_split_creation() {
        let splits = DataSplit::create_splits(60000, 2);
        
        assert_eq!(splits.len(), 2);
        assert_eq!(splits[0].start_idx, 0);
        assert_eq!(splits[0].end_idx, 30000);
        assert_eq!(splits[1].start_idx, 30000);
        assert_eq!(splits[1].end_idx, 60000);
    }
    
    #[test]
    fn test_data_split_uneven() {
        let splits = DataSplit::create_splits(60000, 3);
        
        assert_eq!(splits.len(), 3);
        // First two get 20000 each
        assert_eq!(splits[0].size(), 20000);
        assert_eq!(splits[1].size(), 20000);
        // Last one gets remainder
        assert_eq!(splits[2].size(), 20000);
    }
    
    #[test]
    fn test_job_request_serialization() {
        let job_req = JobRequest::new(
            "test-job-1".to_string(),
            "mnist".to_string(),
            DataSplit::new(0, 30000, 60000),
            "tiny_linear".to_string(),
            5,
            32,
            "npub1test".to_string(),
            1,
            3,
        );
        
        let json = serde_json::to_string(&job_req).unwrap();
        let parsed: JobRequest = serde_json::from_str(&json).unwrap();
        
        assert_eq!(parsed.job_id, "test-job-1");
        assert_eq!(parsed.dataset, "mnist");
        assert_eq!(parsed.data_split.start_idx, 0);
        assert_eq!(parsed.data_split.end_idx, 30000);
    }
    
    #[test]
    fn test_job_request_with_model_params() {
        use sha2::{Sha256, Digest};
        
        let model_data = b"fake_model_parameters";
        let mut hasher = Sha256::new();
        hasher.update(model_data);
        let hash = format!("{:x}", hasher.finalize());
        
        let job_req = JobRequest::new(
            "test-job-2".to_string(),
            "mnist".to_string(),
            DataSplit::new(0, 30000, 60000),
            "tiny_linear".to_string(),
            5,
            32,
            "npub1test".to_string(),
            2,  // Round 2
            3,
        )
        .with_model_params("base64_encoded_params".to_string(), hash.clone());
        
        assert!(job_req.current_model_params.is_some());
        assert_eq!(job_req.current_model_params_hash, Some(hash));
    }
    
    #[test]
    fn test_verify_model_hash_success() {
        use sha2::{Sha256, Digest};
        
        let model_data = b"fake_model_parameters";
        let mut hasher = Sha256::new();
        hasher.update(model_data);
        let expected_hash = format!("{:x}", hasher.finalize());
        
        let job_req = JobRequest::new(
            "test-job-3".to_string(),
            "mnist".to_string(),
            DataSplit::new(0, 30000, 60000),
            "tiny_linear".to_string(),
            5,
            32,
            "npub1test".to_string(),
            2,
            3,
        )
        .with_model_params("params".to_string(), expected_hash);
        
        // Verify with correct data
        let result = job_req.verify_model_hash(model_data).unwrap();
        assert!(result); // Should return true for successful verification
    }
    
    #[test]
    fn test_verify_model_hash_mismatch() {
        use sha2::{Sha256, Digest};
        
        let model_data = b"fake_model_parameters";
        let mut hasher = Sha256::new();
        hasher.update(model_data);
        let correct_hash = format!("{:x}", hasher.finalize());
        
        let job_req = JobRequest::new(
            "test-job-4".to_string(),
            "mnist".to_string(),
            DataSplit::new(0, 30000, 60000),
            "tiny_linear".to_string(),
            5,
            32,
            "npub1test".to_string(),
            2,
            3,
        )
        .with_model_params("params".to_string(), correct_hash);
        
        // Verify with WRONG data
        let wrong_data = b"different_model_parameters";
        let result = job_req.verify_model_hash(wrong_data);
        
        assert!(result.is_err()); // Should error on hash mismatch
        assert!(result.unwrap_err().to_string().contains("hash mismatch"));
    }
    
    #[test]
    fn test_verify_model_hash_round_one() {
        let job_req = JobRequest::new(
            "test-job-5".to_string(),
            "mnist".to_string(),
            DataSplit::new(0, 30000, 60000),
            "tiny_linear".to_string(),
            5,
            32,
            "npub1test".to_string(),
            1,  // Round 1 - no previous model
            3,
        );
        
        // Round 1 has no hash to verify
        let result = job_req.verify_model_hash(b"any_data").unwrap();
        assert!(!result); // Returns false (no verification needed)
    }
}
