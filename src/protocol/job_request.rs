use nostr::{Event, EventBuilder, Keys, Kind, Tag};
use serde::{Deserialize, Serialize};
use color_eyre::eyre::Result;

use super::KIND_JOB_REQUEST;

/// Job Request Event (Kind 8000)
///
/// Customer publishes this to request federated learning training from DVMs.
/// The job request is the single source of truth for all training configuration.
/// DVMs need nothing from their own environment except credentials (keypair, auth token).
/// Corresponds to Event Type 2 in the FEDSTR paper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRequest {
    /// Unique job ID (UUID)
    pub job_id: String,

    /// Target DVM public key (npub or hex)
    pub service_provider: String,

    /// Federated learning round info
    pub round: u32,
    pub total_rounds: u32,

    // ── Algorithm selection ───────────────────────────────────────────────
    /// Algorithm: "fedavg" | "diloco"
    pub algorithm: String,

    /// Dataset: "mnist" | "shakespeare"
    pub dataset: String,

    /// Model architecture: "tiny_linear" | "nanogpt"
    pub model_type: String,

    // ── Data split ────────────────────────────────────────────────────────
    /// Which portion of the dataset this DVM should use
    pub data_split: DataSplit,

    // ── Inner optimization hyperparameters ────────────────────────────────
    /// Number of inner optimization steps (H in the DiLoCo paper)
    pub epochs: u32,

    /// Training batch size
    pub batch_size: u32,

    /// Inner optimizer learning rate (AdamW for DiLoCo, SGD for FedAvg)
    pub lr_inner: f64,

    /// AdamW weight decay
    pub weight_decay: f64,

    /// Gradient clipping value (0.0 = disabled)
    pub grad_clip: f64,

    // ── Model architecture (nanoGPT) ──────────────────────────────────────
    /// Number of transformer layers
    pub n_layer: u32,

    /// Number of attention heads
    pub n_head: u32,

    /// Embedding dimension
    pub n_embd: u32,

    /// Context length (block size)
    pub block_size: u32,

    /// Dropout rate
    pub dropout: f64,

    // ── Model state ───────────────────────────────────────────────────────
    /// Current global model parameters URL or base64
    /// None for bootstrap round, Some for subsequent rounds
    pub current_model_params: Option<String>,

    /// SHA256 hash of current model parameters for integrity verification
    pub current_model_params_hash: Option<String>,

    // ── Execution constraints ─────────────────────────────────────────────
    pub max_execution_time_secs: u64,
    pub timeout_secs: u64,

    /// Payment amount (dummy for Phase 1)
    pub payment_msats: u64,
}

/// Specifies which portion of the dataset this DVM should use
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSplit {
    pub start_idx: usize,
    pub end_idx: usize,
    pub total_size: usize,
}

/// Training configuration read from algorithms/{algo}/config.yaml
/// Customer reads this and embeds it into every JobRequest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmConfig {
    pub algorithm: String,
    pub model: String,
    pub dataset: String,
    // Inner optimization
    pub lr_inner: f64,
    pub weight_decay: f64,
    pub grad_clip: f64,
    pub batch_size: u32,
    // nanoGPT architecture
    pub n_layer: u32,
    pub n_head: u32,
    pub n_embd: u32,
    pub block_size: u32,
    pub dropout: f64,
    // Outer optimization (customer-side only, not sent to DVMs)
    pub lr_outer: f64,
    pub momentum: f64,
}

impl AlgorithmConfig {
    /// Load from algorithms/{algo}/config.yaml
    pub fn load(algorithm: &str) -> Result<Self> {
        let path = format!("/opt/fedstr/algorithms/{}/config.yaml", algorithm);
        let contents = std::fs::read_to_string(&path)
            .map_err(|e| color_eyre::eyre::eyre!("Failed to read {}: {}", path, e))?;
        let config: AlgorithmConfig = serde_yaml::from_str(&contents)
            .map_err(|e| color_eyre::eyre::eyre!("Failed to parse {}: {}", path, e))?;
        Ok(config)
    }

    /// Load with fallback to local path (for development outside Docker)
    pub fn load_with_fallback(algorithm: &str) -> Self {
        // Try Docker path first
        if let Ok(config) = Self::load(algorithm) {
            return config;
        }
        // Fall back to defaults
        println!("  ⚠️  Could not load config.yaml for {}, using defaults", algorithm);
        match algorithm {
            "diloco" => Self {
                algorithm: "diloco".to_string(),
                model: "nanogpt".to_string(),
                dataset: "shakespeare".to_string(),
                lr_inner: 0.001,
                weight_decay: 0.1,
                grad_clip: 1.0,
                batch_size: 32,
                n_layer: 4,
                n_head: 4,
                n_embd: 128,
                block_size: 64,
                dropout: 0.0,
                lr_outer: 0.7,
                momentum: 0.9,
            },
            _ => Self {
                algorithm: "fedavg".to_string(),
                model: "tiny_linear".to_string(),
                dataset: "mnist".to_string(),
                lr_inner: 0.01,
                weight_decay: 0.0,
                grad_clip: 0.0,
                batch_size: 32,
                n_layer: 0,
                n_head: 0,
                n_embd: 0,
                block_size: 0,
                dropout: 0.0,
                lr_outer: 1.0,
                momentum: 0.0,
            },
        }
    }
}

impl JobRequest {
    /// Create a new job request from algorithm config
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
            algorithm: "fedavg".to_string(),
            epochs,
            batch_size,
            lr_inner: 0.01,
            weight_decay: 0.0,
            grad_clip: 0.0,
            n_layer: 0,
            n_head: 0,
            n_embd: 0,
            block_size: 0,
            dropout: 0.0,
            current_model_params: None,
            current_model_params_hash: None,
            round,
            total_rounds,
            service_provider,
            max_execution_time_secs: 7200,
            timeout_secs: 7200,
            payment_msats: 1000,
        }
    }

    /// Create a new job request from AlgorithmConfig — preferred constructor
    pub fn from_config(
        job_id: String,
        config: &AlgorithmConfig,
        data_split: DataSplit,
        epochs: u32,
        service_provider: String,
        round: u32,
        total_rounds: u32,
    ) -> Self {
        Self {
            job_id,
            service_provider,
            round,
            total_rounds,
            algorithm: config.algorithm.clone(),
            dataset: config.dataset.clone(),
            model_type: config.model.clone(),
            data_split,
            epochs,
            batch_size: config.batch_size,
            lr_inner: config.lr_inner,
            weight_decay: config.weight_decay,
            grad_clip: config.grad_clip,
            n_layer: config.n_layer,
            n_head: config.n_head,
            n_embd: config.n_embd,
            block_size: config.block_size,
            dropout: config.dropout,
            current_model_params: None,
            current_model_params_hash: None,
            max_execution_time_secs: 7200,
            timeout_secs: 7200,
            payment_msats: 1000,
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
            Tag::parse(&["algorithm", &self.algorithm])?,
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

    /// Verify that model parameters match the expected hash
    pub fn verify_model_hash(&self, model_bytes: &[u8]) -> Result<bool> {
        use sha2::{Sha256, Digest};

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

        let mut hasher = Sha256::new();
        hasher.update(model_bytes);
        let actual_hash = format!("{:x}", hasher.finalize());

        println!("     Computed hash: {}", actual_hash);

        if actual_hash != *expected_hash {
            println!("  ❌ HASH MISMATCH - Model verification FAILED!");
            return Err(color_eyre::eyre::eyre!(
                "Model hash mismatch! Expected: {}, Got: {}",
                expected_hash,
                actual_hash
            ));
        }

        println!("  ✅ Hash verified - Model integrity confirmed!");
        Ok(true)
    }
}

impl DataSplit {
    pub fn new(start_idx: usize, end_idx: usize, total_size: usize) -> Self {
        Self { start_idx, end_idx, total_size }
    }

    pub fn size(&self) -> usize {
        self.end_idx - self.start_idx
    }

    pub fn is_valid(&self) -> bool {
        self.start_idx < self.end_idx && self.end_idx <= self.total_size
    }

    pub fn create_splits(total_size: usize, num_splits: usize) -> Vec<DataSplit> {
        let split_size = total_size / num_splits;
        let remainder = total_size % num_splits;

        let mut splits = Vec::new();
        let mut current_start = 0;

        for i in 0..num_splits {
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
        assert_eq!(splits[0].size(), 20000);
        assert_eq!(splits[1].size(), 20000);
        assert_eq!(splits[2].size(), 20000);
    }

    #[test]
    fn test_job_request_serialization() {
        let job_req = JobRequest::new(
            "test-job-1".to_string(),
            "shakespeare".to_string(),
            DataSplit::new(0, 30000, 60000),
            "nanogpt".to_string(),
            500,
            32,
            "npub1test".to_string(),
            1,
            10,
        );

        let json = serde_json::to_string(&job_req).unwrap();
        let parsed: JobRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.job_id, "test-job-1");
        assert_eq!(parsed.dataset, "shakespeare");
        assert_eq!(parsed.data_split.start_idx, 0);
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
            "shakespeare".to_string(),
            DataSplit::new(0, 30000, 60000),
            "nanogpt".to_string(),
            500,
            32,
            "npub1test".to_string(),
            2,
            10,
        )
        .with_model_params("base64_encoded_params".to_string(), hash.clone());

        assert!(job_req.current_model_params.is_some());
        assert_eq!(job_req.current_model_params_hash, Some(hash));
    }
}