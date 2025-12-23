/// FEDSTR: Federated Learning on NOSTR
/// 
/// A proof-of-concept implementation of the FEDSTR protocol for decentralized
/// federated learning using the NOSTR protocol.

pub mod protocol;
pub mod ml;
pub mod storage;

// Re-export commonly used items
pub use protocol::{
    JobRequest, JobResult, JobFeedback, JobStatus,
    ModelParams, TrainingMetrics, DataSplit,
    PaymentRequest, PaymentReceipt,
};

pub use ml::{
    PyTorchTrainer, TrainingResult,
    FedAvgAggregator,
    Validator, ValidationResult,
};

pub use storage::{ModelStorage, StorageBackend, UploadResponse};
