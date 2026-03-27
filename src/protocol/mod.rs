/// FEDSTR Protocol Event Types
/// 
/// This module implements the event types defined in the FEDSTR paper:
/// - Job Request (kind 8000): Customer requests training job
/// - Job Result (kind 6000): DVM returns trained model
/// - Job Feedback (kind 7000): DVM sends status updates
/// - Payment events (kind 9734/9735): Dummy payment coordination

pub mod job_request;
pub mod job_result;
pub mod job_feedback;
pub mod payments;

pub use job_request::{JobRequest, DataSplit, AlgorithmConfig};
pub use job_result::{JobResult, ModelParams, TrainingMetrics};
pub use job_feedback::{JobFeedback, JobStatus, PartialMetrics};
pub use payments::{PaymentRequest, PaymentReceipt};

use nostr::{Event, Kind};

/// FEDSTR protocol event kinds
pub const KIND_JOB_REQUEST: u16 = 8000;
pub const KIND_JOB_RESULT: u16 = 6000;
pub const KIND_JOB_FEEDBACK: u16 = 7000;
pub const KIND_PAYMENT_REQUEST: u16 = 9734;
pub const KIND_PAYMENT_RECEIPT: u16 = 9735;
pub const KIND_DVM_ANNOUNCEMENT: u16 = 31990;

/// Helper to determine if an event is a FEDSTR protocol event
pub fn is_fedstr_event(event: &Event) -> bool {
    matches!(
        event.kind.as_u16(),
        KIND_JOB_REQUEST | KIND_JOB_RESULT | KIND_JOB_FEEDBACK 
        | KIND_PAYMENT_REQUEST | KIND_PAYMENT_RECEIPT
        | KIND_DVM_ANNOUNCEMENT
    )
}

/// Get human-readable name for event kind
pub fn event_kind_name(kind: Kind) -> &'static str {
    match kind.as_u16() {
        KIND_JOB_REQUEST => "Job Request",
        KIND_JOB_RESULT => "Job Result",
        KIND_JOB_FEEDBACK => "Job Feedback",
        KIND_PAYMENT_REQUEST => "Payment Request",
        KIND_PAYMENT_RECEIPT => "Payment Receipt",
        KIND_DVM_ANNOUNCEMENT => "DVM Announcement",
        _ => "Unknown",
    }
}
