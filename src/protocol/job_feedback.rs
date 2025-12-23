use nostr::{Event, EventBuilder, Keys, Kind, Tag};
use serde::{Deserialize, Serialize};
use color_eyre::eyre::Result;

use super::KIND_JOB_FEEDBACK;

/// Job Feedback Event (Kind 7000)
/// 
/// DVM publishes this periodically to provide status updates during training.
/// Corresponds to Event Type 4 in the FEDSTR paper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobFeedback {
    /// Reference to the original job request
    pub job_request_id: String,
    
    /// Job ID
    pub job_id: String,
    
    /// Current status
    pub status: JobStatus,
    
    /// Progress information (0.0 to 1.0)
    pub progress: f64,
    
    /// Optional message
    pub message: Option<String>,
    
    /// Current epoch (if training)
    pub current_epoch: Option<u32>,
    
    /// Partial metrics (if available)
    pub partial_metrics: Option<PartialMetrics>,
}

/// Job status enum
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum JobStatus {
    /// Payment required before starting
    PaymentRequired,
    
    /// Job accepted, preparing to start
    Accepted,
    
    /// Currently processing
    Processing,
    
    /// Job completed successfully
    Success,
    
    /// Job failed with error
    Error(String),
    
    /// Job timed out
    Timeout,
}

/// Partial training metrics for progress updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialMetrics {
    pub current_loss: f64,
    pub current_accuracy: f64,
    pub samples_processed: usize,
}

impl JobFeedback {
    /// Create a new feedback event
    pub fn new(
        job_request_id: String,
        job_id: String,
        status: JobStatus,
    ) -> Self {
        Self {
            job_request_id,
            job_id,
            status,
            progress: 0.0,
            message: None,
            current_epoch: None,
            partial_metrics: None,
        }
    }
    
    /// Create payment required feedback
    pub fn payment_required(job_request_id: String, job_id: String) -> Self {
        Self::new(job_request_id, job_id, JobStatus::PaymentRequired)
    }
    
    /// Create processing feedback with progress
    pub fn processing(
        job_request_id: String,
        job_id: String,
        progress: f64,
        current_epoch: u32,
    ) -> Self {
        let mut feedback = Self::new(job_request_id, job_id, JobStatus::Processing);
        feedback.progress = progress;
        feedback.current_epoch = Some(current_epoch);
        feedback
    }
    
    /// Create success feedback
    pub fn success(job_request_id: String, job_id: String) -> Self {
        let mut feedback = Self::new(job_request_id, job_id, JobStatus::Success);
        feedback.progress = 1.0;
        feedback
    }
    
    /// Create error feedback
    pub fn error(job_request_id: String, job_id: String, error_msg: String) -> Self {
        Self::new(job_request_id, job_id, JobStatus::Error(error_msg))
    }
    
    /// Add partial metrics
    pub fn with_metrics(mut self, metrics: PartialMetrics) -> Self {
        self.partial_metrics = Some(metrics);
        self
    }
    
    /// Add message
    pub fn with_message(mut self, message: String) -> Self {
        self.message = Some(message);
        self
    }
    
    /// Convert to NOSTR event
    pub fn to_event(&self, keys: &Keys) -> Result<Event> {
        let content = serde_json::to_string(self)?;
        
        let status_str = match &self.status {
            JobStatus::PaymentRequired => "payment_required",
            JobStatus::Accepted => "accepted",
            JobStatus::Processing => "processing",
            JobStatus::Success => "success",
            JobStatus::Error(_) => "error",
            JobStatus::Timeout => "timeout",
        };
        
        let tags = vec![
            Tag::parse(&["e", &self.job_request_id])?,
            Tag::parse(&["job_id", &self.job_id])?,
            Tag::parse(&["status", status_str])?,
        ];
        
        Ok(EventBuilder::new(Kind::Custom(KIND_JOB_FEEDBACK), content, tags)
            .to_event(keys)?)
    }
    
    /// Parse from NOSTR event
    pub fn from_event(event: &Event) -> Result<Self> {
        if event.kind.as_u16() != KIND_JOB_FEEDBACK {
            return Err(color_eyre::eyre::eyre!(
                "Invalid event kind: expected {}, got {}",
                KIND_JOB_FEEDBACK,
                event.kind.as_u16()
            ));
        }
        
        let feedback: JobFeedback = serde_json::from_str(&event.content)?;
        Ok(feedback)
    }
}

impl PartialMetrics {
    pub fn new(current_loss: f64, current_accuracy: f64, samples_processed: usize) -> Self {
        Self {
            current_loss,
            current_accuracy,
            samples_processed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_job_feedback_creation() {
        let feedback = JobFeedback::processing(
            "req123".to_string(),
            "job1".to_string(),
            0.5,
            3,
        );
        
        assert_eq!(feedback.status, JobStatus::Processing);
        assert_eq!(feedback.progress, 0.5);
        assert_eq!(feedback.current_epoch, Some(3));
    }
    
    #[test]
    fn test_job_feedback_with_metrics() {
        let feedback = JobFeedback::processing(
            "req123".to_string(),
            "job1".to_string(),
            0.7,
            5,
        )
        .with_metrics(PartialMetrics::new(0.3, 0.92, 5000));
        
        assert!(feedback.partial_metrics.is_some());
        let metrics = feedback.partial_metrics.unwrap();
        assert_eq!(metrics.current_loss, 0.3);
    }
    
    #[test]
    fn test_status_enum() {
        let statuses = vec![
            JobStatus::PaymentRequired,
            JobStatus::Processing,
            JobStatus::Success,
            JobStatus::Error("test error".to_string()),
        ];
        
        assert_eq!(statuses[0], JobStatus::PaymentRequired);
        assert!(matches!(statuses[3], JobStatus::Error(_)));
    }
}
