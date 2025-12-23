use nostr::{Event, EventBuilder, Keys, Kind, Tag};
use serde::{Deserialize, Serialize};
use color_eyre::eyre::Result;

use super::{KIND_PAYMENT_REQUEST, KIND_PAYMENT_RECEIPT};

/// Payment Request Event (Kind 9734)
/// 
/// Dummy implementation for Phase 1 - just publishes payment intent.
/// Real Lightning integration in Phase 2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentRequest {
    /// Amount in millisatoshis
    pub amount_msats: u64,
    
    /// Recipient public key
    pub recipient: String,
    
    /// Job ID this payment is for
    pub job_id: String,
    
    /// Optional message
    pub message: Option<String>,
    
    /// Dummy Lightning invoice (not real for Phase 1)
    pub invoice: Option<String>,
}

/// Payment Receipt Event (Kind 9735)
/// 
/// Dummy implementation for Phase 1 - just confirms payment intent.
/// Real Lightning proof of payment in Phase 2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentReceipt {
    /// Reference to payment request event
    pub payment_request_id: String,
    
    /// Amount paid
    pub amount_msats: u64,
    
    /// Payer public key
    pub payer: String,
    
    /// Recipient public key
    pub recipient: String,
    
    /// Job ID
    pub job_id: String,
    
    /// Dummy payment preimage (not real for Phase 1)
    pub preimage: Option<String>,
}

impl PaymentRequest {
    /// Create a new payment request
    pub fn new(
        amount_msats: u64,
        recipient: String,
        job_id: String,
    ) -> Self {
        Self {
            amount_msats,
            recipient,
            job_id,
            message: None,
            invoice: None,
        }
    }
    
    /// Add a message
    pub fn with_message(mut self, message: String) -> Self {
        self.message = Some(message);
        self
    }
    
    /// Convert to NOSTR event
    pub fn to_event(&self, keys: &Keys) -> Result<Event> {
        let content = serde_json::to_string(self)?;
        
        let tags = vec![
            Tag::parse(&["p", &self.recipient])?,
            Tag::parse(&["job_id", &self.job_id])?,
            Tag::parse(&["amount", &self.amount_msats.to_string()])?,
        ];
        
        Ok(EventBuilder::new(Kind::Custom(KIND_PAYMENT_REQUEST), content, tags)
            .to_event(keys)?)
    }
    
    /// Parse from NOSTR event
    pub fn from_event(event: &Event) -> Result<Self> {
        if event.kind.as_u16() != KIND_PAYMENT_REQUEST {
            return Err(color_eyre::eyre::eyre!(
                "Invalid event kind: expected {}, got {}",
                KIND_PAYMENT_REQUEST,
                event.kind.as_u16()
            ));
        }
        
        let payment: PaymentRequest = serde_json::from_str(&event.content)?;
        Ok(payment)
    }
}

impl PaymentReceipt {
    /// Create a new payment receipt
    pub fn new(
        payment_request_id: String,
        amount_msats: u64,
        payer: String,
        recipient: String,
        job_id: String,
    ) -> Self {
        Self {
            payment_request_id,
            amount_msats,
            payer,
            recipient,
            job_id,
            preimage: Some("dummy_preimage".to_string()), // Dummy for Phase 1
        }
    }
    
    /// Convert to NOSTR event
    pub fn to_event(&self, keys: &Keys) -> Result<Event> {
        let content = serde_json::to_string(self)?;
        
        let tags = vec![
            Tag::parse(&["e", &self.payment_request_id])?,
            Tag::parse(&["p", &self.recipient])?,
            Tag::parse(&["job_id", &self.job_id])?,
            Tag::parse(&["amount", &self.amount_msats.to_string()])?,
        ];
        
        Ok(EventBuilder::new(Kind::Custom(KIND_PAYMENT_RECEIPT), content, tags)
            .to_event(keys)?)
    }
    
    /// Parse from NOSTR event
    pub fn from_event(event: &Event) -> Result<Self> {
        if event.kind.as_u16() != KIND_PAYMENT_RECEIPT {
            return Err(color_eyre::eyre::eyre!(
                "Invalid event kind: expected {}, got {}",
                KIND_PAYMENT_RECEIPT,
                event.kind.as_u16()
            ));
        }
        
        let receipt: PaymentReceipt = serde_json::from_str(&event.content)?;
        Ok(receipt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_payment_request() {
        let payment = PaymentRequest::new(
            1000,
            "npub123".to_string(),
            "job1".to_string(),
        )
        .with_message("Payment for training".to_string());
        
        assert_eq!(payment.amount_msats, 1000);
        assert!(payment.message.is_some());
    }
    
    #[test]
    fn test_payment_receipt() {
        let receipt = PaymentReceipt::new(
            "payment_req_123".to_string(),
            1000,
            "npub_payer".to_string(),
            "npub_recipient".to_string(),
            "job1".to_string(),
        );
        
        assert_eq!(receipt.amount_msats, 1000);
        assert!(receipt.preimage.is_some());
    }
}
