use color_eyre::eyre::Result;

use crate::protocol::JobResult;

/// Validator for DVM results
/// 
/// Implements simplified version of Algorithm 3 from the FEDSTR paper.
pub struct Validator {
    /// Tolerance for loss increase (e.g., 0.1 = 10%)
    loss_tolerance: f64,
    
    /// Minimum expected accuracy
    min_accuracy: f64,
}

impl Validator {
    /// Create a new validator
    pub fn new(loss_tolerance: f64, min_accuracy: f64) -> Self {
        Self {
            loss_tolerance,
            min_accuracy,
        }
    }
    
    /// Default validator with reasonable settings
    pub fn default() -> Self {
        Self {
            loss_tolerance: 0.1,   // Allow 10% loss increase
            min_accuracy: 0.5,      // Minimum 50% accuracy
        }
    }
    
    /// Validate a single DVM result (Test Type A from paper)
    /// 
    /// Checks if the loss improved compared to previous round
    pub fn validate_improvement(
        &self,
        result: &JobResult,
        previous_loss: f64,
    ) -> Result<bool> {
        let current_loss = result.metrics.final_loss;
        
        // Check if loss decreased (with tolerance)
        let max_acceptable_loss = previous_loss * (1.0 + self.loss_tolerance);
        
        if current_loss > max_acceptable_loss {
            tracing::warn!(
                "DVM {} failed validation: loss={:.4} > max_acceptable={:.4}",
                result.service_provider,
                current_loss,
                max_acceptable_loss
            );
            return Ok(false);
        }
        
        // Check minimum accuracy
        if result.metrics.final_accuracy < self.min_accuracy {
            tracing::warn!(
                "DVM {} failed validation: accuracy={:.4} < min={:.4}",
                result.service_provider,
                result.metrics.final_accuracy,
                self.min_accuracy
            );
            return Ok(false);
        }
        
        Ok(true)
    }
    
    /// Validate against other service providers (Test Type A from paper)
    /// 
    /// Compares this DVM's loss against the average of others
    pub fn validate_against_peers(
        &self,
        result: &JobResult,
        all_results: &[JobResult],
    ) -> Result<bool> {
        if all_results.len() < 2 {
            // Can't compare against peers if there's only one
            return Ok(true);
        }
        
        // Calculate average loss of other DVMs
        let other_results: Vec<_> = all_results
            .iter()
            .filter(|r| r.service_provider != result.service_provider)
            .collect();
        
        if other_results.is_empty() {
            return Ok(true);
        }
        
        let avg_other_loss: f64 = other_results
            .iter()
            .map(|r| r.metrics.final_loss)
            .sum::<f64>() / other_results.len() as f64;
        
        // This DVM's loss should not be significantly worse than others
        let max_acceptable = avg_other_loss * (1.0 + self.loss_tolerance * 2.0);
        
        if result.metrics.final_loss > max_acceptable {
            tracing::warn!(
                "DVM {} failed peer validation: loss={:.4} > peer_avg={:.4}",
                result.service_provider,
                result.metrics.final_loss,
                avg_other_loss
            );
            return Ok(false);
        }
        
        Ok(true)
    }
    
    /// Comprehensive validation combining multiple checks
    pub fn validate_comprehensive(
        &self,
        result: &JobResult,
        previous_loss: Option<f64>,
        all_results: &[JobResult],
    ) -> Result<ValidationResult> {
        let mut checks = Vec::new();
        
        // Check 1: Improvement over previous round
        if let Some(prev_loss) = previous_loss {
            let improved = self.validate_improvement(result, prev_loss)?;
            checks.push(("improvement", improved));
        }
        
        // Check 2: Peer comparison (if multiple DVMs)
        if all_results.len() > 1 {
            let peer_ok = self.validate_against_peers(result, all_results)?;
            checks.push(("peer_comparison", peer_ok));
        }
        
        // Check 3: Minimum accuracy
        let meets_min_acc = result.metrics.final_accuracy >= self.min_accuracy;
        checks.push(("minimum_accuracy", meets_min_acc));
        
        // Check 4: Loss is not NaN or Inf
        let loss_valid = result.metrics.final_loss.is_finite() 
            && result.metrics.final_loss > 0.0;
        checks.push(("loss_validity", loss_valid));
        
        let all_passed = checks.iter().all(|(_, passed)| *passed);
        
        Ok(ValidationResult {
            passed: all_passed,
            checks,
            final_loss: result.metrics.final_loss,
            final_accuracy: result.metrics.final_accuracy,
        })
    }
}

/// Result of validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub passed: bool,
    pub checks: Vec<(&'static str, bool)>,
    pub final_loss: f64,
    pub final_accuracy: f64,
}

impl ValidationResult {
    /// Get failed checks
    pub fn failed_checks(&self) -> Vec<&'static str> {
        self.checks
            .iter()
            .filter(|(_, passed)| !passed)
            .map(|(name, _)| *name)
            .collect()
    }
    
    /// Print validation report
    pub fn report(&self) -> String {
        let mut report = String::new();
        report.push_str(&format!("Validation Result: {}\n", if self.passed { "PASS" } else { "FAIL" }));
        report.push_str(&format!("Final Loss: {:.4}\n", self.final_loss));
        report.push_str(&format!("Final Accuracy: {:.2}%\n", self.final_accuracy * 100.0));
        report.push_str("\nChecks:\n");
        
        for (name, passed) in &self.checks {
            let status = if *passed { "✓" } else { "✗" };
            report.push_str(&format!("  {} {}\n", status, name));
        }
        
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{ModelParams, TrainingMetrics};
    
    fn create_test_result(loss: f64, accuracy: f64) -> JobResult {
        let metrics = TrainingMetrics::new(
            loss,
            accuracy,
            vec![loss],
            vec![accuracy],
            1,
        );
        
        let model_params = ModelParams::from_base64(
            "test".to_string(),
            "hash".to_string(),
            100,
        );
        
        JobResult::new(
            "req1".to_string(),
            "job1".to_string(),
            "dvm1".to_string(),
            1,
            model_params,
            metrics,
            1000,
        )
    }
    
    #[test]
    fn test_validate_improvement_success() {
        let validator = Validator::default();
        let result = create_test_result(0.3, 0.9);
        
        // Previous loss was 0.5, current is 0.3 - should pass
        assert!(validator.validate_improvement(&result, 0.5).unwrap());
    }
    
    #[test]
    fn test_validate_improvement_failure() {
        let validator = Validator::default();
        let result = create_test_result(0.8, 0.9);
        
        // Previous loss was 0.5, current is 0.8 - should fail
        assert!(!validator.validate_improvement(&result, 0.5).unwrap());
    }
    
    #[test]
    fn test_validate_min_accuracy() {
        let validator = Validator::default();
        let result = create_test_result(0.3, 0.3);  // Low accuracy
        
        // Should fail minimum accuracy check
        assert!(!validator.validate_improvement(&result, 0.5).unwrap());
    }
    
    #[test]
    fn test_comprehensive_validation() {
        let validator = Validator::default();
        let result = create_test_result(0.3, 0.9);
        let all_results = vec![result.clone()];
        
        let validation = validator.validate_comprehensive(
            &result,
            Some(0.5),
            &all_results,
        ).unwrap();
        
        assert!(validation.passed);
        println!("{}", validation.report());
    }
}
