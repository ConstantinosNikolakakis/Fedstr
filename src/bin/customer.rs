/// FEDSTR Customer (Coordinator) - Phase 2
/// 
/// Implements Algorithm 1 from the FEDSTR paper.
/// 
/// This Customer:
/// 1. Splits dataset into N parts
/// 2. Publishes N JobRequests (kind 8000) to DVMs
/// 3. Waits for N JobResults (kind 6000)
/// 4. Aggregates using FedAvg
/// 5. Validates results
/// 6. Repeats for multiple rounds

use fedstr_poc::{
    JobRequest, JobResult, DataSplit,
    FedAvgAggregator, Validator, ValidationResult,
    PaymentReceipt,
    ModelStorage, StorageBackend,
};

use nostr::{Keys, Kind, Filter, Timestamp, SingleLetterTag, Alphabet, prelude::*};
use nostr_sdk::Client;
use color_eyre::eyre::{Result, eyre};
use std::time::Duration;
use std::collections::HashMap;
use std::str::FromStr;
use clap::Parser;
use uuid::Uuid;
use base64::{Engine as _, engine::general_purpose};
use fedstr_poc::protocol::AlgorithmConfig;

/// Command-line arguments for Customer
#[derive(Parser, Debug)]
#[command(name = "fedstr-customer")]
#[command(about = "FEDSTR Customer (Federated Learning Coordinator)", long_about = None)]
struct Args {
    /// Number of DVMs to use
    #[arg(short, long, default_value = "2")]
    num_dvms: usize,
    
    /// Number of federated rounds
    #[arg(short, long, default_value = "3")]
    rounds: u32,
    
    /// Dataset to use
    #[arg(short, long, default_value = "mnist")]
    dataset: String,
    
    /// Model type
    #[arg(short, long, default_value = "tiny_linear")]
    model: String,
    
    /// Epochs per round
    #[arg(short, long, default_value = "3")]
    epochs: u32,
    
    /// Batch size
    #[arg(short, long, default_value = "32")]
    batch_size: u32,
    
    /// DVM public keys (npub or hex, comma-separated)
    #[arg(long)]
    dvms: Option<String>,

    /// How long to poll relay for DVMs during auto-discovery, in seconds.
    #[arg(long, default_value = "60")]
    discovery_timeout: u64,

    /// Storage backend to use (local, http://..., or custom path)
    #[arg(long, default_value = "local")]
    storage: String,
    
    /// Relays to connect to
    #[arg(long, default_values_t = vec![
    //"ws://localhost:8080".to_string(),
    //"wss://nos.lol".to_string(),
    //"wss://relay.nostr.band".to_string(),
    "wss://relay.knikolakakis.com".to_string()
    ])]
    relays: Vec<String>,
}

/// Customer coordinator struct
struct Customer {
    client: Client,
    keys: Keys,
    aggregator: FedAvgAggregator,
    validator: Validator,
    storage: ModelStorage,
    dataset_size: usize,  // Total MNIST training set size
}

impl Customer {
    /// Create a new customer
    async fn new(storage_arg: &str) -> Result<Self> {
        let keys = Keys::generate();
        
        // Configure client with larger message limits
        let opts = nostr_sdk::Options::new()
            .max_avg_latency(Duration::from_secs(30))
            .connection_timeout(Some(Duration::from_secs(60)));
        
        let client = Client::with_opts(&keys, opts);
        let aggregator = FedAvgAggregator::new()?;
        let validator = Validator::default();
        
        // Initialize storage for downloading external models
        let storage_backend = if storage_arg == "local" {
            StorageBackend::LocalFile("/tmp/fedstr_models".to_string())
        } else if storage_arg.starts_with("http") {
            println!("📁 Using HTTP storage: {}", storage_arg);
            StorageBackend::HttpServer(storage_arg.to_string())
        } else {
            println!("📁 Using local storage: {}", storage_arg);
            StorageBackend::LocalFile(storage_arg.to_string())
        };
        let storage = ModelStorage::new(storage_backend);
        
        Ok(Self {
            client,
            keys,
            aggregator,
            validator,
            storage,
            dataset_size: 60000,  // MNIST training set size
        })
    }
    
    /// Connect to relays
    async fn connect_to_relays(&self, relay_urls: &[String]) -> Result<()> {
        println!("\n🔌 Connecting to relays...");
        for url in relay_urls {
            match self.client.add_relay(url).await {
                Ok(_) => println!("  ✓ {}", url),
                Err(e) => println!("  ✗ {}: {}", url, e),
            }
        }
        
        self.client.connect().await;
        tokio::time::sleep(Duration::from_secs(2)).await;
        Ok(())
    }
    
    /// Main federated training loop (Algorithm 1)
    async fn run_federated_training(&self, args: Args) -> Result<()> {

        // Load algorithm config — single source of truth for all training hyperparameters
        let algorithm = std::env::var("ALGORITHM").unwrap_or_else(|_| "fedavg".to_string());
        let algo_config = AlgorithmConfig::load_with_fallback(&algorithm);
        // Single source of truth — override CLI args with config values
        let dataset = algo_config.dataset.clone();
        let model = algo_config.model.clone();
        let epochs = args.epochs;      // keep from CLI — user sets H via .env
        let batch_size = algo_config.batch_size;

        println!("\n╔═══════════════════════════════════════════════════╗");
        println!("║   FEDSTR Federated Learning Session               ║");
        println!("╚═══════════════════════════════════════════════════╝");
        println!("\nConfiguration:");
        println!("  Algorithm:    {}", algo_config.algorithm);
        println!("  Dataset:      {}", dataset);
        println!("  Model:        {}", model);
        println!("  DVMs:         {}", args.num_dvms);
        println!("  Rounds:       {}", args.rounds);
        println!("  Epochs/round: {}", epochs);
        println!("  Batch size:   {}", batch_size);
        println!("  lr_inner:     {}", algo_config.lr_inner);
        println!("  lr_outer:     {}", algo_config.lr_outer);
        println!("  momentum:     {}", algo_config.momentum);
        
        // Resolve DVM pubkeys: explicit list OR relay auto-discovery
        let dvm_pubkeys = if let Some(dvms_str) = &args.dvms {
            println!("\n[OK] Using manually specified DVMs.");
            self.parse_dvm_pubkeys(dvms_str)?
        } else {
            println!("\n[SEARCH] No --dvms provided -- discovering DVMs via relay (kind 31990)...");
            self.discover_dvms(args.num_dvms, args.discovery_timeout).await?
        };
        
        if dvm_pubkeys.len() != args.num_dvms {
            return Err(eyre!("Expected {} DVMs, got {}", args.num_dvms, dvm_pubkeys.len()));
        }
        
        println!("\n📋 Using DVMs:");
        for (i, pubkey) in dvm_pubkeys.iter().enumerate() {
            println!("  {}. {}", i + 1, pubkey.to_bech32().unwrap_or_else(|_| pubkey.to_string()));
        }
        
        // Create data splits
        let splits = DataSplit::create_splits(self.dataset_size, args.num_dvms);
        
        println!("\n📊 Data Distribution:");
        for (i, split) in splits.iter().enumerate() {
            let start_pct = split.start_idx as f64 / self.dataset_size as f64 * 100.0;
            let end_pct = split.end_idx as f64 / self.dataset_size as f64 * 100.0;
            println!("  DVM {}: {:.0}%-{:.0}% of dataset ({} reference units)", 
                i + 1, start_pct, end_pct, split.size());
        }
        

        // Setup customer training log
        let log_dir = "/opt/fedstr/logs";
        std::fs::create_dir_all(log_dir).ok();
        let log_path = format!("{}/customer_training.log", log_dir);
        let mut log_file = std::fs::File::create(&log_path)?;
        use std::io::Write;
        let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC");


        println!("  Algorithm:    {}", algo_config.algorithm);
        println!("  Dataset:      {}", algo_config.dataset);
        println!("  Model:        {}", algo_config.model);

        writeln!(log_file, "FEDSTR DiLoCo Training Log — Customer")?;
        writeln!(log_file, "Run started: {}", now)?;
        writeln!(log_file, "Algorithm Config: {:?}", algo_config)?;
        writeln!(log_file, "DVMs: {}, Rounds: {}, Inner steps (H): {}", args.num_dvms, args.rounds, epochs)?;
        writeln!(log_file, "{}", "=".repeat(60))?;
        writeln!(log_file, "")?;
        
        // Initialize model parameters (None for first round)
        let mut current_params: Option<String> = None;
        let mut current_params_hash: Option<String> = None;
        let mut current_v_t: Option<String> = None;  // DiLoCo momentum state
        let mut previous_loss: Option<f64> = None;
        
        // Bootstrap Round: Train from scratch to get initial models
        println!("\n╔═══════════════════════════════════════════════════╗");
        println!("║   Bootstrap Round (0/{})                           ║", args.rounds);
        println!("╚═══════════════════════════════════════════════════╝");
        println!("📝 Training from scratch to initialize models...");
        
        let bootstrap_results = self.execute_round(
            0,
            args.rounds,
            &dvm_pubkeys,
            &splits,
            &algo_config,
            epochs,
            None,
            None,
        ).await?;
        
        println!("\n📥 Received {} bootstrap results", bootstrap_results.len());
        
        // Validate bootstrap results
        println!("\n🔍 Validating bootstrap results...");
        let valid_bootstrap: Vec<_> = bootstrap_results.iter()
            .filter(|result| {
                let validation = self.validator.validate_comprehensive(
                    result,
                    None,
                    &bootstrap_results,
                ).unwrap_or_else(|_| ValidationResult {
                    passed: false,
                    checks: vec![],
                    final_loss: 0.0,
                    final_accuracy: 0.0,
                });
                validation.passed
            })
            .cloned()
            .collect();
        
        println!("  ✓ {}/{} bootstrap results passed validation", valid_bootstrap.len(), bootstrap_results.len());
        
        if valid_bootstrap.is_empty() {
            return Err(eyre!("No valid bootstrap results"));
        }
        
        // Aggregate bootstrap results
        println!("\n⚙️  Aggregating bootstrap models...");
        println!("  Bootstrap - Aggregating results from {} DVMs:", valid_bootstrap.len());
        for (i, result) in valid_bootstrap.iter().enumerate() {
            println!("    DVM {}: loss={:.4}, accuracy={:.2}%, samples={}", 
                i + 1,
                result.metrics.final_loss,
                result.metrics.final_accuracy,
                result.training_samples);
            if let Some(ref url) = result.model_params.params_url {
                println!("      Model URL: {}", url);
            }
        }
        
        let bootstrap_aggregated = self.aggregator.aggregate(
            &valid_bootstrap, &self.storage, None, None).await?;

        
        // Upload bootstrap aggregated model
        println!("\n📤 Uploading bootstrap aggregated model...");
        if let Some(ref base64_data) = bootstrap_aggregated.theta_global.params_base64 {
            let model_bytes = base64::engine::general_purpose::STANDARD.decode(base64_data)?;
            let upload_result = self.storage.upload_model(
                &model_bytes,
                &bootstrap_aggregated.theta_global.params_hash,
            ).await?;
            println!("  ✓ Uploaded to: {}", upload_result.url);
            current_params = Some(upload_result.url.clone());
            current_v_t = bootstrap_aggregated.v_t.as_ref()
                .and_then(|vt| vt.params_url.clone());
            current_params_hash = Some(bootstrap_aggregated.theta_global.params_hash.clone());  // 🔐 Store hash!
            println!("  🔐 Stored hash for next round: {}", bootstrap_aggregated.theta_global.params_hash);
        } else {
            println!("  ⚠️  WARNING: Bootstrap aggregated model has no base64 data!");
        }
        
        let avg_loss: f64 = valid_bootstrap.iter()
            .map(|r| r.metrics.final_loss)
            .sum::<f64>() / valid_bootstrap.len() as f64;
        
        let avg_accuracy: f64 = valid_bootstrap.iter()
            .map(|r| r.metrics.final_accuracy)
            .sum::<f64>() / valid_bootstrap.len() as f64;
        
        let avg_val_loss: f64 = valid_bootstrap.iter()
            .map(|r| r.metrics.val_loss)
            .sum::<f64>() / valid_bootstrap.len() as f64;
        let avg_val_ppl = avg_val_loss.exp();

        println!("\n📊 Bootstrap Results:");
        println!("  Average Train Loss:  {:.4}", avg_loss);
        println!("  Average Val Loss:    {:.4}", avg_val_loss);
        println!("  Average Val PPL:     {:.4}", avg_val_ppl);

        writeln!(log_file, "Bootstrap Round:")?;
        writeln!(log_file, "  {:>6}  {:>10}  {:>10}  {:>12}", "Round", "Train Loss", "Val Loss", "Val PPL")?;
        writeln!(log_file, "  {:>6}  {:>10.4}  {:>10.4}  {:>12.4}", 0, avg_loss, avg_val_loss, avg_val_ppl)?;
                
        previous_loss = Some(avg_loss);
        
        // Federated training rounds (now with initial params)
        for round in 1..=args.rounds {
            println!("\n╔═══════════════════════════════════════════════════╗");
            println!("║   Round {}/{}                                        ║", round, args.rounds);
            println!("╚═══════════════════════════════════════════════════╝");
            
            // Execute round
            let round_results = self.execute_round(
                round,
                args.rounds,
                &dvm_pubkeys,
                &splits,
                &algo_config,
                epochs,
                current_params.clone(),
                current_params_hash.clone(),
            ).await?;
            
            println!("\n📥 Received {} results", round_results.len());
            
            // Validate results
            println!("\n🔍 Validating results...");
            let valid_results: Vec<_> = round_results.iter()
                .filter(|result| {
                    let validation = self.validator.validate_comprehensive(
                        result,
                        previous_loss,
                        &round_results,
                    ).unwrap_or_else(|_| ValidationResult {
                        passed: false,
                        checks: vec![],
                        final_loss: 0.0,
                        final_accuracy: 0.0,
                    });
                    
                    if !validation.passed {
                        println!("  ✗ DVM {} failed validation", result.service_provider);
                        for check in validation.failed_checks() {
                            println!("    - Failed: {}", check);
                        }
                    }
                    
                    validation.passed
                })
                .cloned()
                .collect();
            
            println!("  ✓ {}/{} results passed validation", valid_results.len(), round_results.len());
            
            if valid_results.is_empty() {
                return Err(eyre!("No valid results in round {}", round));
            }
            
            // Send payments (dummy for Phase 1)
            println!("\n₿ Sending payments...");
            for result in &valid_results {
                self.send_payment(&result).await?;
            }
            
            // Aggregate using FedAvg
            let algo = std::env::var("ALGORITHM").unwrap_or_else(|_| "fedavg".to_string());
            println!("\n⚙️  Aggregating with {}...", algo.to_uppercase());
            println!("  Round {} - Aggregating results from {} DVMs:", round, valid_results.len());
            for (i, result) in valid_results.iter().enumerate() {
                println!("    DVM {}: loss={:.4}, accuracy={:.2}%, samples={}", 
                    i + 1,
                    result.metrics.final_loss,
                    result.metrics.final_accuracy,
                    result.training_samples);
                if let Some(ref url) = result.model_params.params_url {
                    println!("      Model URL: {}", url);
                }
            }
            
            let aggregated_params = self.aggregator.aggregate(
                &valid_results, &self.storage,
                current_params.clone(),
                current_v_t.clone(),   // ← pass momentum state
            ).await?;
            
            // Upload aggregated model to storage for next round
            let next_round_params = if round < args.rounds {
                // Not the last round - upload for DVMs to use
                println!("\n📤 Uploading aggregated model for next round...");
                
                if let Some(ref base64_data) = aggregated_params.theta_global.params_base64 {
                    // Decode base64 to bytes
                    let model_bytes = base64::engine::general_purpose::STANDARD.decode(base64_data)?;
                    
                    // Upload to storage
                    let upload_result = self.storage.upload_model(
                        &model_bytes,
                        &aggregated_params.theta_global.params_hash,
                    ).await?;
                    
                    println!("  ✓ Uploaded to: {}", upload_result.url);
                    
                    // Return URL for next round
                    Some(upload_result.url)
                } else {
                    // Already a URL
                    aggregated_params.theta_global.params_url.clone()
                }
            } else {
                // Last round - save final model
                println!("\n💾 Saving final aggregated model...");
                
                if let Some(ref base64_data) = aggregated_params.theta_global.params_base64 {
                    // Decode base64 to bytes
                    let model_bytes = base64::engine::general_purpose::STANDARD.decode(base64_data)?;
                    
                    // Upload to storage
                    let upload_result = self.storage.upload_model(
                        &model_bytes,
                        &aggregated_params.theta_global.params_hash,
                    ).await?;
                    
                    println!("  ✓ Final model saved to: {}", upload_result.url);
                    Some(upload_result.url)
                } else {
                    // Already a URL
                    aggregated_params.theta_global.params_url.clone()
                }
            };
            
            // Calculate metrics
            let avg_loss: f64 = valid_results.iter()
                .map(|r| r.metrics.final_loss)
                .sum::<f64>() / valid_results.len() as f64;
            
            let avg_accuracy: f64 = valid_results.iter()
                .map(|r| r.metrics.final_accuracy)
                .sum::<f64>() / valid_results.len() as f64;
            
            let avg_val_loss: f64 = valid_results.iter()
                .map(|r| r.metrics.val_loss)
                .sum::<f64>() / valid_results.len() as f64;
            let avg_val_ppl = avg_val_loss.exp();

            println!("\n📊 Round {} Results:", round);
            println!("  Average Train Loss:  {:.4}", avg_loss);
            println!("  Average Val Loss:    {:.4}", avg_val_loss);
            println!("  Average Val PPL:     {:.4}", avg_val_ppl);

            if let Some(prev_loss) = previous_loss {
                let improvement = ((prev_loss - avg_loss) / prev_loss) * 100.0;
                println!("  Improvement:        {:.2}%", improvement);
            }

            writeln!(log_file, "  {:>6}  {:>10.4}  {:>10.4}  {:>12.4}", round, avg_loss, avg_val_loss, avg_val_ppl)?;
            
            // Update for next round
            current_params = next_round_params.clone();
            current_params_hash = Some(aggregated_params.theta_global.params_hash.clone());  // 🔐 Update hash!
            // Upload v_t momentum state for next round (DiLoCo only)
            if let Some(ref vt_params) = aggregated_params.v_t {
                if let Some(ref vt_b64) = vt_params.params_base64 {
                    let vt_bytes = base64::engine::general_purpose::STANDARD.decode(vt_b64)?;
                    let vt_upload = self.storage.upload_model(
                        &vt_bytes,
                        &vt_params.params_hash,
                    ).await?;
                    current_v_t = Some(vt_upload.url.clone());
                    println!("  📦 Momentum state v_t uploaded: {}", vt_upload.url);
                }
            }
            println!("  🔐 Stored hash for round {}: {}", round + 1, aggregated_params.theta_global.params_hash);
            previous_loss = Some(avg_loss);
        }
        
        // FINAL AGGREGATION: The last round's results haven't been aggregated yet
        // DVMs trained with round N params, we need to aggregate their results one more time
        println!("\n╔═══════════════════════════════════════════════════╗");
        println!("║   Final Aggregation                                 ║");
        println!("╚═══════════════════════════════════════════════════╝");
        println!("\n⚙️  Performing final aggregation of all DVM results...");
        
        // The DVMs have completed their final training, but we haven't aggregated their results yet
        // We need to wait for and aggregate the results from the last round
        // (This is already done in the loop above, so we just need to save the final model)
        
        println!("\n╔═══════════════════════════════════════════════════╗");
        println!("║   Federated Training Complete!                     ║");
        println!("╚═══════════════════════════════════════════════════╝");
        
        if let Some(final_loss) = previous_loss {
            println!("\n📊 Final Results:");
            println!("  Final Loss:   {:.4}", final_loss);
            println!("  Rounds:       {}", args.rounds);
            println!("  DVMs:         {}", args.num_dvms);
            if let Some(ref final_model_url) = current_params {
                println!("  Final Model:  {}", final_model_url);
            }
        }
        
        Ok(())
    }
    
    /// Execute a single federated round
    async fn execute_round(
        &self,
        round: u32,
        total_rounds: u32,
        dvm_pubkeys: &[PublicKey],
        splits: &[DataSplit],
        algo_config: &AlgorithmConfig,
        epochs: u32,
        current_params: Option<String>,
        current_params_hash: Option<String>,
    ) -> Result<Vec<JobResult>> {
        let job_id = Uuid::new_v4().to_string();
        
        println!("\n📤 Publishing {} JobRequests...", dvm_pubkeys.len());
        
        // Publish JobRequests to all DVMs
        let mut request_ids = HashMap::new();
        
        for (i, (pubkey, split)) in dvm_pubkeys.iter().zip(splits.iter()).enumerate() {
            let mut job_request = JobRequest::from_config(
                job_id.clone(),
                algo_config,
                split.clone(),
                epochs,
                pubkey.to_string(),
                round,
                total_rounds,
            );
            
            // 🔐 Set current model params WITH HASH if not first round
            if let (Some(ref params), Some(ref hash)) = (&current_params, &current_params_hash) {
                println!("  DEBUG: Setting model params with hash for DVM {}", i + 1);
                println!("         URL: {}", params);
                println!("         Hash: {}", hash);
                job_request = job_request.with_model_params(
                    params.clone(),
                    hash.clone(),
                );
            } else {
                println!("  DEBUG: No hash to set for DVM {} (round {}, params: {:?}, hash: {:?})", 
                    i + 1, round, current_params.is_some(), current_params_hash.is_some());
            }
            
            let event = job_request.to_event(&self.keys)?;
            self.client.send_event(event.clone()).await?;
            
            request_ids.insert(pubkey.to_string(), event.id);
            
            println!("  ✓ DVM {} - Request sent (Job ID: {}, Event ID: {}, Hash: {})", 
                i + 1, 
                job_id,
                event.id.to_hex().chars().take(12).collect::<String>(),
                current_params_hash.as_ref().map(|h| h.chars().take(12).collect::<String>()).unwrap_or_else(|| "None".to_string()));
        }
        
        // Wait for results
        println!("\n⏳ Waiting for JobResults...");
        println!("   (Training can take 15-30 minutes per DVM)");
        let timeout = Duration::from_secs(7200); // 120 minutes
        let start = std::time::Instant::now();
        
        let mut results = Vec::new();
        let since = Timestamp::now();
        let mut last_progress = std::time::Instant::now();
        
        while results.len() < dvm_pubkeys.len() && start.elapsed() < timeout {
            // Print progress every 30 seconds
            if last_progress.elapsed() > Duration::from_secs(30) {
                let elapsed = start.elapsed().as_secs();
                let remaining = (timeout.as_secs() - elapsed) / 60;
                println!("   ⏱  Waiting... {}/{} results received ({} minutes remaining)", 
                    results.len(), 
                    dvm_pubkeys.len(),
                    remaining);
                last_progress = std::time::Instant::now();
            }
            
            // Query for JobResult events from our specific DVMs only.
            // Use a short relay timeout so we re-poll on each loop iteration
            // rather than holding one long subscription that the relay closes.
            let filter = Filter::new()
                .kind(Kind::Custom(6000))
                .authors(dvm_pubkeys.to_vec())
                .since(since);
            
            let events = self.client.get_events_of(
                vec![filter],
                nostr_sdk::EventSource::relays(Some(Duration::from_secs(10))),
            ).await.unwrap_or_default();
            
            for event in events {
                // Parse JobResult
                if let Ok(job_result) = JobResult::from_event(&event) {
                    // Check if this is for our job and round
                    if job_result.job_id == job_id && job_result.round == round {
                        // Check if we haven't already added this result
                        if !results.iter().any(|r: &JobResult| r.service_provider == job_result.service_provider) {
                            println!("  ✓ Received result from {} (loss: {:.4}, accuracy: {:.2}%)", 
                                job_result.service_provider.chars().take(12).collect::<String>(),
                                job_result.metrics.final_loss,
                                job_result.metrics.final_accuracy);
                            results.push(job_result);
                        }
                    }
                }
            }
            
            if results.len() < dvm_pubkeys.len() {
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        }
        
        if results.len() < dvm_pubkeys.len() {
            let elapsed_mins = start.elapsed().as_secs() / 60;
            println!("  ⚠️  Timeout after {} minutes: Received {}/{} results", 
                elapsed_mins,
                results.len(), 
                dvm_pubkeys.len());
            println!("  💡 Tip: Training takes 5-10 minutes per DVM. Check DVM logs for errors.");
        }
        
        Ok(results)
    }
    
    /// Auto-discover available DVMs via relay (kind 31990, NIP-89).
    ///
    /// Polls the relay for discoverability events tagged with k=8000
    /// (FEDSTR federated learning jobs). Returns exactly `num_dvms`
    /// pubkeys once found, or errors if the timeout elapses first.
    ///
    /// This replaces the need to pass --dvms manually or to grep DVM logs.
    async fn discover_dvms(&self, num_dvms: usize, timeout_secs: u64) -> Result<Vec<PublicKey>> {
        use std::collections::HashSet;

        let timeout    = Duration::from_secs(timeout_secs);
        let poll_every = Duration::from_secs(3);
        let start      = std::time::Instant::now();
        let mut seen: HashSet<String> = HashSet::new();
        let mut found: Vec<PublicKey> = Vec::new();

        println!("  Polling relay every {}s (timeout {}s) ...", poll_every.as_secs(), timeout_secs);

        // Only consider announcements from the last 5 minutes.
        // Prevents picking up stale events from previous runs on the relay.
        let recent_cutoff = Timestamp::now() - 300u64;

        while start.elapsed() < timeout {
            // Filter: kind 31990 + k=8000 + recent only
            let filter = Filter::new()
                .kind(Kind::Custom(31990))
                .custom_tag(
                    SingleLetterTag::lowercase(Alphabet::K),
                    vec!["8000"],
                )
                .since(recent_cutoff);

            let mut events = self
                .client
                .get_events_of(
                    vec![filter],
                    nostr_sdk::EventSource::relays(Some(Duration::from_secs(5))),
                )
                .await
                .unwrap_or_default();

            // Sort newest-first: prefer most recently announced DVM
            events.sort_by(|a, b| b.created_at.cmp(&a.created_at));

            for event in &events {
                let pk_hex = event.pubkey.to_string();
                if !seen.contains(&pk_hex) {
                    seen.insert(pk_hex.clone());
                    found.push(event.pubkey);

                    let name = serde_json::from_str::<serde_json::Value>(&event.content)
                        .ok()
                        .and_then(|v| v["name"].as_str().map(|s| s.to_string()))
                        .unwrap_or_else(|| "unnamed".to_string());

                    let age_secs = Timestamp::now().as_u64()
                        .saturating_sub(event.created_at.as_u64());

                    println!(
                        "  [+] DVM {}/{}: {} | {} ({}s ago)",
                        found.len(), num_dvms, name,
                        event.pubkey.to_bech32().unwrap_or(pk_hex),
                        age_secs,
                    );
                }
            }

            if found.len() >= num_dvms {
                found.truncate(num_dvms);
                println!(
                    "  [OK] Discovery complete: {}/{} DVMs in {:.1}s",
                    num_dvms,
                    num_dvms,
                    start.elapsed().as_secs_f32()
                );
                return Ok(found);
            }

            let remaining = timeout_secs.saturating_sub(start.elapsed().as_secs());
            println!(
                "  [{}/{}] found so far — {}s remaining ...",
                found.len(),
                num_dvms,
                remaining
            );
            tokio::time::sleep(poll_every).await;
        }

        Err(eyre!(
            "DVM discovery timed out after {}s: found only {}/{} DVMs.\n\
             Make sure DVMs are running and connected to the same relay.\n\
             Alternatively, pass pubkeys directly with: --dvms npub1...,npub2,...",
            timeout_secs,
            found.len(),
            num_dvms
        ))
    }

    /// Parse DVM pubkeys from comma-separated string
    fn parse_dvm_pubkeys(&self, dvms_str: &str) -> Result<Vec<PublicKey>> {
        dvms_str.split(',')
            .map(|s| {
                let s = s.trim();
                if s.starts_with("npub") {
                    PublicKey::from_bech32(s)
                        .map_err(|e| eyre!("Invalid npub: {}", e))
                } else {
                    PublicKey::from_str(s)
                        .map_err(|e| eyre!("Invalid hex pubkey: {}", e))
                }
            })
            .collect()
    }
    
    /// Send payment receipt (dummy for Phase 1)
    async fn send_payment(&self, result: &JobResult) -> Result<()> {
        let payment = PaymentReceipt::new(
            result.job_request_id.clone(),
            1000,  // Dummy amount
            self.keys.public_key().to_string(),
            result.service_provider.clone(),
            result.job_id.clone(),
        );
        
        let event = payment.to_event(&self.keys)?;
        self.client.send_event(event).await?;
        
        println!("  ✓ Payment sent to DVM");
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    
    color_eyre::install()?;
    
    // Parse arguments
    let args = Args::parse();
    
    println!("╔═══════════════════════════════════════════════════╗");
    println!("║   FEDSTR Customer (Coordinator)                   ║");
    println!("║   Phase 2 - Proof of Concept                      ║");
    println!("╚═══════════════════════════════════════════════════╝");
    
    // Create customer
    let customer = Customer::new(&args.storage).await?;
    
    // Connect to relays
    customer.connect_to_relays(&args.relays).await?;
    
    println!("\n📡 Customer Information:");
    println!("  Pubkey (hex):  {}", customer.keys.public_key());
    println!("  Pubkey (npub): {}", customer.keys.public_key().to_bech32()?);
    
    // Run federated training
    customer.run_federated_training(args).await?;
    
    Ok(())
}