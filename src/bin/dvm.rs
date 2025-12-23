/// FEDSTR Service Provider (DVM) - Phase 2
/// 
/// Implements Algorithm 4 from the FEDSTR paper.
/// 
/// This DVM:
/// 1. Connects to NOSTR relays
/// 2. Announces capabilities (kind 31990)
/// 3. Listens for JobRequest events (kind 8000)
/// 4. Trains models using PyTorch
/// 5. Publishes JobResult events (kind 6000)
/// 6. Sends periodic JobFeedback (kind 7000)

use fedstr_poc::{
    JobRequest, JobResult, JobFeedback,
    PyTorchTrainer, PaymentRequest,
    ModelStorage, StorageBackend,
};

use nostr::{Keys, Kind, Event, EventBuilder, Filter, Tag, Timestamp, prelude::*};
use nostr_sdk::Client;
use color_eyre::eyre::Result;
use std::time::Duration;
use std::collections::HashSet;
use chrono::{DateTime, Utc};
use clap::Parser;
use base64::{Engine as _, engine::general_purpose};

/// Command-line arguments for DVM
#[derive(Parser, Debug)]
#[command(name = "fedstr-dvm")]
#[command(about = "FEDSTR Service Provider (Data Vending Machine)", long_about = None)]
struct Args {
    /// DVM identifier (for multi-DVM testing)
    #[arg(short, long, default_value = "0")]
    id: usize,
    
    /// Path to keypair file (optional, generates new if not provided)
    #[arg(short, long)]
    keypair: Option<String>,
    
    /// Relays to connect to
    #[arg(long, default_values_t = vec![
    //"ws://localhost:8080".to_string(),
    "wss://nos.lol".to_string(),
    //"wss://relay.nostr.band".to_string(),
    ])]
    relays: Vec<String>,
    
    /// Storage backend for model parameters
    #[arg(long, default_value = "local")]
    storage: String,  // "local", "nostr.build", or "http://server"
}

/// Service Provider (DVM) struct
struct ServiceProvider {
    id: usize,
    client: Client,
    keys: Keys,
    trainer: PyTorchTrainer,
    storage: ModelStorage,
    processed_events: HashSet<nostr::EventId>,
}

impl ServiceProvider {
    /// Create a new service provider
    async fn new(id: usize, keypair_path: Option<String>, storage_arg: String) -> Result<Self> {
        // Load or generate keys
        let keys = if let Some(_path) = keypair_path {
            // TODO: Load from file
            println!("⚠ Keypair loading not implemented yet, generating new keys");
            Keys::generate()
        } else {
            Keys::generate()
        };
        
        let client = Client::new(&keys);
        let trainer = PyTorchTrainer::new()?;
        
        // Configure storage backend
        let storage = if storage_arg == "local" {
            ModelStorage::new(StorageBackend::LocalFile("/tmp/fedstr_models".to_string()))
        } else if storage_arg.starts_with("http") {
            ModelStorage::new(StorageBackend::HttpServer(storage_arg))
        } else {
            ModelStorage::new(StorageBackend::LocalFile(storage_arg))
        };
        
        Ok(Self {
            id,
            client,
            keys,
            trainer,
            storage,
            processed_events: HashSet::new(),
        })
    }
    
    /// Announce DVM capabilities (kind 31990)
    async fn announce_capabilities(&self) -> Result<()> {
        let content = serde_json::json!({
            "name": format!("FEDSTR DVM {}", self.id),
            "about": "Federated Learning Service Provider",
            "capabilities": {
                "models": ["tiny_linear"],
                "datasets": ["mnist", "fashion-mnist"],
                "max_epochs": 10,
                "max_batch_size": 128,
            }
        });
        
        let tags = vec![
            Tag::parse(&["k", "8000"])?,  // Supports kind 8000
            Tag::parse(&["t", "federated_learning"])?,
        ];
        
        let event = EventBuilder::new(
            Kind::Custom(31990),
            serde_json::to_string(&content)?,
            tags,
        ).to_event(&self.keys)?;
        
        self.client.send_event(event).await?;
        println!("✓ Announced capabilities");
        Ok(())
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
        println!("👂 Polling for new JobRequests...");
        tokio::time::sleep(Duration::from_secs(20)).await;
        Ok(())
    }
    
    /// Main event loop (Algorithm 4)
    async fn start(&mut self, relay_urls: Vec<String>) -> Result<()> {
        // Connect to relays
        self.connect_to_relays(&relay_urls).await?;
        
        // Print DVM info
        println!("\n📡 FEDSTR Service Provider (DVM) Started");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("DVM ID:       {}", self.id);
        println!("Pubkey (hex): {}", self.keys.public_key());
        println!("Pubkey (npub):{}", self.keys.public_key().to_bech32()?);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        // Announce capabilities
        println!("\n📢 Announcing capabilities...");
        self.announce_capabilities().await?;
        
        // Set up subscription filter
        let since = Timestamp::now();
        println!("\n👂 Listening for JobRequest events (kind 8000)...");
        println!("Start time: {}", 
            DateTime::<Utc>::from_timestamp(since.as_u64() as i64, 0)
                .unwrap_or_default()
                .format("%Y-%m-%d %H:%M:%S UTC"));
        
        let filter = Filter::new()
            .kind(Kind::Custom(8000))  // JobRequest
            .pubkey(self.keys.public_key())  // Only events tagging this DVM
            .since(since);
        
        let _ = self.client.subscribe(vec![filter.clone()], None).await;
        
        // Main event loop
        loop {
            match self.client.get_events_of(
                vec![filter.clone()],
                nostr_sdk::EventSource::relays(None),
            ).await {
                Ok(events) => {
                    for event in events {
                        // Skip already processed
                        if self.processed_events.contains(&event.id) {
                            continue;
                        }
                        
                        // Skip historical events
                        if event.created_at.as_u64() < since.as_u64() {
                            continue;
                        }
                        
                        // Process the job request
                        if let Err(e) = self.handle_job_request(event.clone()).await {
                            eprintln!("❌ Error handling request: {}", e);
                        }
                        
                        self.processed_events.insert(event.id);
                    }
                }
                Err(e) => {
                    eprintln!("❌ Error fetching events: {}", e);
                }
            }
            
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    }
    
    /// Handle incoming JobRequest (Algorithm 4, steps 4-20)
    async fn handle_job_request(&self, event: Event) -> Result<()> {
        println!("\n╔═══════════════════════════════════════╗");
        println!("║     New JobRequest Received          ║");
        println!("╚═══════════════════════════════════════╝");
        println!("Event ID: {}", event.id.to_hex());
        
        // Parse JobRequest
        let job_request = match JobRequest::from_event(&event) {
            Ok(req) => req,
            Err(e) => {
                eprintln!("❌ Failed to parse JobRequest: {}", e);
                return Err(e);
            }
        };
        
        println!("Job ID:       {}", job_request.job_id);
        println!("Dataset:      {}", job_request.dataset);
        println!("Model:        {}", job_request.model_type);
        println!("Data range:   {} - {}", 
            job_request.data_split.start_idx,
            job_request.data_split.end_idx);
        println!("Round:        {}/{}", 
            job_request.round,
            job_request.total_rounds);
        
        // Send payment request (dummy for Phase 1)
        println!("\n₿ Sending payment request...");
        self.send_payment_request(&job_request).await?;
        
        // Download initial params if URL provided (rounds 2+)
        let initial_params_base64 = if let Some(ref params) = job_request.current_model_params {
            if params.starts_with("http://") || params.starts_with("https://") || params.starts_with("file://") {
                // It's a URL - download it
                println!("\n📥 Downloading initial model from: {}", params);
                let model_bytes = self.storage.download_model(params).await?;
                println!("  ✓ Downloaded {} bytes", model_bytes.len());
                
                // 🔐 CRITICAL: Verify hash before using model
                println!("\n  🔐 Hash Verification:");
                if let Some(ref expected_hash) = job_request.current_model_params_hash {
                    println!("     Expected hash: {}", expected_hash);
                    match job_request.verify_model_hash(&model_bytes) {
                        Ok(true) => {
                            println!("     ✅ Hash verified - Model integrity confirmed!");
                        }
                        Ok(false) => {
                            println!("     ⚠️  Unexpected: Hash verification returned false but hash was present");
                        }
                        Err(e) => {
                            println!("     ❌ Hash verification FAILED!");
                            return Err(e);
                        }
                    }
                } else {
                    println!("     ℹ️  No hash provided in JobRequest (round {})", job_request.round);
                    println!("     ⚠️  WARNING: Cannot verify model integrity!");
                }
                
                // Encode to base64 for Python
                let encoded = general_purpose::STANDARD.encode(&model_bytes);
                println!("  ✅ Passing downloaded model to training ({} bytes)", model_bytes.len());
                Some(encoded)
            } else {
                // It's already base64 (shouldn't happen, but handle it)
                Some(params.clone())
            }
        } else {
            None
        };
        
        // Perform training (no intermediate feedback to avoid rate-limiting)
        println!("\n⚙️  Starting training (this will take 5-10 minutes)...");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        let training_result = self.trainer.train_tiny_model(
            &job_request.dataset,
            job_request.data_split.start_idx,
            job_request.data_split.end_idx,
            job_request.epochs,
            job_request.batch_size,
            initial_params_base64,
        )?;
        
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("✓ Training completed!");
        println!("  Final Loss:     {:.4}", training_result.metrics.final_loss);
        println!("  Final Accuracy: {:.2}%", training_result.metrics.final_accuracy);
        println!("  Samples:        {}", training_result.training_samples);
        
        // Check if we should use external storage (model > 50KB)
        let use_external_storage = training_result.model_params.size_bytes > 50_000;
        
        let final_model_params = if use_external_storage {
            println!("\n📤 Model too large ({}KB), uploading to external storage...", 
                training_result.model_params.size_bytes / 1024);
            
            // Get the base64 data to upload
            let model_base64 = training_result.model_params.params_base64
                .as_ref()
                .ok_or_else(|| color_eyre::eyre::eyre!("No model params"))?;
            
            let model_bytes = general_purpose::STANDARD.decode(model_base64)?;
            
            // Upload to storage
            let upload_result = self.storage.upload_model(
                &model_bytes,
                &training_result.model_params.params_hash,
            ).await?;
            
            println!("  ✓ Uploaded to: {}", upload_result.url);
            
            // Create ModelParams with URL
            fedstr_poc::protocol::ModelParams::from_url(
                upload_result.url,
                training_result.model_params.params_hash.clone(),
                training_result.model_params.size_bytes,
            )
        } else {
            println!("\n✓ Model small enough ({}KB), embedding directly in event", 
                training_result.model_params.size_bytes / 1024);
            training_result.model_params
        };
        
        // Send job result
        println!("\n📤 Publishing JobResult (storage: {})...", 
            final_model_params.storage_info());
        
        // DEBUG: Check what we're actually sending
        println!("  DEBUG: params_base64 present: {}", final_model_params.params_base64.is_some());
        println!("  DEBUG: params_url present: {}", final_model_params.params_url.is_some());
        if let Some(ref url) = final_model_params.params_url {
            println!("  DEBUG: URL: {}", url);
        }
        if let Some(ref b64) = final_model_params.params_base64 {
            println!("  DEBUG: base64 length: {} bytes", b64.len());
        }
        
        let training_result_final = fedstr_poc::ml::TrainingResult {
            model_params: final_model_params,
            metrics: training_result.metrics,
            training_samples: training_result.training_samples,
        };
        
        self.send_job_result(&event, &job_request, training_result_final).await?;
        
        println!("\n✓ Job completed and result published successfully!");
        println!("  Job ID: {}", job_request.job_id);
        println!("  Round:  {}", job_request.round);
        println!("\n👂 Ready for next JobRequest...");
        println!("\n");
        Ok(())
    }
    
    /// Send payment request (dummy for Phase 1)
    async fn send_payment_request(&self, job_request: &JobRequest) -> Result<()> {
        let payment = PaymentRequest::new(
            job_request.payment_msats,
            self.keys.public_key().to_string(),
            job_request.job_id.clone(),
        );
        
        let event = payment.to_event(&self.keys)?;
        self.client.send_event(event).await?;
        
        println!("  ✓ Payment request sent (dummy)");
        Ok(())
    }
    
    /// Send JobFeedback event (kind 7000)
    async fn send_feedback(&self, _job_request: &JobRequest, feedback: JobFeedback) -> Result<()> {
        let event = feedback.to_event(&self.keys)?;
        self.client.send_event(event).await?;
        
        println!("  ✓ Feedback sent: {:?}", feedback.status);
        Ok(())
    }
    
    /// Send JobResult event (kind 6000)
    async fn send_job_result(
        &self,
        request_event: &Event,
        job_request: &JobRequest,
        training_result: fedstr_poc::ml::TrainingResult,
    ) -> Result<()> {
        let job_result = JobResult::new(
            request_event.id.to_hex(),
            job_request.job_id.clone(),
            self.keys.public_key().to_string(),
            job_request.round,
            training_result.model_params,
            training_result.metrics,
            training_result.training_samples,
        );
        
        let event = job_result.to_event(&self.keys)?;
        self.client.send_event(event.clone()).await?;
        
        println!("  ✓ JobResult published (Event ID: {})", event.id.to_hex().chars().take(12).collect::<String>());
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
    
    println!("╔═══════════════════════════════════════╗");
    println!("║   FEDSTR Service Provider (DVM)      ║");
    println!("║   Phase 2 - Proof of Concept         ║");
    println!("╚═══════════════════════════════════════╝\n");
    
    // Create and start DVM
    let mut dvm = ServiceProvider::new(args.id, args.keypair, args.storage).await?;
    dvm.start(args.relays).await?;
    
    Ok(())
}
