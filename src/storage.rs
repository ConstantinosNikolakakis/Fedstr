use color_eyre::eyre::{eyre, Result};
use reqwest::multipart::{Form, Part};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;
use tokio::fs;

/// Storage backend configuration
#[derive(Debug, Clone)]
pub enum StorageBackend {
    /// Local file storage
    LocalFile(String),
    
    /// Blossom server (free Nostr-native file storage)
    Blossom {
        server_url: String,
    },
    
    /// HTTP server storage
    HttpServer(String),
}

/// Response from upload operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UploadResponse {
    pub url: String,
    pub hash: String,
}

/// Model storage handler
pub struct ModelStorage {
    backend: StorageBackend,
    client: reqwest::Client,
    auth_token: Option<String>,
}

impl ModelStorage {
    /// Create new storage instance
    pub fn new(backend: StorageBackend) -> Self {
        // Try to get auth token from environment
        let auth_token = std::env::var("FEDSTR_AUTH_TOKEN").ok();
        
        if auth_token.is_some() && matches!(backend, StorageBackend::HttpServer(_)) {
            println!("🔐 Using authentication token for HTTP uploads");
        }
        
        Self { 
            backend,
            auth_token,
            client: reqwest::Client::new(),
        }
    }
    
    /// Upload model to storage
    /// 
    /// # Arguments
    /// * `model_bytes` - Raw model bytes
    /// * `hash` - SHA256 hash of the model
    /// 
    /// # Returns
    /// URL where model can be downloaded
    pub async fn upload_model(&self, model_bytes: &[u8], hash: &str) -> Result<UploadResponse> {
        match &self.backend {
            StorageBackend::LocalFile(base_path) => {
                self.upload_local(model_bytes, hash, base_path).await
            }
            StorageBackend::Blossom { server_url } => {
                self.upload_blossom(model_bytes, hash, server_url).await
            }
            StorageBackend::HttpServer(server_url) => {
                self.upload_http(model_bytes, hash, server_url).await
            }
        }
    }
    
    /// Download model from URL
    pub async fn download_model(&self, url: &str) -> Result<Vec<u8>> {
        if url.starts_with("file://") {
            // Local file
            let path = url.strip_prefix("file://").unwrap();
            let bytes = fs::read(path).await?;
            println!("✓ Downloaded {} bytes from local file", bytes.len());
            Ok(bytes)
        } else if url.starts_with("http://") || url.starts_with("https://") {
            // HTTP download
            let response = self.client.get(url).send().await?;
            
            if !response.status().is_success() {
                return Err(eyre!("HTTP error: {}", response.status()));
            }
            
            let bytes = response.bytes().await?.to_vec();
            println!("✓ Downloaded {} bytes from {}", bytes.len(), url);
            Ok(bytes)
        } else {
            Err(eyre!("Unsupported URL scheme: {}", url))
        }
    }
    
    /// Upload to local file storage
    async fn upload_local(
        &self,
        model_bytes: &[u8],
        hash: &str,
        base_path: &str,
    ) -> Result<UploadResponse> {
        // Create directory if it doesn't exist
        fs::create_dir_all(base_path).await?;
        
        // Use first 12 chars of hash as filename
        let short_hash = &hash[..12];
        let filename = format!("model_{}.bin", short_hash);
        let filepath = Path::new(base_path).join(&filename);
        
        println!("💾 Saving to local file: {}", base_path);
        
        // Write file
        fs::write(&filepath, model_bytes).await?;
        
        println!("✓ Saved to: {}", filepath.display());
        
        // Return file:// URL
        let url = format!("file://{}", filepath.display());
        
        Ok(UploadResponse {
            url,
            hash: hash.to_string(),
        })
    }
    
    /// Upload to Blossom server (free Nostr file storage)
    /// Requires Nostr authentication
    async fn upload_blossom(
        &self,
        model_bytes: &[u8],
        hash: &str,
        server_url: &str,
    ) -> Result<UploadResponse> {
        println!("🌸 Uploading to Blossom server: {}", server_url);
        
        // Blossom requires authentication via Nostr event signature
        // We need to get the Nostr keys from environment or generate them
        
        // For now, try unauthenticated upload first
        // TODO: Add proper Nostr authentication if required
        
        let upload_url = format!("{}/upload", server_url.trim_end_matches('/'));
        
        println!("  📤 Uploading to: {}", upload_url);
        
        // Create multipart form with 'file' field
        let part = Part::bytes(model_bytes.to_vec())
            .file_name(format!("{}.bin", &hash[..12]))
            .mime_str("application/octet-stream")?;
        
        let form = Form::new()
            .part("file", part);
        
        // POST multipart form
        let response = self.client
            .post(&upload_url)
            .multipart(form)
            .send()
            .await?;
        
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            
            // Check if authentication is required
            if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
                return Err(eyre!(
                    "Blossom server requires authentication. This feature is not yet implemented.\n\
                    Try using local storage instead: --storage local"
                ));
            }
            
            return Err(eyre!("Blossom upload failed: {} - {}", status, error_text));
        }
        
        // Parse JSON response
        let response_json: serde_json::Value = response.json().await?;
        
        // Get URL from response - Blossom returns: {"url": "https://..."}
        let url = response_json.get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| eyre!("No URL in Blossom response: {:?}", response_json))?
            .to_string();
        
        println!("  ✓ Uploaded to: {}", url);
        println!("  📊 Size: {:.2} KB", model_bytes.len() as f64 / 1024.0);
        
        Ok(UploadResponse {
            url,
            hash: hash.to_string(),
        })
    }
    
    /// Upload to HTTP server
    async fn upload_http(
            &self,
            model_bytes: &[u8],
            hash: &str,
            server_url: &str,
        ) -> Result<UploadResponse> {
            println!("📤 Uploading to HTTP server: {}", server_url);
            
            // Create filename from hash
            let short_hash = &hash[..12];
            let filename = format!("model_{}.bin", short_hash);
            
            // PUT to server with filename in URL
            let upload_url = format!("{}/{}", server_url.trim_end_matches('/'), filename);
            
            println!("  📤 Uploading to: {}", upload_url);
            
            // Simple PUT request with raw bytes
            let mut request = self.client
                .put(&upload_url)
                .header("Content-Type", "application/octet-stream");
            
            // Add authentication header if token is available
            if let Some(token) = &self.auth_token {
                request = request.header("Authorization", format!("Bearer {}", token));
            }
            
            let response = request
                .body(model_bytes.to_vec())
                .send()
                .await?;
            
            if !response.status().is_success() {
                let status = response.status();
                let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                return Err(eyre!("HTTP upload failed: {} - {}", status, error_text));
            }
            
            // Parse JSON response from file_server.py
            let result: serde_json::Value = response.json().await?;
            let url = result.get("url")
                .and_then(|v| v.as_str())
                .ok_or_else(|| eyre!("No URL in server response"))?
                .to_string();
            
            println!("  ✓ Uploaded to: {}", url);
            println!("  📊 Size: {:.2} KB", model_bytes.len() as f64 / 1024.0);
            
            Ok(UploadResponse {
                url,
                hash: hash.to_string(),
            })
        }
    }

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_local_storage() {
        let storage = ModelStorage::new(
            StorageBackend::LocalFile("/tmp/test_storage".to_string())
        );
        
        let test_data = b"test model data";
        let mut hasher = Sha256::new();
        hasher.update(test_data);
        let hash = format!("{:x}", hasher.finalize());
        
        // Upload
        let result = storage.upload_model(test_data, &hash).await.unwrap();
        assert!(result.url.starts_with("file://"));
        
        // Download
        let downloaded = storage.download_model(&result.url).await.unwrap();
        assert_eq!(downloaded, test_data);
    }
}
