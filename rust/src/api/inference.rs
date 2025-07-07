use crate::engines::{EngineFactory, EngineType, ModelFormat};
use crate::engines::factory::EngineConfig;
use crate::models::{InferenceError, Tensor, DataType, TensorSpec};
use crate::models::session::Session;
use crate::utils::ModelDetector;
use flutter_rust_bridge::frb;
use std::collections::HashMap;
use std::sync::Arc;
use std::path::PathBuf;
use tokio::sync::RwLock;

/// Session handle for managing loaded models
pub type SessionHandle = u64;

/// Global session storage
static SESSION_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
static SESSIONS: once_cell::sync::Lazy<Arc<RwLock<HashMap<SessionHandle, Session>>>> = 
    once_cell::sync::Lazy::new(|| Arc::new(RwLock::new(HashMap::new())));

/// Input data for inference
#[derive(Debug, Clone)]
pub struct InferenceInput {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub data_type: String,
}

/// Result from inference
#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub data_type: String,
}

// Using TensorSpec from models instead of separate TensorInfo

/// Session information
#[derive(Debug, Clone)]
pub struct SessionInfo {
    pub handle: SessionHandle,
    pub engine_type: String,
    pub input_specs: Vec<TensorSpec>,
    pub output_specs: Vec<TensorSpec>,
}

/// Configuration for inference sessions
#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub engine_type: Option<String>,
    pub gpu_acceleration: bool,
    pub num_threads: Option<usize>,
    pub optimization_level: Option<String>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            engine_type: None,
            gpu_acceleration: true,
            num_threads: None,
            optimization_level: None,
        }
    }
}

/// Load a model with automatic engine detection
pub async fn load_model(model_path: String) -> Result<SessionInfo, InferenceError> {
    load_model_with_config(model_path, SessionConfig::default()).await
}

/// Load a model with specific configuration
pub async fn load_model_with_config(
    model_path: String,
    config: SessionConfig,
) -> Result<SessionInfo, InferenceError> {
    println!("🦀 Rust: load_model_with_config - path: {}, config: {:?}", model_path, config);
    
    // Detect engine type if not specified
    let engine_type = if let Some(engine_str) = &config.engine_type {
        let parsed = parse_engine_type(engine_str)?;
        println!("🦀 Rust: Using specified engine type: {:?}", parsed);
        parsed
    } else {
        let detected = ModelDetector::detect_engine_from_path(&model_path);
        println!("🦀 Rust: Detected engine type: {:?}", detected);
        detected
    };
    
    // Create engine configuration
    let engine_config = EngineConfig::new()
        .with_preferred_engine(engine_type)
        .with_gpu_acceleration(config.gpu_acceleration);
    println!("🦀 Rust: Created engine config: {:?}", engine_config);
    
    // Detect model format
    let format = match ModelDetector::detect_format_from_path(&model_path) {
        Ok(f) => {
            println!("🦀 Rust: Detected format: {:?}", f);
            f
        }
        Err(e) => {
            println!("🦀 Rust: Failed to detect format: {:?}", e);
            return Err(e);
        }
    };
    
    // Create engine and load model
    let engine = match engine_config.create_engine(format) {
        Ok(e) => {
            println!("🦀 Rust: Created engine successfully");
            e
        }
        Err(e) => {
            println!("🦀 Rust: Failed to create engine: {:?}", e);
            return Err(e);
        }
    };
    
    let model = match engine.load_model(&model_path).await {
        Ok(m) => {
            println!("🦀 Rust: Model loaded successfully");
            m
        }
        Err(e) => {
            println!("🦀 Rust: Failed to load model: {:?}", e);
            return Err(e);
        }
    };
    
    // Create session
    let session = Session::new(model, engine_type);
    println!("🦀 Rust: Created session");
    
    // Store session and return info
    let handle = SESSION_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let session_info = create_session_info(handle, &session);
    
    let mut sessions = SESSIONS.write().await;
    sessions.insert(handle, session);
    
    println!("🦀 Rust: Successfully stored session with handle: {}", handle);
    Ok(session_info)
}

/// Load a model from bytes
pub async fn load_model_from_bytes(
    model_bytes: Vec<u8>,
    config: SessionConfig,
) -> Result<SessionInfo, InferenceError> {
    // Detect engine type if not specified
    let engine_type = if let Some(engine_str) = &config.engine_type {
        parse_engine_type(engine_str)?
    } else {
        ModelDetector::detect_engine_from_bytes(&model_bytes)?
    };
    
    // Create engine configuration
    let engine_config = EngineConfig::new()
        .with_preferred_engine(engine_type)
        .with_gpu_acceleration(config.gpu_acceleration);
    
    // Determine model format based on engine type when explicitly specified
    let format = if config.engine_type.is_some() {
        // Use engine type to infer format when explicitly specified
        match engine_type {
            EngineType::Candle => ModelFormat::SafeTensors, // Default for Candle
            EngineType::Linfa => ModelFormat::Linfa,        // Default for Linfa
        }
    } else {
        // Only do content detection when engine type is not specified
        ModelDetector::detect_format_from_bytes(&model_bytes)?
    };
    
    // Create engine and load model
    let engine = engine_config.create_engine(format)?;
    let model = engine.load_from_bytes(&model_bytes).await?;
    
    // Create session
    let session = Session::new(model, engine_type);
    
    // Store session and return info
    let handle = SESSION_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let session_info = create_session_info(handle, &session);
    
    let mut sessions = SESSIONS.write().await;
    sessions.insert(handle, session);
    
    Ok(session_info)
}

/// Load a model with explicit Candle engine
pub async fn load_model_with_candle(model_path: String) -> Result<SessionInfo, InferenceError> {
    println!("🦀 Rust: load_model_with_candle called with path: {}", model_path);
    
    let mut config = SessionConfig::default();
    config.engine_type = Some("candle".to_string());
    
    match load_model_with_config(model_path.clone(), config).await {
        Ok(session_info) => {
            println!("🦀 Rust: Successfully loaded Candle model");
            Ok(session_info)
        }
        Err(e) => {
            println!("🦀 Rust: Failed to load Candle model: {:?}", e);
            println!("🦀 Rust: Error display: {}", e);
            Err(e)
        }
    }
}



/// Train a Linfa model
pub async fn train_linfa_model(
    features: Vec<Vec<f64>>,
    algorithm: String,
    params: HashMap<String, String>,
) -> Result<SessionInfo, InferenceError> {
    #[cfg(feature = "linfa")]
    {
        use crate::engines::LinfaEngine;
        
        // Convert features to tensor (Linfa requires f64 data)
        let num_rows = features.len();
        let num_cols = features.first().map(|row| row.len()).unwrap_or(0);
        let flat_data: Vec<f64> = features.into_iter().flatten().collect();
        
        let features_tensor = Tensor::from_f64(flat_data, vec![num_rows, num_cols])?;
        
        // Create Linfa engine based on algorithm
        let engine = match algorithm.as_str() {
            "kmeans" => {
                let k = params.get("k")
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(2);
                LinfaEngine::k_means(k)?
            }
            "linear_regression" => LinfaEngine::linear_regression()?,
            "svm" => {
                let c = params.get("c")
                    .and_then(|v| v.parse::<f64>().ok())
                    .unwrap_or(1.0);
                LinfaEngine::svm(c)?
            }
            "decision_tree" => {
                let max_depth = params.get("max_depth")
                    .and_then(|v| v.parse::<usize>().ok());
                LinfaEngine::decision_tree(max_depth)?
            }
            _ => return Err(InferenceError::unsupported_format(
                format!("Unsupported Linfa algorithm: {}", algorithm)
            )),
        };
        
        // Train model
        let model = engine.train(&features_tensor, None).await?;
        
        // Create session
        let session = Session::new(model, EngineType::Linfa);
        
        // Store session and return info
        let handle = SESSION_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let session_info = create_session_info(handle, &session);
        
        let mut sessions = SESSIONS.write().await;
        sessions.insert(handle, session);
        
        Ok(session_info)
    }
    #[cfg(not(feature = "linfa"))]
    {
        Err(InferenceError::configuration(
            "Linfa engine not available - compile with 'linfa' feature"
        ))
    }
}

/// Make a prediction with a loaded model
pub async fn predict(
    session_handle: SessionHandle,
    input: InferenceInput,
) -> Result<InferenceResult, InferenceError> {
    let sessions = SESSIONS.read().await;
    let session = sessions.get(&session_handle)
        .ok_or_else(|| InferenceError::model_load("Invalid session handle".to_string()))?;
    
    // Convert input to tensor
    let input_tensor = Tensor::from_f32(input.data, input.shape)?;
    
    // Make prediction
    let output_tensor = session.predict(&input_tensor).await?;
    
    // Convert result
    Ok(InferenceResult {
        data: output_tensor.to_f32_vec()?,
        shape: output_tensor.shape().to_vec(),
        data_type: format!("{:?}", output_tensor.data_type()),
    })
}

/// Make batch predictions
pub async fn predict_batch(
    session_handle: SessionHandle,
    inputs: Vec<InferenceInput>,
) -> Result<Vec<InferenceResult>, InferenceError> {
    let sessions = SESSIONS.read().await;
    let session = sessions.get(&session_handle)
        .ok_or_else(|| InferenceError::model_load("Invalid session handle".to_string()))?;
    
    // Convert inputs to tensors
    let input_tensors: Result<Vec<_>, _> = inputs.into_iter()
        .map(|input| Tensor::from_f32(input.data, input.shape))
        .collect();
    let input_tensors = input_tensors?;
    
    // Make predictions
    let output_tensors = session.predict_batch(&input_tensors).await?;
    
    // Convert results
    let results: Result<Vec<_>, _> = output_tensors.into_iter()
        .map(|tensor| -> Result<InferenceResult, InferenceError> {
            Ok(InferenceResult {
                data: tensor.to_f32_vec()?,
                shape: tensor.shape().to_vec(),
                data_type: format!("{:?}", tensor.data_type()),
            })
        })
        .collect();
    let results = results?;
    
    Ok(results)
}

/// Get session information
pub async fn get_session_info(session_handle: SessionHandle) -> Result<SessionInfo, InferenceError> {
    let sessions = SESSIONS.read().await;
    let session = sessions.get(&session_handle)
        .ok_or_else(|| InferenceError::model_load("Invalid session handle".to_string()))?;
    
    Ok(create_session_info(session_handle, session))
}

/// Dispose of a session and free resources
pub async fn dispose_session(session_handle: SessionHandle) -> Result<(), InferenceError> {
    let mut sessions = SESSIONS.write().await;
    sessions.remove(&session_handle);
    Ok(())
}

/// Get list of available engines
#[frb(sync)]
pub fn get_available_engines() -> Vec<String> {
    EngineFactory::available_engines()
        .into_iter()
        .map(|engine| format!("{:?}", engine).to_lowercase())
        .collect()
}

/// Check if a specific engine is available
#[frb(sync)]
pub fn is_engine_available(engine_type: String) -> bool {
    if let Ok(engine_type) = parse_engine_type(&engine_type) {
        EngineFactory::is_engine_available(engine_type)
    } else {
        false
    }
}

/// Detect engine type from file path
#[frb(sync)]
pub fn detect_engine_from_path(model_path: String) -> String {
    let engine_type = ModelDetector::detect_engine_from_path(&model_path);
    format!("{:?}", engine_type).to_lowercase()
}

/// Detect engine type from bytes
#[frb(sync)]
pub fn detect_engine_from_bytes(model_bytes: Vec<u8>) -> Result<String, InferenceError> {
    let engine_type = ModelDetector::detect_engine_from_bytes(&model_bytes)?;
    Ok(format!("{:?}", engine_type).to_lowercase())
}

/// Load a model from a URL with caching
pub async fn load_model_from_url(
    url: String,
    cache: bool,
    cache_key: Option<String>,
) -> Result<SessionInfo, InferenceError> {
    let model_bytes = if cache {
        // Try to load from cache first
        let key = cache_key.unwrap_or_else(|| url_to_cache_key(&url));
        if let Some(cached_bytes) = load_from_cache(&key).await? {
            cached_bytes
        } else {
            // Download and cache
            let bytes = download_model(&url).await?;
            save_to_cache(&key, &bytes).await?;
            bytes
        }
    } else {
        // Direct download without caching
        download_model(&url).await?
    };

    load_model_from_bytes(model_bytes, SessionConfig::default()).await
}

/// Load a model from local file path
pub async fn load_model_from_file(file_path: String) -> Result<SessionInfo, InferenceError> {
    load_model_with_config(file_path, SessionConfig::default()).await
}

/// HuggingFace integration - load model from hub
pub async fn load_from_huggingface(
    repo: String,
    revision: Option<String>,
    filename: Option<String>,
) -> Result<SessionInfo, InferenceError> {
    let revision = revision.unwrap_or_else(|| "main".to_string());
    let filename = filename.unwrap_or_else(|| "model.safetensors".to_string());
    
    // Build Hugging Face URL
    let url = format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        repo, revision, filename
    );
    
    // Use caching with a descriptive key
    let cache_key = format!("hf_{}_{}/{}", repo.replace('/', "_"), revision, filename);
    
    load_model_from_url(url, true, Some(cache_key)).await
}

/// Download a model from URL
async fn download_model(url: &str) -> Result<Vec<u8>, InferenceError> {
    let client = reqwest::Client::builder()
        .user_agent("inference-flutter/1.0.0")
        .timeout(std::time::Duration::from_secs(300)) // 5 minute timeout
        .build()
        .map_err(|e| InferenceError::model_load(format!("Failed to create HTTP client: {}", e)))?;

    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| InferenceError::model_load(format!("Failed to download model: {}", e)))?;

    if !response.status().is_success() {
        return Err(InferenceError::model_load(format!(
            "HTTP error {}: Failed to download model from {}",
            response.status(),
            url
        )));
    }

    let bytes = response
        .bytes()
        .await
        .map_err(|e| InferenceError::model_load(format!("Failed to read response bytes: {}", e)))?;

    Ok(bytes.to_vec())
}

/// Get cache directory path
fn get_cache_dir() -> Result<PathBuf, InferenceError> {
    let cache_dir = if let Some(cache_home) = std::env::var_os("XDG_CACHE_HOME") {
        PathBuf::from(cache_home).join("inference")
    } else if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".cache").join("inference")
    } else if let Some(app_data) = std::env::var_os("APPDATA") {
        PathBuf::from(app_data).join("inference").join("cache")
    } else {
        return Err(InferenceError::model_load(
            "Unable to determine cache directory".to_string(),
        ));
    };

    Ok(cache_dir)
}

/// Load model from cache
async fn load_from_cache(cache_key: &str) -> Result<Option<Vec<u8>>, InferenceError> {
    let cache_dir = get_cache_dir()?;
    let cache_path = cache_dir.join(format!("{}.bin", cache_key));

    if cache_path.exists() {
        match tokio::fs::read(&cache_path).await {
            Ok(bytes) => Ok(Some(bytes)),
            Err(e) => {
                eprintln!("Warning: Failed to read from cache: {}", e);
                Ok(None)
            }
        }
    } else {
        Ok(None)
    }
}

/// Save model to cache
async fn save_to_cache(cache_key: &str, data: &[u8]) -> Result<(), InferenceError> {
    let cache_dir = get_cache_dir()?;
    
    // Create cache directory if it doesn't exist
    if let Err(e) = tokio::fs::create_dir_all(&cache_dir).await {
        eprintln!("Warning: Failed to create cache directory: {}", e);
        return Ok(()); // Don't fail if caching fails
    }

    let cache_path = cache_dir.join(format!("{}.bin", cache_key));
    
    if let Err(e) = tokio::fs::write(&cache_path, data).await {
        eprintln!("Warning: Failed to write to cache: {}", e);
    }

    Ok(())
}

/// Convert URL to cache key
fn url_to_cache_key(url: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    url.hash(&mut hasher);
    format!("url_{:x}", hasher.finish())
}

/// Clear model cache
pub async fn clear_cache() -> Result<(), InferenceError> {
    let cache_dir = get_cache_dir()?;
    
    if cache_dir.exists() {
        tokio::fs::remove_dir_all(&cache_dir)
            .await
            .map_err(|e| InferenceError::model_load(format!("Failed to clear cache: {}", e)))?;
    }
    
    Ok(())
}

/// Get cache size in bytes
pub async fn get_cache_size() -> Result<u64, InferenceError> {
    let cache_dir = get_cache_dir()?;
    
    if !cache_dir.exists() {
        return Ok(0);
    }

    let mut total_size = 0u64;
    let mut entries = tokio::fs::read_dir(&cache_dir)
        .await
        .map_err(|e| InferenceError::model_load(format!("Failed to read cache directory: {}", e)))?;

    while let Ok(Some(entry)) = entries.next_entry().await {
        if let Ok(metadata) = entry.metadata().await {
            total_size += metadata.len();
        }
    }

    Ok(total_size)
}

/// Helper function to create session info
fn create_session_info(handle: SessionHandle, session: &Session) -> SessionInfo {
    let input_specs = session.input_specs().to_vec();
    let output_specs = session.output_specs().to_vec();
    
    SessionInfo {
        handle,
        engine_type: format!("{:?}", session.engine_type()).to_lowercase(),
        input_specs,
        output_specs,
    }
}

/// Helper function to parse engine type from string
fn parse_engine_type(engine_str: &str) -> Result<EngineType, InferenceError> {
    match engine_str.to_lowercase().as_str() {
        "candle" => Ok(EngineType::Candle),
        "linfa" => Ok(EngineType::Linfa),

        _ => Err(InferenceError::unsupported_format(
            format!("Unknown engine type: {}", engine_str)
        )),
    }
}

/// Helper function to parse data type from string
fn parse_data_type(data_type_str: &str) -> Result<DataType, InferenceError> {
    match data_type_str.to_lowercase().as_str() {
        "f32" | "float32" => Ok(DataType::F32),
        "f64" | "float64" => Ok(DataType::F64),
        "i32" | "int32" => Ok(DataType::I32),
        "i64" | "int64" => Ok(DataType::I64),
        "u8" | "uint8" => Ok(DataType::U8),
        "u32" | "uint32" => Ok(DataType::U32),
        _ => Err(InferenceError::unsupported_format(
            format!("Unknown data type: {}", data_type_str)
        )),
    }
}

/// Initialize the inference library
#[frb(init)]
pub fn init_inference() {
    flutter_rust_bridge::setup_default_user_utils();
} 