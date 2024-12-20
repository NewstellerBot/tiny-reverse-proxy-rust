use std::fs;
use std::io::Write;
use tempfile::NamedTempFile;

/// Extracts path from HTTP request
pub fn extract_path(buffer: &[u8]) -> String {
    let request = String::from_utf8_lossy(buffer);
    if let Some(start) = request.find(' ') {
        if let Some(end) = request[start + 1..].find(' ') {
            let path = &request[start + 1..start + 1 + end];
            if !path.is_empty() {
                return path.to_string();
            }
        }
    }
    "/".to_string()
}

/// Loads port number from config file
pub fn load_port_from_config(file_path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let config_content = fs::read_to_string(file_path)?;
    let parsed: toml::Value = toml::from_str(&config_content)?;
    if let Some(port) = parsed.get("port").and_then(|p| p.as_integer()) {
        Ok(port.to_string())
    } else {
        Err("Port not found in configuration file.".into())
    }
}

/// Creates a temporary file with the given content
#[allow(dead_code)]
pub fn create_temp_config(content: &str) -> NamedTempFile {
    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(content.as_bytes()).unwrap();
    temp_file
}
