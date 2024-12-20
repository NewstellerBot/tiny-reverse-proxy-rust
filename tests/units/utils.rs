#[cfg(test)]
mod tests {
    use tiny_reverse_proxy_rust::utils::{create_temp_config, extract_path, load_port_from_config};
    #[test]
    fn test_load_port_valid_config() {
        let config_content = r#"
            port = 8080
        "#;

        let temp_file = create_temp_config(config_content);
        let port = load_port_from_config(temp_file.path().to_str().unwrap()).unwrap();

        assert_eq!(port, "8080");
    }

    #[test]
    fn test_load_port_missing_port() {
        let config_content = r#"
            host = "127.0.0.1"
        "#;

        let temp_file = create_temp_config(config_content);
        let result = load_port_from_config(temp_file.path().to_str().unwrap());

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "Port not found in configuration file."
        );
    }

    #[test]
    fn test_load_port_invalid_config() {
        let config_content = "invalid_toml_content";

        let temp_file = create_temp_config(config_content);
        let result = load_port_from_config(temp_file.path().to_str().unwrap());

        assert!(result.is_err());
    }

    #[test]
    fn test_load_port_file_not_found() {
        let result = load_port_from_config("nonexistent.toml");
        assert!(result.is_err());
    }
    #[test]
    fn test_extract_path_valid_request() {
        let request = b"GET /api/resource HTTP/1.1\r\nHost: example.com\r\n\r\n";
        let path = extract_path(request);
        assert_eq!(path, "/api/resource");
    }

    #[test]
    fn test_extract_path_no_path() {
        let request = b"GET  HTTP/1.1\r\nHost: example.com\r\n\r\n";
        let path = extract_path(request);
        assert_eq!(path, "/");
    }

    #[test]
    fn test_extract_path_empty_buffer() {
        let request = b"";
        let path = extract_path(request);
        assert_eq!(path, "/");
    }

    #[test]
    fn test_extract_path_invalid_format() {
        let request = b"INVALID_REQUEST";
        let path = extract_path(request);
        assert_eq!(path, "/");
    }
}
