use glob::Pattern;
use std::fs;

#[derive(Debug)]
pub struct Router {
    routes: Vec<(Pattern, Vec<String>)>,
}

pub trait PathResolution {
    fn get_servers(&self, path: &str) -> Option<&Vec<String>>;
}

impl Router {
    /// Load routes from a TOML configuration file
    pub fn load_from_file(file_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let config_content = fs::read_to_string(file_path)?;
        let parsed: toml::Value = toml::from_str(&config_content)?;

        let mut routes = Vec::new();
        if let Some(paths) = parsed.get("paths").and_then(|p| p.as_table()) {
            for (path, servers) in paths {
                if let Some(servers_array) = servers.as_array() {
                    let server_list = servers_array
                        .iter()
                        .filter_map(|s| s.as_str().map(String::from))
                        .collect();

                    let pattern = Pattern::new(path)?;
                    routes.push((pattern, server_list));
                }
            }
        }

        // Sort routes by pattern specificity (longer patterns first)
        routes.sort_by(|a, b| b.0.as_str().len().cmp(&a.0.as_str().len()));

        Ok(Router { routes })
    }
}

impl PathResolution for Router {
    /// Get the list of servers for a given path
    fn get_servers(&self, path: &str) -> Option<&Vec<String>> {
        for (pattern, servers) in &self.routes {
            if pattern.matches(path) {
                return Some(servers);
            }
        }
        None
    }
}
