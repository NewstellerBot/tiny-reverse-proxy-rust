use std::path::{Path, PathBuf};

use dashmap::DashMap;
use semantic_safety_protocol::ProjectSemanticPolicy;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedProjectIndex {
    pub policy: ProjectSemanticPolicy,
    pub exemplar_embeddings: Vec<Vec<f32>>,
    pub stored_exemplar_count: u64,
}

#[derive(Clone)]
pub struct FileProjectIndexStore {
    root: PathBuf,
    cache: DashMap<String, PersistedProjectIndex>,
}

impl FileProjectIndexStore {
    pub fn new(root: PathBuf) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        std::fs::create_dir_all(&root)?;
        Ok(Self {
            root,
            cache: DashMap::new(),
        })
    }

    pub fn load_all(
        &self,
    ) -> Result<Vec<PersistedProjectIndex>, Box<dyn std::error::Error + Send + Sync>> {
        let mut entries = Vec::new();
        for item in std::fs::read_dir(&self.root)? {
            let item = item?;
            if item.file_type()?.is_file()
                && item.path().extension().and_then(|ext| ext.to_str()) == Some("json")
            {
                let bytes = std::fs::read(item.path())?;
                let record: PersistedProjectIndex = serde_json::from_slice(&bytes)?;
                self.cache
                    .insert(record.policy.project_id.clone(), record.clone());
                entries.push(record);
            }
        }
        Ok(entries)
    }

    pub fn upsert(
        &self,
        index: PersistedProjectIndex,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let path = self.path_for(&index.policy.project_id);
        let bytes = serde_json::to_vec_pretty(&index)?;
        std::fs::write(path, bytes)?;
        self.cache.insert(index.policy.project_id.clone(), index);
        Ok(())
    }

    pub fn delete(&self, project_id: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let path = self.path_for(project_id);
        if Path::new(&path).exists() {
            std::fs::remove_file(path)?;
        }
        self.cache.remove(project_id);
        Ok(())
    }

    pub fn get(&self, project_id: &str) -> Option<PersistedProjectIndex> {
        self.cache
            .get(project_id)
            .map(|entry| entry.value().clone())
    }

    fn path_for(&self, project_id: &str) -> PathBuf {
        self.root.join(format!("{project_id}.json"))
    }
}
