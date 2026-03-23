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
        let mut refreshed = Vec::new();
        for item in std::fs::read_dir(&self.root)? {
            let item = item?;
            if item.file_type()?.is_file()
                && item.path().extension().and_then(|ext| ext.to_str()) == Some("json")
            {
                let bytes = std::fs::read(item.path())?;
                let record: PersistedProjectIndex = serde_json::from_slice(&bytes)?;
                refreshed.push((record.policy.project_id.clone(), record.clone()));
                entries.push(record);
            }
        }
        self.cache.clear();
        for (project_id, entry) in refreshed {
            self.cache.insert(project_id, entry);
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

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_safety_protocol::{ProjectSemanticPolicy, SemanticEntity, SemanticTopic};
    use tempfile::tempdir;

    fn sample_index(project_id: &str, version: &str) -> PersistedProjectIndex {
        PersistedProjectIndex {
            policy: ProjectSemanticPolicy {
                project_id: project_id.to_string(),
                version: version.to_string(),
                enabled: true,
                entities: vec![SemanticEntity {
                    entity_id: "entity-1".to_string(),
                    name: "Entity 1".to_string(),
                    aliases: vec!["entity1".to_string()],
                }],
                topics: vec![SemanticTopic {
                    topic_id: "topic-1".to_string(),
                    name: "Topic 1".to_string(),
                    exemplars: vec!["example".to_string()],
                    rerank_threshold: 0.5,
                    require_entity_match: false,
                }],
                updated_at: "1".to_string(),
            },
            exemplar_embeddings: vec![vec![0.1, 0.2, 0.3]],
            stored_exemplar_count: 1,
        }
    }

    #[test]
    fn load_all_refreshes_cache_after_deleted_files() {
        let dir = tempdir().unwrap();
        let store = FileProjectIndexStore::new(dir.path().to_path_buf()).unwrap();
        let first = sample_index("project-a", "1");
        store.upsert(first.clone()).unwrap();
        assert_eq!(store.get("project-a").unwrap().policy.version, "1");

        std::fs::remove_file(dir.path().join("project-a.json")).unwrap();

        let loaded = store.load_all().unwrap();
        assert!(loaded.is_empty());
        assert!(store.get("project-a").is_none());
    }
}
