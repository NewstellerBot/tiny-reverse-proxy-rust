#[cfg(test)]
mod tests {
    use tiny_reverse_proxy_rust::{
        router::{PathResolution, Router},
        utils::create_temp_config,
    };

    #[test]
    fn test_router_with_wildcards() {
        let config = r#"
        [paths]
        "/api/*" = ["server1.com:80", "server2.com:80"]
        "/static/*" = ["static1.com:80", "static2.com:80"]
        "/images/*" = ["image1.com:80", "image2.com:80"]
        "#;

        let f = create_temp_config(config);
        let router = Router::load_from_file(f.path().to_str().unwrap()).unwrap();

        assert_eq!(
            router.get_servers("/api/test"),
            Some(&vec![
                "server1.com:80".to_string(),
                "server2.com:80".to_string()
            ])
        );
        assert_eq!(
            router.get_servers("/static/css/style.css"),
            Some(&vec![
                "static1.com:80".to_string(),
                "static2.com:80".to_string()
            ])
        );
        assert_eq!(router.get_servers("/unknown/path"), None);
    }

    #[test]
    fn test_router_exact_match() {
        let config = r#"
        [paths]
        "/api" = ["server1.com:80"]
        "/static/*" = ["static1.com:80", "static2.com:80"]
        "/images/*" = ["image1.com:80", "image2.com:80"]
        "#;

        let f = create_temp_config(config);
        let router = Router::load_from_file(f.path().to_str().unwrap()).unwrap();

        assert_eq!(
            router.get_servers("/api"),
            Some(&vec!["server1.com:80".to_string()])
        );
        assert_eq!(
            router.get_servers("/static/js/script.js"),
            Some(&vec![
                "static1.com:80".to_string(),
                "static2.com:80".to_string()
            ])
        );
        assert_eq!(
            router.get_servers("/images/photo.jpg"),
            Some(&vec![
                "image1.com:80".to_string(),
                "image2.com:80".to_string()
            ])
        );
    }

    #[test]
    fn test_router_no_match() {
        let config = r#"
        [paths]
        "/api/*" = ["server1.com:80", "server2.com:80"]
        "/static/*" = ["static1.com:80", "static2.com:80"]
        "#;

        let f = create_temp_config(config);
        let router = Router::load_from_file(f.path().to_str().unwrap()).unwrap();

        assert_eq!(router.get_servers("/unknown/path"), None);
        assert_eq!(router.get_servers("/random"), None);
    }

    #[test]
    fn test_router_partial_match() {
        let config = r#"
        [paths]
        "/api/*" = ["server1.com:80", "server2.com:80"]
        "/api/v1/*" = ["server3.com:80", "server4.com:80"]
        "#;

        let f = create_temp_config(config);
        let router = Router::load_from_file(f.path().to_str().unwrap()).unwrap();

        assert_eq!(
            router.get_servers("/api/v1/resource"),
            Some(&vec![
                "server3.com:80".to_string(),
                "server4.com:80".to_string()
            ])
        );
        assert_eq!(
            router.get_servers("/api/resource"),
            Some(&vec![
                "server1.com:80".to_string(),
                "server2.com:80".to_string()
            ])
        );
    }
}
