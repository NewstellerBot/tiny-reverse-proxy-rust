use std::sync::Arc;

use rustls_pki_types::pem::{Error as PemError, PemObject};
use tokio_rustls::rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use tokio_rustls::TlsAcceptor;

/// Generate a self-signed certificate for the given hostnames.
///
/// Uses rcgen to create a certificate with the provided hostnames as Subject
/// Alternative Names. Returns DER-encoded certificate(s) and the private key.
pub fn generate_self_signed_cert(
    hostnames: &[&str],
) -> Result<(Vec<CertificateDer<'static>>, PrivateKeyDer<'static>), Box<dyn std::error::Error>> {
    let san_names: Vec<String> = hostnames.iter().map(|h| h.to_string()).collect();
    let mut params = rcgen::CertificateParams::new(san_names)?;
    params
        .distinguished_name
        .push(rcgen::DnType::CommonName, hostnames[0].to_string());
    params
        .distinguished_name
        .push(rcgen::DnType::OrganizationName, "tiny-reverse-proxy");
    // Mark as CA so browsers trust it when added to the system root store.
    params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);

    let signing_key = rcgen::KeyPair::generate()?;
    let cert = params.self_signed(&signing_key)?;

    let cert_der = cert.der().clone();
    let key_der = signing_key.serialize_der();

    let certs = vec![cert_der];
    let key = PrivateKeyDer::from(PrivatePkcs8KeyDer::from(key_der));

    Ok((certs, key))
}

/// Compute the SHA-256 fingerprint of a DER-encoded certificate.
///
/// Returns the fingerprint as a lowercase colon-separated hex string,
/// e.g. "ab:cd:ef:01:23:...".
pub fn cert_fingerprint(cert: &CertificateDer) -> String {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hasher.update(cert.as_ref());
    let hash = hasher.finalize();

    hash.iter()
        .map(|b| format!("{:02x}", b))
        .collect::<Vec<_>>()
        .join(":")
}

/// Load a TLS acceptor from PEM-encoded certificate and private key files.
///
/// Reads the certificate chain and private key from the given file paths,
/// configures ALPN protocols for h2 and http/1.1, and returns a
/// [`TlsAcceptor`] ready for use.
pub fn load_tls_material(
    cert_path: &str,
    key_path: &str,
) -> Result<(Vec<CertificateDer<'static>>, PrivateKeyDer<'static>), Box<dyn std::error::Error>> {
    let certs = CertificateDer::pem_file_iter(cert_path)?.collect::<Result<Vec<_>, _>>()?;
    let key = match PrivateKeyDer::from_pem_file(key_path) {
        Ok(key) => key,
        Err(PemError::NoItemsFound) => return Err("no private key found in key file".into()),
        Err(error) => return Err(error.into()),
    };

    Ok((certs, key))
}

pub fn load_tls_acceptor(
    cert_path: &str,
    key_path: &str,
) -> Result<TlsAcceptor, Box<dyn std::error::Error>> {
    let (certs, key) = load_tls_material(cert_path, key_path)?;
    build_tls_acceptor(certs, key)
}

/// Build a [`TlsAcceptor`] from DER-encoded certificates and a private key.
///
/// Configures ALPN protocol negotiation for `h2` and `http/1.1`.
pub fn build_tls_acceptor(
    certs: Vec<CertificateDer<'static>>,
    key: PrivateKeyDer<'static>,
) -> Result<TlsAcceptor, Box<dyn std::error::Error>> {
    let mut config = tokio_rustls::rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)?;
    config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];

    Ok(TlsAcceptor::from(Arc::new(config)))
}

/// Build a QUIC-compatible [`ServerConfig`] for HTTP/3.
///
/// Configures ALPN for `h3` only. The returned config can be used with
/// [`quinn::ServerConfig::with_crypto`].
pub fn build_quic_server_config(
    certs: Vec<CertificateDer<'static>>,
    key: PrivateKeyDer<'static>,
) -> Result<Arc<tokio_rustls::rustls::ServerConfig>, Box<dyn std::error::Error>> {
    let mut config = tokio_rustls::rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)?;
    config.alpn_protocols = vec![b"h3".to_vec()];
    config.max_early_data_size = u32::MAX;
    Ok(Arc::new(config))
}

/// Encode DER bytes as a PEM string with the given label.
///
/// Produces standard PEM with 64-character base64 lines.
fn der_to_pem(label: &str, der: &[u8]) -> String {
    use std::fmt::Write;

    // Simple base64 encoder (RFC 4648)
    const CHARS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    let mut b64 = String::with_capacity((der.len() + 2) / 3 * 4);
    let mut i = 0;
    while i < der.len() {
        let remaining = der.len() - i;
        let (b0, b1, b2) = match remaining {
            1 => (der[i], 0u8, 0u8),
            2 => (der[i], der[i + 1], 0u8),
            _ => (der[i], der[i + 1], der[i + 2]),
        };
        let triple = ((b0 as u32) << 16) | ((b1 as u32) << 8) | (b2 as u32);

        b64.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        b64.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);

        if remaining > 1 {
            b64.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            b64.push('=');
        }
        if remaining > 2 {
            b64.push(CHARS[(triple & 0x3F) as usize] as char);
        } else {
            b64.push('=');
        }

        i += 3;
    }

    let mut pem = String::new();
    let _ = writeln!(pem, "-----BEGIN {label}-----");
    for chunk in b64.as_bytes().chunks(64) {
        let _ = writeln!(pem, "{}", std::str::from_utf8(chunk).unwrap());
    }
    let _ = writeln!(pem, "-----END {label}-----");
    pem
}

/// Convert a DER-encoded private key to PEM format.
pub fn private_key_to_pem(key: &PrivateKeyDer<'_>) -> String {
    der_to_pem("PRIVATE KEY", key.secret_der())
}

/// Persist PEM-encoded certificate and private key files to a directory.
///
/// Writes `cert.pem` and `key.pem` into the specified directory. The
/// directory must already exist.
pub fn persist_certs(
    dir: &std::path::Path,
    certs: &[CertificateDer<'_>],
    key_pem: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    // Write certificate(s) as PEM
    let cert_path = dir.join("cert.pem");
    let mut cert_file = std::fs::File::create(&cert_path)?;
    for cert in certs {
        let pem = der_to_pem("CERTIFICATE", cert.as_ref());
        cert_file.write_all(pem.as_bytes())?;
    }

    // Write private key PEM
    let key_path = dir.join("key.pem");
    std::fs::write(&key_path, key_pem)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_self_signed_cert_single_host() {
        let (certs, _key) = generate_self_signed_cert(&["localhost"]).unwrap();
        assert_eq!(certs.len(), 1);
        // DER-encoded certificate should be non-empty
        assert!(!certs[0].as_ref().is_empty());
    }

    #[test]
    fn test_generate_self_signed_cert_multiple_hosts() {
        let (certs, _key) =
            generate_self_signed_cert(&["example.com", "www.example.com", "localhost"]).unwrap();
        assert_eq!(certs.len(), 1);
        assert!(!certs[0].as_ref().is_empty());
    }

    #[test]
    fn test_cert_fingerprint_format() {
        let (certs, _key) = generate_self_signed_cert(&["localhost"]).unwrap();
        let fingerprint = cert_fingerprint(&certs[0]);

        // SHA-256 fingerprint should be 32 bytes = 64 hex chars + 31 colons = 95 chars
        assert_eq!(fingerprint.len(), 95);

        // All parts should be valid two-character hex
        for part in fingerprint.split(':') {
            assert_eq!(part.len(), 2);
            assert!(u8::from_str_radix(part, 16).is_ok());
        }
    }

    #[test]
    fn test_cert_fingerprint_deterministic() {
        let (certs, _key) = generate_self_signed_cert(&["localhost"]).unwrap();
        let fp1 = cert_fingerprint(&certs[0]);
        let fp2 = cert_fingerprint(&certs[0]);
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_cert_fingerprint_differs_for_different_certs() {
        let (certs_a, _) = generate_self_signed_cert(&["a.example.com"]).unwrap();
        let (certs_b, _) = generate_self_signed_cert(&["b.example.com"]).unwrap();
        let fp_a = cert_fingerprint(&certs_a[0]);
        let fp_b = cert_fingerprint(&certs_b[0]);
        assert_ne!(fp_a, fp_b);
    }

    #[test]
    fn test_build_tls_acceptor_from_generated() {
        let (certs, key) = generate_self_signed_cert(&["localhost"]).unwrap();
        let acceptor = build_tls_acceptor(certs, key);
        assert!(acceptor.is_ok());
    }

    #[test]
    fn test_load_tls_material_from_persisted_pem() {
        let (certs, key) = generate_self_signed_cert(&["localhost"]).unwrap();
        let key_pem = private_key_to_pem(&key);
        let dir = tempfile::tempdir().unwrap();
        persist_certs(dir.path(), &certs, &key_pem).unwrap();

        let loaded = load_tls_material(
            dir.path().join("cert.pem").to_str().unwrap(),
            dir.path().join("key.pem").to_str().unwrap(),
        )
        .unwrap();

        assert_eq!(loaded.0.len(), 1);
        assert!(!loaded.0[0].as_ref().is_empty());
        assert!(!loaded.1.secret_der().is_empty());
    }

    #[test]
    fn test_persist_certs() {
        let (certs, _key) = generate_self_signed_cert(&["localhost"]).unwrap();
        let key_pem = "-----BEGIN PRIVATE KEY-----\ntest\n-----END PRIVATE KEY-----\n";

        let dir = tempfile::tempdir().unwrap();
        persist_certs(dir.path(), &certs, key_pem).unwrap();

        let cert_path = dir.path().join("cert.pem");
        let key_path = dir.path().join("key.pem");

        assert!(cert_path.exists());
        assert!(key_path.exists());

        let cert_content = std::fs::read_to_string(&cert_path).unwrap();
        assert!(cert_content.contains("-----BEGIN CERTIFICATE-----"));
        assert!(cert_content.contains("-----END CERTIFICATE-----"));

        let key_content = std::fs::read_to_string(&key_path).unwrap();
        assert_eq!(key_content, key_pem);
    }
}
