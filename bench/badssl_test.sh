#!/usr/bin/env bash
# Test proxy's upstream TLS behavior against badssl.com
# Uses a single catch-all route per test, hitting each endpoint directly.

PROXY_BIN="./target/release/tiny-reverse-proxy-rust"
PASS="PASS"
FAIL="FAIL"

printf "\n%-45s %-8s %s\n" "TEST" "RESULT" "DETAILS"
printf "%-45s %-8s %s\n" "----" "------" "-------"

test_badssl() {
    local name="$1"
    local upstream="$2"
    local expect="$3"
    local desc="$4"

    # Write a minimal config pointing to this upstream
    local cfg=$(mktemp /tmp/badssl_XXXXX.toml)
    cat > "$cfg" <<EOF
port = 18443
tls = true
[paths]
"/*" = ["$upstream"]
EOF

    # Start proxy
    $PROXY_BIN "$cfg" 2>/dev/null &
    local pid=$!
    sleep 0.8

    # Make request
    result=$(curl -ks --max-time 15 -o /dev/null -w "%{http_code}" "https://localhost:18443/" 2>&1)

    # Kill proxy
    kill $pid 2>/dev/null
    wait $pid 2>/dev/null
    rm -f "$cfg"

    if [ "$expect" = "ok" ]; then
        if [ "$result" = "200" ]; then
            printf "%-45s %-8s %s\n" "$name" "$PASS" "$desc (HTTP $result)"
        else
            printf "%-45s %-8s %s\n" "$name" "$FAIL" "$desc (HTTP $result)"
        fi
    elif [ "$expect" = "fail" ]; then
        if [ "$result" != "200" ] && [ "$result" != "000" ]; then
            printf "%-45s %-8s %s\n" "$name" "$PASS" "Rejected (HTTP $result)"
        elif [ "$result" = "000" ]; then
            printf "%-45s %-8s %s\n" "$name" "$PASS" "Connection refused (expected)"
        else
            printf "%-45s %-8s %s\n" "$name" "$FAIL" "Should have rejected (HTTP $result)"
        fi
    else
        printf "%-45s %-8s %s\n" "$name" "INFO" "HTTP $result"
    fi
}

echo ""
echo "========== VALID CERTIFICATES =========="
test_badssl "SHA-256"            "sha256.badssl.com:443"                  "ok"   "Standard modern cert"
test_badssl "ECC-256"            "ecc256.badssl.com:443"                  "ok"   "Elliptic curve P-256"
test_badssl "ECC-384"            "ecc384.badssl.com:443"                  "ok"   "Elliptic curve P-384"
test_badssl "RSA-2048"           "rsa2048.badssl.com:443"                 "ok"   "RSA 2048-bit"
test_badssl "RSA-4096"           "rsa4096.badssl.com:443"                 "ok"   "RSA 4096-bit"
test_badssl "RSA-8192"           "rsa8192.badssl.com:443"                 "ok"   "RSA 8192-bit (slow)"
test_badssl "HSTS"               "hsts.badssl.com:443"                    "ok"   "Strict Transport Security"
test_badssl "1000 SANs"          "1000-sans.badssl.com:443"               "ok"   "1000 Subject Alt Names"
test_badssl "Long Subdomain"     "long-extended-subdomain-name-containing-many-letters-and-dashes.badssl.com:443" "ok" "Long domain"

echo ""
echo "========== INVALID CERTIFICATES (should reject) =========="
test_badssl "Expired"            "expired.badssl.com:443"                 "fail" "Cert expired"
test_badssl "Wrong Host"         "wrong.host.badssl.com:443"              "fail" "CN mismatch"
test_badssl "Self-Signed"        "self-signed.badssl.com:443"             "fail" "Untrusted issuer"
test_badssl "Untrusted Root"     "untrusted-root.badssl.com:443"          "fail" "Unknown CA"
test_badssl "Incomplete Chain"   "incomplete-chain.badssl.com:443"        "fail" "Missing intermediate"

echo ""
echo "========== WEAK CRYPTO (should reject) =========="
test_badssl "DH-480"             "dh480.badssl.com:443"                   "fail" "480-bit DH"
test_badssl "DH-512"             "dh512.badssl.com:443"                   "fail" "512-bit DH"
test_badssl "DH-1024"            "dh1024.badssl.com:443"                  "fail" "1024-bit DH"
test_badssl "DH-2048"            "dh2048.badssl.com:443"                  "info" "2048-bit DH"
test_badssl "NULL cipher"        "null.badssl.com:443"                    "fail" "No encryption"
test_badssl "RC4"                "rc4.badssl.com:443"                     "fail" "RC4 (broken)"
test_badssl "RC4-MD5"            "rc4-md5.badssl.com:443"                 "fail" "RC4+MD5 (broken)"
test_badssl "3DES"               "3des.badssl.com:443"                    "fail" "Triple-DES (weak)"

echo ""
echo "========== PROTOCOL VERSIONS =========="
test_badssl "TLS 1.0"            "https://tls-v1-0.badssl.com:1010"      "fail" "Deprecated"
test_badssl "TLS 1.1"            "https://tls-v1-1.badssl.com:1011"      "fail" "Deprecated"
test_badssl "TLS 1.2"            "https://tls-v1-2.badssl.com:1012"      "ok"   "Current standard"

echo ""
echo "========== MISCELLANEOUS =========="
test_badssl "Pinning Test"       "pinning-test.badssl.com:443"            "ok"   "HPKP test domain"
test_badssl "Revoked"            "revoked.badssl.com:443"                 "info" "OCSP revoked cert"
test_badssl "HTTP (plaintext)"   "http.badssl.com:80"                     "info" "Plain HTTP upstream"
echo ""
