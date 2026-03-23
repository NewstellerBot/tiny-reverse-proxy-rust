use async_trait::async_trait;
use pingora::prelude::*;

pub struct MyProxy;

#[async_trait]
impl ProxyHttp for MyProxy {
    type CTX = ();

    fn new_ctx(&self) -> Self::CTX {}

    async fn upstream_peer(
        &self,
        _session: &mut Session,
        _ctx: &mut Self::CTX,
    ) -> Result<Box<HttpPeer>> {
        let peer = Box::new(HttpPeer::new(
            "127.0.0.1:9000",
            false,
            String::new(),
        ));
        Ok(peer)
    }
}

fn main() {
    let mut server = Server::new(None).unwrap();
    server.bootstrap();

    let mut proxy = http_proxy_service(&server.configuration, MyProxy);
    proxy.add_tcp("0.0.0.0:8300");

    server.add_service(proxy);
    server.run_forever();
}
