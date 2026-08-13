use super::*;

// fn receive_movement(mut server: ResMut<RenetServer>) {
//     for client_id in server.clients_id() {
//         while let Some(bytes) = server.receive_message(client_id, DefaultChannel::Unreliable) {
//             let Ok(msg) = bincode::deserialize::<ClientMessage>(&bytes) else {
//                 warn!("bad packet from {client_id}, dropping");
//                 continue;
//             };
//             match msg {
//                 ClientMessage::Move { x, y, z } => {}
//             }
//         }
//     }
// }

pub fn new_transport() -> NetcodeServerTransport {
    let bind_addr: SocketAddr = std::env::var("EXSOLIS_BIND_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:5000".to_string())
        .parse()
        .expect("bad EXSOLIS_BIND_ADDR");
    let public_addr: SocketAddr = std::env::var("EXSOLIS_PUBLIC_ADDR")
        .unwrap_or_else(|_| SERVER_ADDR.to_string())
        .parse()
        .expect("bad EXSOLIS_PUBLIC_ADDR");

    let socket = UdpSocket::bind(bind_addr).expect("failed to bind server socket");
    let current_time = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap();

    let server_config = ServerConfig {
        current_time,
        max_clients: 32,
        protocol_id: PROTOCOL_ID,
        public_addresses: vec![public_addr],
        authentication: ServerAuthentication::Unsecure,
    };

    info!("listening on {bind_addr}, advertising {public_addr} to clients");
    NetcodeServerTransport::new(server_config, socket).expect("failed to start transport")
}

pub fn log_netcode_error(error: On<NetcodeErrorEvent>) {
    warn!("netcode error: {}", *error);
}
