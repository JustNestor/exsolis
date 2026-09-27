//! Single door for the Renet client.
//!
//! Inbound: drain every channel → [`IncomingPacket`] messages.
//! Outbound: gameplay writes [`OutgoingPacket`] → flushed to the socket.
//!
//! Nothing outside this module (and transport setup in `main`) should
//! touch `RenetClient`. `receive_message` is a consume, not a broadcast:
//! when player.rs and streaming.rs both drained Unreliable, one of them
//! silently ate the other's packets (`YourPosition` vs `PlayerMoved`).

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::{debug, warn};
use bevy_renet::RenetClient;
use bevy_renet::renet::DefaultChannel;
use exsolis_common::{ClientMessage, ServerMessage};

/// All inbound packets for this frame, after a single deserialize.
/// Many systems may `MessageReader` this; each sees every packet.
#[derive(Message)]
pub struct IncomingPacket(pub ServerMessage);

/// Gameplay's request to send a packet. Flushed once, at the end of Update.
#[derive(Message)]
pub struct OutgoingPacket {
    pub msg: ClientMessage,
    pub channel: u8,
}

impl OutgoingPacket {
    pub fn reliable(msg: ClientMessage) -> Self {
        Self {
            msg,
            channel: DefaultChannel::ReliableOrdered.into(),
        }
    }

    pub fn unreliable(msg: ClientMessage) -> Self {
        Self {
            msg,
            channel: DefaultChannel::Unreliable.into(),
        }
    }
}

/// Latest connection flag, updated in [`NetSet::Receive`] before gameplay.
/// Prefer this over reaching into `RenetClient`.
#[derive(Resource, Default, Clone, Copy, PartialEq, Eq)]
pub struct NetConnected(bool);

impl NetConnected {
    #[inline]
    pub fn get(self) -> bool {
        self.0
    }
}

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum NetSet {
    Receive,
    Send,
}

/// Gameplay that reads [`IncomingPacket`] or writes [`OutgoingPacket`]
/// belongs here so it runs after the drain and before the flush.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct GameplaySet;

pub struct NetPlugin;

impl Plugin for NetPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<NetConnected>()
            .add_message::<IncomingPacket>()
            .add_message::<OutgoingPacket>()
            .configure_sets(Update, (NetSet::Receive, GameplaySet, NetSet::Send).chain())
            .add_systems(
                Update,
                (sync_connection_state, drain_incoming)
                    .chain()
                    .in_set(NetSet::Receive),
            )
            .add_systems(Update, flush_outgoing.in_set(NetSet::Send));
    }
}

fn sync_connection_state(client: Res<RenetClient>, mut connected: ResMut<NetConnected>) {
    connected.0 = client.is_connected();
}

const INCOMING_CHANNELS: [u8; 3] = [0, 1, 2];

fn drain_incoming(mut client: ResMut<RenetClient>, mut incoming: MessageWriter<IncomingPacket>) {
    if !client.is_connected() {
        return;
    }
    for channel in INCOMING_CHANNELS {
        while let Some(bytes) = client.receive_message(channel) {
            match bincode::deserialize::<ServerMessage>(&bytes) {
                Ok(msg) => {
                    if let ServerMessage::ChunkData { .. } = msg {
                    } else {
                        debug!("incoming packet {:?}", msg);
                    }
                    incoming.write(IncomingPacket(msg));
                }
                Err(_) => {
                    warn!("bad packet from server, dropping");
                }
            }
        }
    }
}

fn flush_outgoing(mut client: ResMut<RenetClient>, mut outgoing: MessageReader<OutgoingPacket>) {
    let connected = client.is_connected();
    for packet in outgoing.read() {
        if !connected {
            // Still consume so a reconnect doesn't flush stale break/move packets.
            continue;
        }
        debug!("outgoing packet {:?}", packet.msg);
        match bincode::serialize(&packet.msg) {
            Ok(bytes) => client.send_message(packet.channel, bytes),
            Err(e) => warn!("failed to serialize client packet: {e}"),
        }
    }
}
