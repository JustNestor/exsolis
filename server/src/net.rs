//! Single door for the Renet server.
//!
//! Inbound: drain every channel for every connected client → one
//! deserialize each → [`IncomingPacket`] messages.
//! Outbound: gameplay writes [`OutgoingPacket`] → flushed to the socket.
//!
//! Nothing outside this module should touch `RenetServer` directly.
//! Mirrors the client's `net.rs`: `receive_message` is a consume, not a
//! broadcast, so if two systems both drained the same channel one would
//! silently steal the other's packets. Centralizing the drain here means
//! that bug can't happen as more systems start caring about client
//! packets — they all just read `IncomingPacket` instead.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::warn;
use bevy_renet::RenetServer;
use bevy_renet::renet::DefaultChannel;
use exsolis_common::{ClientMessage, PlayerId, ServerMessage};

/// One deserialized client packet for this frame, tagged with who sent it.
/// Many systems may `MessageReader` this; each sees every packet.
#[derive(Message, Debug, Clone)]
pub struct IncomingPacket {
    pub player_id: PlayerId,
    pub msg: ClientMessage,
}

/// Gameplay's request to send a packet. Flushed once, at the end of Update.
///
/// Usually built via `OutgoingPacket::to`/`broadcast`/etc, which carry the
/// `ServerMessage` and let `flush_outgoing` serialize it. A caller that
/// already needed the serialized length up front for its own bookkeeping
/// (e.g. `chunks::stream_chunks_to_players` checking channel capacity
/// before committing to a send) can serialize once and pass the bytes via
/// `OutgoingPacket::raw` instead of serializing twice.
#[derive(Message, Debug, Clone)]
pub struct OutgoingPacket {
    pub target: SendTarget,
    pub payload: Payload,
    pub channel: u8,
}

#[derive(Debug, Clone)]
pub enum Payload {
    Msg(ServerMessage),
    Bytes(Vec<u8>),
}

/// Who an `OutgoingPacket` goes to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SendTarget {
    One(PlayerId),
    All,
    AllExcept(PlayerId),
}

impl OutgoingPacket {
    pub fn to(player_id: PlayerId, msg: ServerMessage) -> Self {
        Self {
            target: SendTarget::One(player_id),
            payload: Payload::Msg(msg),
            channel: DefaultChannel::ReliableOrdered.into(),
        }
    }

    pub fn to_unreliable(player_id: PlayerId, msg: ServerMessage) -> Self {
        Self {
            target: SendTarget::One(player_id),
            payload: Payload::Msg(msg),
            channel: DefaultChannel::Unreliable.into(),
        }
    }

    pub fn broadcast(msg: ServerMessage) -> Self {
        Self {
            target: SendTarget::All,
            payload: Payload::Msg(msg),
            channel: DefaultChannel::ReliableOrdered.into(),
        }
    }

    pub fn broadcast_unreliable(msg: ServerMessage) -> Self {
        Self {
            target: SendTarget::All,
            payload: Payload::Msg(msg),
            channel: DefaultChannel::Unreliable.into(),
        }
    }

    pub fn broadcast_except(player_id: PlayerId, msg: ServerMessage) -> Self {
        Self {
            target: SendTarget::AllExcept(player_id),
            payload: Payload::Msg(msg),
            channel: DefaultChannel::ReliableOrdered.into(),
        }
    }

    pub fn broadcast_except_unreliable(player_id: PlayerId, msg: ServerMessage) -> Self {
        Self {
            target: SendTarget::AllExcept(player_id),
            payload: Payload::Msg(msg),
            channel: DefaultChannel::Unreliable.into(),
        }
    }

    /// Pre-serialized bytes, for a caller that already needed to
    /// serialize the message for its own bookkeeping (e.g. checking the
    /// encoded length against remaining channel capacity) and shouldn't
    /// pay for a second serialize at flush time.
    pub fn raw(target: SendTarget, channel: impl Into<u8>, bytes: Vec<u8>) -> Self {
        Self {
            target,
            payload: Payload::Bytes(bytes),
            channel: channel.into(),
        }
    }

    /// Override the default channel picked by the constructor above.
    pub fn on_channel(mut self, channel: impl Into<u8>) -> Self {
        self.channel = channel.into();
        self
    }
}

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum NetSet {
    Receive,
    Send,
}

/// Gameplay that reads `IncomingPacket` or writes `OutgoingPacket` belongs
/// here so it runs after the drain and before the flush.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct GameplaySet;

pub struct NetPlugin;

impl Plugin for NetPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<IncomingPacket>()
            .add_message::<OutgoingPacket>()
            .configure_sets(Update, (NetSet::Receive, GameplaySet, NetSet::Send).chain())
            .add_systems(Update, drain_incoming.in_set(NetSet::Receive))
            .add_systems(Update, flush_outgoing.in_set(NetSet::Send));
    }
}

fn drain_incoming(mut server: ResMut<RenetServer>, mut incoming: MessageWriter<IncomingPacket>) {
    let channels: [u8; 2] = [
        DefaultChannel::ReliableOrdered.into(),
        DefaultChannel::Unreliable.into(),
    ];

    for client_id in server.clients_id() {
        for channel in channels {
            while let Some(bytes) = server.receive_message(client_id, channel) {
                match bincode::deserialize::<ClientMessage>(&bytes) {
                    Ok(msg) => {
                        incoming.write(IncomingPacket {
                            player_id: client_id,
                            msg,
                        });
                    }
                    Err(_) => {
                        warn!("bad packet from {client_id}, dropping");
                    }
                }
            }
        }
    }
}

// `pub(crate)` (was private) so other plugins can anchor a system
// `.before(flush_outgoing)` explicitly when they write `OutgoingPacket`
// from a system that isn't itself inside `NetSet::Send` — see
// `scripting.rs::ScriptingPlugin` for why `drain_script_requests` needs
// this.
pub(crate) fn flush_outgoing(
    mut server: ResMut<RenetServer>,
    mut outgoing: MessageReader<OutgoingPacket>,
) {
    for packet in outgoing.read() {
        let bytes = match &packet.payload {
            Payload::Bytes(bytes) => bytes.clone(),
            Payload::Msg(msg) => match bincode::serialize(msg) {
                Ok(bytes) => bytes,
                Err(e) => {
                    warn!("failed to serialize server packet: {e}");
                    continue;
                }
            },
        };

        match packet.target {
            SendTarget::One(player_id) => {
                server.send_message(player_id, packet.channel, bytes);
            }
            SendTarget::All => {
                server.broadcast_message(packet.channel, bytes);
            }
            SendTarget::AllExcept(player_id) => {
                server.broadcast_message_except(player_id, packet.channel, bytes);
            }
        }
    }
}
