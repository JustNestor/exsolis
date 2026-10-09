//! Minimal shared inventory: hotbar only.
//!
//! Server-side hotbar state for validating block placement. Player-facing
//! container and inventory UI state lives in Lua (`mods/container.lua` and
//! `mods/player_inventory.lua`); this small type currently backs the block
//! interaction rules until those rules are moved to the Lua inventory.

use serde::{Deserialize, Serialize};

use crate::{BlockType, HOTBAR_SIZE};

/// Placed as a hard ceiling here (not just a UI convention) because
/// `add_item` on the server refuses to grow a stack past it — that's what
/// makes the server authoritative over "how many can I hold."
pub const MAX_STACK: u8 = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Slot {
    pub item: BlockType,
    pub amount: u8,
}

/// Hotbar-only inventory. `Default` starts empty, `selected` at slot 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Inventory {
    slots: [Option<Slot>; HOTBAR_SIZE],
    selected: u8,
}

impl Inventory {
    pub fn new() -> Self {
        Self {
            slots: [None; HOTBAR_SIZE],
            selected: 0,
        }
    }

    pub fn slots(&self) -> &[Option<Slot>; HOTBAR_SIZE] {
        &self.slots
    }

    pub fn selected_index(&self) -> u8 {
        self.selected
    }

    pub fn selected_slot(&self) -> Option<Slot> {
        self.slots[self.selected as usize]
    }

    /// No-op (not an error) if `index >= HOTBAR_SIZE`: an out-of-range
    /// `ClientMessage::SelectHotbar` from a modified/buggy client just
    /// gets ignored rather than panicking the server.
    pub fn select(&mut self, index: u8) {
        if (index as usize) < HOTBAR_SIZE {
            self.selected = index;
        }
    }

    /// Adds up to `amount` of `item`, filling existing same-type stacks
    /// before using empty slots. Returns how much didn't fit (0 = all of
    /// it went in). Mirrors the old project's `add_item`, minus the
    /// separate main-storage range (there's only the hotbar here).
    pub fn add_item(&mut self, item: BlockType, amount: u8) -> u8 {
        let mut remaining = amount;

        for slot in self.slots.iter_mut().flatten() {
            if remaining == 0 {
                break;
            }
            if slot.item == item && slot.amount < MAX_STACK {
                let space = MAX_STACK - slot.amount;
                let add = space.min(remaining);
                slot.amount += add;
                remaining -= add;
            }
        }
        for slot in self.slots.iter_mut() {
            if remaining == 0 {
                break;
            }
            if slot.is_none() {
                let add = remaining.min(MAX_STACK);
                *slot = Some(Slot { item, amount: add });
                remaining -= add;
            }
        }

        remaining
    }

    /// Removes exactly one item from slot `index` if it holds `item`.
    /// Used by place: the server decrements the selected slot only after
    /// confirming the placement is otherwise valid, so a rejected place
    /// never costs an item. Returns `true` on success.
    pub fn remove_one(&mut self, index: u8, item: BlockType) -> bool {
        let Some(slot) = self.slots.get_mut(index as usize) else {
            return false;
        };
        let Some(s) = slot else { return false };
        if s.item != item {
            return false;
        }
        s.amount -= 1;
        if s.amount == 0 {
            *slot = None;
        }
        true
    }
}

impl Default for Inventory {
    fn default() -> Self {
        Self::new()
    }
}
