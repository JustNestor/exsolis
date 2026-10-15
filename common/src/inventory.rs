//! Server-authoritative player inventory: 9 hotbar slots + 27 main slots.
//!
//! Slots 0..9 are the hotbar (the wire `InventorySync` and the HUD mirror
//! exactly these); 9..36 is main storage. Lua renders this state through
//! the `__inv` snapshot — it never owns items itself anymore, so the
//! E-screen, the HUD and placement can never disagree.

use serde::{Deserialize, Serialize};

use crate::{BlockId, HOTBAR_SIZE};

/// Full player storage: hotbar first, then main rows.
pub const INVENTORY_SIZE: usize = 36;

/// Placed as a hard ceiling here (not just a UI convention) because
/// `add_item` on the server refuses to grow a stack past it — that's what
/// makes the server authoritative over "how many can I hold."
pub const MAX_STACK: u8 = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Slot {
    pub item: BlockId,
    pub amount: u8,
}

/// Full inventory plus the selected hotbar index and the cursor-held stack.
/// `Default` starts empty, `selected` at slot 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Inventory {
    slots: [Option<Slot>; INVENTORY_SIZE],
    selected: u8,
    held: Option<Slot>,
}

impl Inventory {
    pub fn new() -> Self {
        Self {
            slots: [None; INVENTORY_SIZE],
            selected: 0,
            held: None,
        }
    }

    pub fn slots(&self) -> &[Option<Slot>; INVENTORY_SIZE] {
        &self.slots
    }

    /// Hotbar slice (slots 0..9): what the wire sync and HUD mirror.
    pub fn hotbar(&self) -> &[Option<Slot>] {
        &self.slots[..HOTBAR_SIZE]
    }

    pub fn selected_index(&self) -> u8 {
        self.selected
    }

    pub fn selected_slot(&self) -> Option<Slot> {
        self.slots[self.selected as usize]
    }

    pub fn held(&self) -> Option<Slot> {
        self.held
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
    /// it went in).
    pub fn add_item(&mut self, item: BlockId, amount: u8) -> u8 {
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

    /// Every occupied slot, in slot order.
    pub fn iter(&self) -> impl Iterator<Item = (usize, Slot)> + '_ {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.map(|s| (i, s)))
    }

    /// Move everything in `other` into this inventory, returning what did
    /// not fit. Used when a chest is broken: the breaker keeps what fits
    /// and the remainder is reported rather than silently destroyed.
    pub fn absorb(&mut self, other: &mut Inventory) -> u8 {
        let mut remaining = 0u8;
        for (_, slot) in other.iter() {
            remaining = remaining.saturating_add(self.add_item(slot.item, slot.amount));
        }
        *other = Inventory::new();
        remaining
    }

    /// Removes exactly one item from slot `index` if it holds `item`.
    /// Used by place: the server decrements the selected slot only after
    /// confirming the placement is otherwise valid, so a rejected place
    /// never costs an item. Returns `true` on success.
    pub fn remove_one(&mut self, index: u8, item: BlockId) -> bool {
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

    /// Minecraft click semantics against the cursor-held stack. `right` is
    /// the secondary button. Out-of-range slots are ignored. Returns true
    /// when anything changed (caller re-syncs + re-renders on true).
    pub fn click(&mut self, index: usize, right: bool) -> bool {
        if index >= INVENTORY_SIZE {
            return false;
        }
        match (self.slots[index].is_some(), self.held.is_some()) {
            // Empty slot, empty cursor: nothing.
            (false, false) => false,
            // Pick up: whole stack to cursor (right = half, rounded up).
            (true, false) => {
                let mut stack = self.slots[index].take().unwrap();
                if right && stack.amount > 1 {
                    let half = (stack.amount + 1) / 2;
                    stack.amount -= half;
                    self.slots[index] = Some(stack);
                    self.held = Some(Slot {
                        item: stack.item,
                        amount: half,
                    });
                } else {
                    self.held = Some(stack);
                }
                true
            }
            // Drop held onto empty slot.
            (false, true) => {
                let mut held = self.held.unwrap();
                if right {
                    self.slots[index] = Some(Slot {
                        item: held.item,
                        amount: 1,
                    });
                    held.amount -= 1;
                    self.held = (held.amount > 0).then_some(held);
                } else {
                    self.slots[index] = Some(held);
                    self.held = None;
                }
                true
            }
            // Both occupied.
            (true, true) => {
                let mut stack = self.slots[index].unwrap();
                let mut held = self.held.unwrap();
                if stack.item == held.item {
                    // Stack together (right = single item).
                    let space = MAX_STACK - stack.amount;
                    if space == 0 {
                        return false;
                    }
                    let move_n = if right { 1 } else { held.amount }.min(space);
                    stack.amount += move_n;
                    held.amount -= move_n;
                    self.slots[index] = Some(stack);
                    self.held = (held.amount > 0).then_some(held);
                    true
                } else if !right {
                    // Swap different stacks (left button only).
                    self.slots[index] = Some(held);
                    self.held = Some(stack);
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Return the held stack to the inventory (used on close), dropping
    /// overflow with a count.
    pub fn stash_held(&mut self) -> u8 {
        let Some(held) = self.held.take() else {
            return 0;
        };
        self.add_item(held.item, held.amount)
    }
}

impl Default for Inventory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn click_pickup_and_drop() {
        let mut inv = Inventory::new();
        inv.add_item(7, 32);
        assert!(inv.click(0, false));
        assert_eq!(inv.held(), Some(Slot { item: 7, amount: 32 }));
        assert!(inv.slots()[0].is_none());
        assert!(inv.click(5, false));
        assert_eq!(inv.held(), None);
        assert_eq!(inv.slots()[5], Some(Slot { item: 7, amount: 32 }));
    }

    #[test]
    fn click_right_splits_and_places_single() {
        let mut inv = Inventory::new();
        inv.add_item(7, 10);
        assert!(inv.click(0, true));
        assert_eq!(inv.held(), Some(Slot { item: 7, amount: 5 }));
        assert_eq!(inv.slots()[0], Some(Slot { item: 7, amount: 5 }));
        assert!(inv.click(1, true));
        assert_eq!(inv.slots()[1], Some(Slot { item: 7, amount: 1 }));
        assert_eq!(inv.held(), Some(Slot { item: 7, amount: 4 }));
    }

    #[test]
    fn click_swap_and_reject_overflow() {
        let mut inv = Inventory::new();
        inv.add_item(7, 64);
        inv.add_item(3, 10);
        // Pick up the dirt (slot 1), swap with stone (slot 0).
        assert!(inv.click(1, false));
        assert!(inv.click(0, false));
        assert_eq!(inv.slots()[0], Some(Slot { item: 3, amount: 10 }));
        assert_eq!(inv.held(), Some(Slot { item: 7, amount: 64 }));
        // Right-click different stack: no-op.
        assert!(!inv.click(0, true));
    }

    #[test]
    fn click_out_of_range_is_noop() {
        let mut inv = Inventory::new();
        assert!(!inv.click(36, false));
        assert!(!inv.click(999, true));
    }
}
