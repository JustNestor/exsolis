-- Generic zero-based slot container shared by player, block entity, and
-- entity inventories. Slots are intentionally plain Lua data so mods can
-- attach their own rules without depending on Rust or a particular owner.
Container = Container or {}

function Container.new(size)
    assert(type(size) == "number" and size >= 0 and size % 1 == 0,
        "container size must be a non-negative integer")
    local container = { size = size, slots = {} }
    for index = 0, size - 1 do container.slots[index] = nil end
    return container
end

function Container.get(container, index)
    if index < 0 or index >= container.size then return nil end
    return container.slots[index]
end

function Container.set(container, index, item)
    if index < 0 or index >= container.size then return false end
    container.slots[index] = item
    return true
end

function Container.swap(container, first, second)
    if first < 0 or first >= container.size or second < 0 or second >= container.size then
        return false
    end
    container.slots[first], container.slots[second] = container.slots[second], container.slots[first]
    return true
end

function Container.each(container)
    local index = -1
    return function()
        index = index + 1
        if index < container.size then return index, container.slots[index] end
    end
end
