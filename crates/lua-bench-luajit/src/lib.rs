//! Identical workload for every Lua backend crate: a recursive `fibonacci`
//! plus an in-place `ecs_step(dt)` over two module-level tables.
//! The Lua sources are byte-identical across backends (valid Luau too).

use std::cell::RefCell;

thread_local! {
    static LUA: RefCell<mlua::Lua> = RefCell::new({
        let lua = mlua::Lua::new();
        lua.load(SCRIPT).exec().expect("bench script loads");
        lua
    });
}

fn with_lua<R>(f: impl FnOnce(&mlua::Lua) -> R) -> R {
    LUA.with(|lua| f(&lua.borrow()))
}

const SCRIPT: &str = r#"
function fibonacci(n)
    if n < 2 then return n end
    return fibonacci(n - 1) + fibonacci(n - 2)
end

local xs, vs = {}, {}
function ecs_init(n)
    for i = 1, n do xs[i] = i * 0.1; vs[i] = 1.0 end
end
function ecs_step(dt)
    for i = 1, #xs do xs[i] = xs[i] + vs[i] * dt end
end
function ecs_sum()
    local s = 0.0
    for i = 1, #xs do s = s + xs[i] end
    return s
end
"#;

/// Call `fibonacci(n)` (LuaJIT backend).
pub fn fib(n: i64) -> i64 {
    with_lua(|lua| {
        lua.globals()
            .get::<mlua::Function>("fibonacci")
            .expect("fibonacci defined")
            .call(n)
            .expect("fibonacci runs")
    })
}

/// Stateful in-Lua row loop, created once per bench.
pub struct Ecs {
    n: usize,
}

impl Ecs {
    pub fn new(n: usize) -> Self {
        with_lua(|lua| {
            lua.globals()
                .get::<mlua::Function>("ecs_init")
                .expect("ecs_init defined")
                .call::<()>(n as i64)
                .expect("ecs_init runs");
        });
        Self { n }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn step(&self, dt: f64) {
        with_lua(|lua| {
            lua.globals()
                .get::<mlua::Function>("ecs_step")
                .expect("ecs_step defined")
                .call::<()>(dt)
                .expect("ecs_step runs");
        });
    }

    pub fn sum(&self) -> f64 {
        with_lua(|lua| {
            lua.globals()
                .get::<mlua::Function>("ecs_sum")
                .expect("ecs_sum defined")
                .call(())
                .expect("ecs_sum runs")
        })
    }
}
