-- fib_timed.lua
local function fibonacci(n)
    if n <= 0 then return 0 end
    if n == 1 then return 1 end

    local a, b = 0, 1
    for _ = 2, n do
        a, b = b, a + b
    end
    return b
end

local n = 40
local iterations = 1000000

-- 1. Перший (холодний) одиничний виклик (в наносекундах)
local t0 = os.clock()
local result = fibonacci(n)
local t1 = os.clock()
local single_1_ns = (t1 - t0) * 1e9

print(string.format("Fibonacci(%d) = %d", n, result))
print(string.format("Single run #1 (cold): %.0f ns\n", single_1_ns))

-- 2. Мікро-бенчмарк (загальний час у мілісекундах)
local bench_start = os.clock()
for _ = 1, iterations do
    fibonacci(n)
end
local bench_end = os.clock()

local total_sec = bench_end - bench_start
local total_ms = total_sec * 1000
local avg_ns_per_op = (total_sec / iterations) * 1e9

-- print(string.format("Benchmark (%d iterations):", iterations))
-- print(string.format("  Total time: %.3f ms", total_ms))
print(string.format("  Average time per call: %.2f ns\n", avg_ns_per_op))

-- 3. Другий (підігрітий) одиничний виклик (в наносекундах)
local t2 = os.clock()
local _ = fibonacci(n)
local t3 = os.clock()
local single_2_ns = (t3 - t2) * 1e9

-- print(string.format("Single run #2 (warm): %.0f ns", single_2_ns))
