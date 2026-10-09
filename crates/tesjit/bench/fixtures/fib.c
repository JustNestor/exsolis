#include <stdio.h>
#include <time.h>
#include <stdlib.h>

// Забороняємо компілятору інлайнити функцію
__attribute__((noinline))
long long fibonacci(long long n) {
    if (n <= 0) return 0;
    if (n == 1) return 1;
    long long a = 0, b = 1;
    for (long long i = 2; i <= n; i++) {
        long long temp = b;
        b = a + b;
        a = temp;
    }
    return b;
}

static inline long long get_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

int main(int argc, char** argv) {
    long long n = (argc > 1) ? atoll(argv[1]) : 40;
    long long iterations = 1000000;

    // 1. Одиничний виклик
    long long t0 = get_ns();
    long long res = fibonacci(n);
    long long t1 = get_ns();
    printf("Fibonacci(%lld) = %lld\n", n, res);
    printf("Single run: %lld ns\n\n", t1 - t0);

    // 2. Бенчмарк із захистом від LICM та DCE
    long long accumulator = 0;
    long long bench_start = get_ns();
    
    for (long long i = 0; i < iterations; i++) {
        // Динамічна зміна аргументу заважає LLVM винести або згорнути цикл
        long long dynamic_n = 38 + (i % 3);
        accumulator += fibonacci(dynamic_n);
    }
    
    long long bench_end = get_ns();

    // Запобіжник, щоб компілятор не викинув увесь цикл через DCE
    if (accumulator == -1) {
        printf("dummy %lld\n", accumulator);
    }

    double total_ms = (bench_end - bench_start) / 1e6;
    double avg_ns = (double)(bench_end - bench_start) / iterations;

    printf("Benchmark (%lld iterations):\n", iterations);
    printf("  Total time: %.3f ms\n", total_ms);
    printf("  Average time per call: %.2f ns\n", avg_ns);

    return 0;
}
