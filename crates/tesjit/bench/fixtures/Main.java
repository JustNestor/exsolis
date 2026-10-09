public class Main {
    public static long fibonacci(long n) {
        if (n <= 0) return 0;
        if (n == 1) return 1;
        long a = 0, b = 1;
        for (long i = 2; i <= n; i++) {
            long temp = b;
            b = a + b;
            a = temp;
        }
        return b;
    }

    public static void main(String[] args) {
        long n = (args.length > 0) ? Long.parseLong(args[0]) : 40;
        long iterations = 1000000;

        // Важливо для Java: прогрів (Warmup), щоб C2 JIT встиг скомпілювати метод у нативний код
        for (int i = 0; i < 20000; i++) {
            fibonacci(38 + (i % 3));
        }

        // 1. Одиничний виклик
        long t0 = System.nanoTime();
        long res = fibonacci(n);
        long t1 = System.nanoTime();
        System.out.println("Fibonacci(" + n + ") = " + res);
        System.out.println("Single run: " + (t1 - t0) + " ns\n");

        // 2. Бенчмарк із захистом від LICM та DCE (зміна аргументу на кожній ітерації)
        long accumulator = 0;
        long benchStart = System.nanoTime();

        for (long i = 0; i < iterations; i++) {
            long dynamicN = 38 + (i % 3);
            accumulator += fibonacci(dynamicN);
        }

        long benchEnd = System.nanoTime();

        // Запобіжник від DCE
        if (accumulator == -1) {
            System.out.println("dummy " + accumulator);
        }

        double totalMs = (benchEnd - benchStart) / 1_000_000.0;
        double avgNs = (double) (benchEnd - benchStart) / iterations;

        System.out.println("Benchmark (" + iterations + " iterations):");
        System.out.println("  Total time: " + String.format("%.3f", totalMs) + " ms");
        System.out.println("  Average time per call: " + String.format("%.2f", avgNs) + " ns");
    }
}
