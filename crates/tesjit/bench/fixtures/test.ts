function fib(n: i32): i32 {
  if (n <= 0) { return 0; }
  let a: i32 = 0;
  let b: i32 = 1;
  for (let i: i32 = 2; i <= n; i++) {
    let temp: i32 = a + b;
    a = b;
    b = temp;
  }
  return b;
}

function fib2(n: i32, iters: i32): i32 {
  let res: i32 = 0;
  for (let i: i32 = 0; i < iters; i++) {
    res = fib(n);
  }
  return res;
}
