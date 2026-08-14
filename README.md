### 🦅 Hawk
Hawk is a friendly, expression-based, immutable functional programming language designed for demonstrating, implementing, and experimenting with algorithms.

### Philosophy
* Simplicity
* No side effects
* No hidden behavior

### Example

```hawk
io := use("io")

fib := |n| {
  if n <= 1 {
    n
  } else {
    fib(n - 1) + fib(n - 2)
  }
}

fib(30) |> io:println(_)
```
