### ⛽ Oil
Oil is a friendly, expression-based, immutable functional programming language for building scalable software.

### 🫧 Philosophy
* Simplicity
* Immutability
* No hidden behavior

### Example

```oil
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
