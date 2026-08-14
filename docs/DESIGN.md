### 🦅 Hawk
Hawk is a friendly, expression-based, immutable functional programming language designed for demonstrating, implementing, and experimenting with algorithms.

### Philosophy
* Simplicity
* No side-effects
* No hidden behavior

### Goals
* Simple and lightweight
* Minimal and readable syntax
* Easy to learn and use

### Non-goals
* High-performance systems programming
* Complex features
* Hidden behaviour
* Mutability with side-effects
64-bit floating point number

### Notes
* Almost everything is an expression
* Block expressions return the last statement evaluation result as their result

### Reserved words
The best way to get a quick feel of a language's style is to look what keywords it uses:
```
if else true false memoize echo
```

### Comments
Here is an example for single-line and multi-line comments:

```
# It's a single-line comment!
#[
    It's a multi-line 
    comment
]#
```

Line comments start with `#` and continue until the end of the line.
Everything after `#` on that line is ignored by the interpreter.

Multiline comments are enclosed in `#[` and `]#` 
and also ignored by the interpreter.

### Identifiers
Identifiers must start with letter or underscore and may contain letters, 
digits, underscores and question marks. Here is some examples:
```
place := 1
flowers := 2
_clever := 4
exists? := true
is_dog := true
```

### Numbers
Numbers must start with digits sequence and may contain decimal part and scientific notation
```
a := 123
b := 0.123
c := 12e-4
```
Numbers are stored as 64-bit floating point number

### Strings
String is a textual data enclosed in quotes that may contain escape sequences, any unicode chars

```
a := "hello, world!"
b := "hello,\n Mike!"
c := "🏆 winner!"
```

Supported escape sequences
* \n
* \r
* \t
* \x{..}
* \o{..}
* \u{....}
* \U{........}
* \"
* \0

### Nil
Nil represents a nothing value. Means here is no specified value:

```
a := nil
io:println(a = nil) # true
```

### Variables
Variables must be declared using `:=` operator
```
a := "hello"
a := "world" # not a mutation, variable shadowing.
```

### Operators
Hawk supports following binary operations:
* =, !=, >, <, >=, <=
* &, |
* +, -, *, /, %

Following unary operations:
* !, -

Following postfix operations:
* : for module field access
* [_] for index or key access

### Arrays
Array represents a sequence of values:

```
a := [1, "hello", true, [1, 2, 3]]
```

To create a copy of a array, or a new array from one or more others, use spread operator to inject
values from some array into specified one

```
a := [1, 2, 3]
b := ["hello", "my", "friend"]
c := [..a, ..b, true, false]
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# [1, 2, 3, "hello", "my", "friend", true, false] 
```

To access array element you can use `[_]` (index) operator:

```
a := [1, 2, 3]
b := a[1] # 2
```

### Dictionaries
Dictionary represents a sequence of key-value pairs:

```
a := {
    "status": "ok",
    "detail": "success!"
}
```

As for arrays, you can use spread operator to inject value from some dict into specified one

```
a := {
    "products": ["floor", "sugar", "eggs"]
}
b := {
    ..a,
    "total": 50000
    "currency": "uzs"
    "discount": 3000
}
```

To access dictionary element you can use `[_]` (index) operator:

```
vegetables := {
    "tomato": 5, 
    "cucumber": 3,
    "onion": 4, 
    "garlic": 1
}
garlic_amount := vegetables["garlic"]
```

### Ranges
To create an array of numbers in some range, you can use range expression:

```
a := 0..5
b := 0..=5

io:println(a) # [0, 1, 2, 3, 4]
io:println(a) # [0, 1, 2, 3, 4, 5]
```

### Echoes
To print some debug information you can use `echo` keyword:

```
echo 1 + 2 # 3
echo 2 * 2 # 4
```

### Functions
Functitons are defined with `|param1, param2, ..n| ...` syntax. Here is some examples:

With block as function body:
```
fib := |n| {
    if n <= 1 {
        n
    } else {
        fib(n - 1) + fib(n - 2)
    }
}
```

With expression as function body:
```
square = |n| n * n
```

All functions which are declared are closures, and have access to their outer scope variables:

```
x := |x| {
    || x * 2
}
y := x()
echo y()
```

### Control Flow: If
If you want to evaluate some code depending on condition, you can use `if`, `else if` and `else` expressions:

```
number := int:parse(io:readln())
sign := if a < 0 {
    -1
} else if a > 0 {
    1
} else {
    0
}
```

```
number := int:parse(io:readln())
is_even := if number % 2 = 0 {
    true
} else {
    false
}
```

The expression always evaluates a value. Nil being returned in the case of 
the alternative branch not being specified and condition not passing.

### Control Flow: Match
Sometimes it's good to replace an `if`'s chain with a pattern matching.
The expression always evaluates a value. Nil being returned in the case of 
a value doesn't match any pattern

Pattern matching supports following patterns:
* Literal patterns
* Array patterns
* Dictionary patterns
* Range patterns
* Compare patterns
* Binding patterns
* Wilcard pattern

Here is some examples:

Literal patterns:
```
fibonacci := |n| match n {
  0 -> 0
  1 -> 1
  n -> fibonacci(n - 1) + fibonacci(n - 2)
};
```

List patterns:
```
describe := |list| match list {
  [] -> "list is empty!"
  [a] -> "list has only one element: " + a
  [a, b] -> "list has two elements: " + a + " and " + b,
  [a, b, ..] -> "list has at least two elements: " + a + " and " + b
}
```

Dict patterns:
```
describe := |dict| match dict {
  {} -> "list is empty!"
  {a: b} -> "dict has only one key-value pair: " + a + ":" + b
  {a: _, b: _} -> "dict has two keys: " + a + " and " + b,
  {a: _, b: _, ..} -> "dict has at least two keys: " + a + " and " + b
}
```

Range patterns:
```
less_then_100 := |n| match n {
    0..99 -> true,
    # ^^^
    _ -> false
}
```

Same code with compare patterns:
```
less_then_100 := |n| match n {
    < 100 -> true,
    # ^^^
    _ -> false
}
```

Binding patterns:
```
flavour := |item| match item {
    "icecream" -> "sweet",
    other -> other
    # ^^^
}
```

Wilcard pattern is `_` you already seen in match arms before

### Control Flow: Loops
Hawk has no loops, so you can use recursion!

```
factorial := |n| match n {
    0 -> 1,
    n -> n * factorial(n - 1)
}
```

### Modularity
Every Hawk file is a module. You can use one module from other one using `use` function and `:` operator:

```
io := use("io")

io:println("Hello, world!")
```

### Pipelines
Hawk allows to rewrite calls chain in a more accurate and readable form using `|>` (pipeline) operator:

Without pipeline:
```
io := use("io")
double := |n| n * 2
square := |n| n * n

io:println(square(double(10)))
```

With pipeline:
```
io := use("io")
square := |n| n * n

double(10) |> square(_)  |> io:println(_)
```

Wildcard (`_`) here means a position for result of a previous pipeline call

### Memoization
Pure functions can be made memoized via `memoize` keyword. Ensure the function is pure before using memoization
to avoid hidden bugs. Here is an example for memoization:

```
io := use("io")

fib := memoize |n| {
    if n <= 1 {
        n
    } else {
        fib(n - 1) + fib(n - 2)
    }
}

fib(35) |> io:println(_)
```

### Standard library
There is a suite of builtin functions and modules which help solve many different class of problem.
