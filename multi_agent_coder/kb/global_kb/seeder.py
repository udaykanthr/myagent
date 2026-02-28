"""
Seeder — populates the global knowledge base with sample data.

Seeds:
- errors.db with 5 errors per language (Python, JavaScript, TypeScript,
  Java, Go, Rust, C#)
- registry/ markdown files with frontmatter for patterns, ADRs, docs,
  and behavioral categories
- Qdrant ``global_kb`` collection with embedded markdown chunks

Designed as a dev-utility that can be re-run to reset sample data.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Optional

from .error_dict import ErrorDict, ErrorFix

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_GLOBAL_DIR = os.path.dirname(os.path.abspath(__file__))
_CORE_DIR = os.path.join(_GLOBAL_DIR, "core")
_REGISTRY_DIR = os.path.join(_GLOBAL_DIR, "registry")


def _errors_db_path() -> str:
    return os.path.join(_CORE_DIR, "errors.db")


# ---------------------------------------------------------------------------
# Error seed data
# ---------------------------------------------------------------------------

_ERROR_SEEDS: list[ErrorFix] = [
    # ── Python ──────────────────────────────────────────────────────────
    ErrorFix(
        error_type="AttributeError",
        language="python",
        pattern=r"AttributeError:\s*'(\w+)'\s+object\s+has\s+no\s+attribute",
        cause="Accessing an attribute that does not exist on the object.",
        fix_template="Check the object type with type(obj) and verify the attribute name. "
                     "Use hasattr(obj, 'attr') before access or handle with getattr(obj, 'attr', default).",
        severity="error",
        tags="attribute,none,object,python",
    ),
    ErrorFix(
        error_type="ImportError",
        language="python",
        pattern=r"(ImportError|ModuleNotFoundError):\s*(No module named|cannot import name)",
        cause="The module or name is not installed or not in sys.path.",
        fix_template="Install the missing package with pip install <package>. "
                     "Check for typos in the import name. Verify the module is in your PYTHONPATH.",
        severity="error",
        tags="import,module,package,python",
    ),
    ErrorFix(
        error_type="TypeError",
        language="python",
        pattern=r"TypeError:\s*(unsupported operand|.+takes\s+\d+\s+positional|.+not\s+(callable|subscriptable|iterable))",
        cause="Operation applied to an object of inappropriate type.",
        fix_template="Inspect the types of all operands with type(). "
                     "Check function signatures match the call. Use isinstance() for type guards.",
        severity="error",
        tags="type,argument,callable,python",
    ),
    ErrorFix(
        error_type="KeyError",
        language="python",
        pattern=r"KeyError:\s*",
        cause="Dictionary key does not exist.",
        fix_template="Use dict.get(key, default) instead of dict[key]. "
                     "Or check with 'if key in dict:' before access.",
        severity="error",
        tags="key,dict,dictionary,missing,python",
    ),
    ErrorFix(
        error_type="RecursionError",
        language="python",
        pattern=r"RecursionError:\s*maximum recursion depth exceeded",
        cause="Infinite or excessively deep recursion.",
        fix_template="Add a proper base case to the recursive function. "
                     "Consider converting to an iterative approach. "
                     "If legitimate, use sys.setrecursionlimit() cautiously.",
        severity="error",
        tags="recursion,stack,overflow,depth,python",
    ),

    # ── JavaScript ──────────────────────────────────────────────────────
    ErrorFix(
        error_type="TypeError",
        language="javascript",
        pattern=r"TypeError:\s*(Cannot read propert|.+is not a function|.+is undefined|.+is null)",
        cause="Attempted to use undefined or null as an object.",
        fix_template="Add null/undefined checks: use optional chaining (obj?.prop) "
                     "and nullish coalescing (obj ?? default). Verify the variable is "
                     "initialized before use.",
        severity="error",
        tags="undefined,null,property,function,javascript",
    ),
    ErrorFix(
        error_type="ReferenceError",
        language="javascript",
        pattern=r"ReferenceError:\s*(\w+)\s+is\s+not\s+defined",
        cause="Variable or function referenced before declaration.",
        fix_template="Ensure the variable is declared with let/const/var before use. "
                     "Check for typos in the variable name. Verify the module is imported.",
        severity="error",
        tags="reference,undefined,variable,scope,javascript",
    ),
    ErrorFix(
        error_type="UnhandledPromiseRejection",
        language="javascript",
        pattern=r"(Unhandled\s*promise\s*rejection|UnhandledPromiseRejectionWarning)",
        cause="A Promise rejected without a .catch() handler or try/catch in async.",
        fix_template="Add .catch() to every promise chain, or wrap async calls in try/catch. "
                     "Add a global handler: process.on('unhandledRejection', handler).",
        severity="error",
        tags="promise,async,rejection,unhandled,javascript",
    ),
    ErrorFix(
        error_type="SyntaxError",
        language="javascript",
        pattern=r"SyntaxError:\s*(Unexpected token|Unexpected end of)",
        cause="Invalid JavaScript syntax — missing bracket, parenthesis, or semicolon.",
        fix_template="Check for missing closing brackets/parens. Verify JSON is valid. "
                     "Look for accidental use of reserved words.",
        severity="error",
        tags="syntax,token,parse,bracket,javascript",
    ),
    ErrorFix(
        error_type="RangeError",
        language="javascript",
        pattern=r"RangeError:\s*(Maximum call stack|Invalid array length)",
        cause="Value out of allowed range — often infinite recursion or invalid array size.",
        fix_template="Add a base case to recursive functions. "
                     "Validate array lengths before allocation. Check for circular references.",
        severity="error",
        tags="range,stack,recursion,array,javascript",
    ),

    # ── TypeScript ──────────────────────────────────────────────────────
    ErrorFix(
        error_type="TS2322",
        language="typescript",
        pattern=r"TS2322:\s*Type\s+'.*?'\s+is\s+not\s+assignable\s+to\s+type",
        cause="Type mismatch: the assigned value doesn't match the expected type.",
        fix_template="Check the expected type and cast or transform the value. "
                     "Use a type guard (if (x instanceof Y)) or assertion (x as Type) if safe.",
        severity="error",
        tags="type,assignable,mismatch,typescript",
    ),
    ErrorFix(
        error_type="TS2339",
        language="typescript",
        pattern=r"TS2339:\s*Property\s+'.*?'\s+does\s+not\s+exist\s+on\s+type",
        cause="Accessing a property not defined in the type declaration.",
        fix_template="Add the property to the type/interface definition. "
                     "Use optional chaining (obj?.prop) or extend the interface.",
        severity="error",
        tags="property,type,interface,missing,typescript",
    ),
    ErrorFix(
        error_type="TS2345",
        language="typescript",
        pattern=r"TS2345:\s*Argument\s+of\s+type\s+'.*?'\s+is\s+not\s+assignable\s+to\s+parameter",
        cause="Function argument type doesn't match parameter type.",
        fix_template="Transform the argument to match the expected type. "
                     "Use generics or overloads if multiple types are valid.",
        severity="error",
        tags="argument,parameter,type,function,typescript",
    ),
    ErrorFix(
        error_type="TS7006",
        language="typescript",
        pattern=r"TS7006:\s*Parameter\s+'.*?'\s+implicitly\s+has\s+an\s+'any'\s+type",
        cause="noImplicitAny is enabled and the parameter lacks a type annotation.",
        fix_template="Add explicit type annotations to function parameters: "
                     "function foo(param: string) instead of function foo(param).",
        severity="warning",
        tags="any,implicit,annotation,noImplicitAny,typescript",
    ),
    ErrorFix(
        error_type="TS2304",
        language="typescript",
        pattern=r"TS2304:\s*Cannot\s+find\s+name\s+'.*?'",
        cause="Identifier not found — missing import, declaration, or type definition.",
        fix_template="Import the missing symbol or install its @types/ package. "
                     "Check tsconfig.json 'lib' and 'typeRoots' settings.",
        severity="error",
        tags="name,import,declaration,types,typescript",
    ),

    # ── Java ────────────────────────────────────────────────────────────
    ErrorFix(
        error_type="NullPointerException",
        language="java",
        pattern=r"(NullPointerException|java\.lang\.NullPointerException)",
        cause="Dereferencing a null reference.",
        fix_template="Add null checks before method calls: if (obj != null). "
                     "Use Optional<T> for values that may be absent. "
                     "Enable @Nullable/@NonNull annotations.",
        severity="error",
        tags="null,pointer,npe,reference,java",
    ),
    ErrorFix(
        error_type="ClassCastException",
        language="java",
        pattern=r"ClassCastException:\s*.*cannot\s+be\s+cast\s+to",
        cause="Invalid type cast between incompatible classes.",
        fix_template="Use instanceof check before casting: "
                     "if (obj instanceof MyClass) { MyClass m = (MyClass) obj; }. "
                     "Prefer generics over raw types.",
        severity="error",
        tags="cast,class,type,instanceof,java",
    ),
    ErrorFix(
        error_type="StackOverflowError",
        language="java",
        pattern=r"(StackOverflowError|java\.lang\.StackOverflowError)",
        cause="Recursive call without proper termination or extremely deep call stack.",
        fix_template="Add a proper base case to recursive methods. "
                     "Convert deep recursion to iteration. "
                     "Increase stack size with -Xss flag if legitimate.",
        severity="error",
        tags="stack,overflow,recursion,depth,java",
    ),
    ErrorFix(
        error_type="ArrayIndexOutOfBoundsException",
        language="java",
        pattern=r"ArrayIndexOutOfBoundsException:\s*Index\s+\d+\s+out\s+of\s+bounds",
        cause="Array index is negative or >= array.length.",
        fix_template="Check array bounds before access: "
                     "if (i >= 0 && i < arr.length). "
                     "Use enhanced for-loop (for-each) when possible.",
        severity="error",
        tags="array,index,bounds,outofbounds,java",
    ),
    ErrorFix(
        error_type="ConcurrentModificationException",
        language="java",
        pattern=r"ConcurrentModificationException",
        cause="Collection modified while being iterated.",
        fix_template="Use Iterator.remove() for removal during iteration. "
                     "Use ConcurrentHashMap or CopyOnWriteArrayList for concurrent access. "
                     "Collect items to remove and remove after iteration.",
        severity="error",
        tags="concurrent,modification,iterator,collection,java",
    ),

    # ── Go ──────────────────────────────────────────────────────────────
    ErrorFix(
        error_type="nil pointer dereference",
        language="go",
        pattern=r"(nil\s+pointer\s+dereference|invalid\s+memory\s+address\s+or\s+nil\s+pointer)",
        cause="Dereferencing a nil pointer.",
        fix_template="Always check for nil before dereferencing: "
                     "if ptr != nil { use ptr }. "
                     "Return (value, error) tuples and check errors.",
        severity="error",
        tags="nil,pointer,dereference,null,go",
    ),
    ErrorFix(
        error_type="index out of range",
        language="go",
        pattern=r"index\s+out\s+of\s+range\s*\[?\d*\]?",
        cause="Slice or array index exceeds length.",
        fix_template="Check slice length before access: "
                     "if i < len(slice) { use slice[i] }. "
                     "Use range loops to avoid manual indexing.",
        severity="error",
        tags="index,range,slice,array,bounds,go",
    ),
    ErrorFix(
        error_type="goroutine leak",
        language="go",
        pattern=r"(goroutine\s+leak|too\s+many\s+goroutines|all\s+goroutines\s+are\s+asleep)",
        cause="Goroutine blocked forever on channel operation or never terminates.",
        fix_template="Use context.WithCancel or context.WithTimeout for goroutine lifecycle. "
                     "Ensure channels are closed when no more values will be sent. "
                     "Use select with a done channel for graceful shutdown.",
        severity="error",
        tags="goroutine,leak,channel,deadlock,go",
    ),
    ErrorFix(
        error_type="data race",
        language="go",
        pattern=r"(DATA\s+RACE|data\s+race|race\s+detected)",
        cause="Concurrent unsynchronized access to shared memory.",
        fix_template="Protect shared state with sync.Mutex or sync.RWMutex. "
                     "Use channels for goroutine communication. "
                     "Run tests with -race flag: go test -race ./...",
        severity="error",
        tags="race,concurrent,mutex,sync,go",
    ),
    ErrorFix(
        error_type="deadlock",
        language="go",
        pattern=r"fatal\s+error:\s+all\s+goroutines\s+are\s+asleep\s*-\s*deadlock",
        cause="All goroutines are blocked — no goroutine can make progress.",
        fix_template="Check for circular channel dependencies. "
                     "Use buffered channels or select with default case. "
                     "Ensure WaitGroup.Done() is called for every Add().",
        severity="error",
        tags="deadlock,goroutine,channel,waitgroup,go",
    ),

    # ── Rust ────────────────────────────────────────────────────────────
    ErrorFix(
        error_type="E0382",
        language="rust",
        pattern=r"(E0382|borrow\s+of\s+moved\s+value|use\s+of\s+moved\s+value)",
        cause="Value used after being moved (ownership transferred).",
        fix_template="Clone the value with .clone() if needed in multiple places. "
                     "Use references (&T or &mut T) instead of transferring ownership. "
                     "Restructure code to avoid needing the value after the move.",
        severity="error",
        tags="borrow,move,ownership,value,rust",
    ),
    ErrorFix(
        error_type="E0502",
        language="rust",
        pattern=r"(E0502|cannot\s+borrow.*as\s+(im)?mutable.*also\s+borrowed)",
        cause="Conflicting borrows: cannot have mutable and immutable borrow simultaneously.",
        fix_template="Restructure to avoid overlapping borrows. "
                     "Use scoping to limit borrow lifetimes. "
                     "Consider using Cell<T> or RefCell<T> for interior mutability.",
        severity="error",
        tags="borrow,mutable,immutable,reference,rust",
    ),
    ErrorFix(
        error_type="E0308",
        language="rust",
        pattern=r"(E0308|mismatched\s+types|expected\s+.*,\s+found)",
        cause="Type mismatch between expected and actual types.",
        fix_template="Check the expected return type. Use .into() or From/Into traits "
                     "for conversions. Use as keyword for primitive casts.",
        severity="error",
        tags="type,mismatch,expected,found,rust",
    ),
    ErrorFix(
        error_type="E0599",
        language="rust",
        pattern=r"(E0599|no\s+method\s+named\s+.*\s+found\s+for)",
        cause="Method not found on the type — missing trait import or wrong type.",
        fix_template="Import the trait that provides the method with 'use TraitName;'. "
                     "Check the type implements the required trait. "
                     "Verify you're calling on the correct type (not a reference or wrapper).",
        severity="error",
        tags="method,trait,impl,found,rust",
    ),
    ErrorFix(
        error_type="thread panic",
        language="rust",
        pattern=r"(thread\s+'.*'\s+panicked|unwrap\(\)\s+on\s+a\s+`(None|Err)`)",
        cause="Panicked due to unwrap() on None or Err, or explicit panic!().",
        fix_template="Replace unwrap() with pattern matching or ? operator. "
                     "Use .unwrap_or_default() or .expect('message') for clearer errors. "
                     "Handle Result/Option types explicitly.",
        severity="error",
        tags="panic,unwrap,none,err,thread,rust",
    ),

    # ── C# ──────────────────────────────────────────────────────────────
    ErrorFix(
        error_type="NullReferenceException",
        language="csharp",
        pattern=r"(NullReferenceException|Object\s+reference\s+not\s+set\s+to\s+an\s+instance)",
        cause="Accessing a member on a null object reference.",
        fix_template="Use null-conditional operator: obj?.Method(). "
                     "Enable nullable reference types (#nullable enable). "
                     "Check for null before access or use ?? for defaults.",
        severity="error",
        tags="null,reference,object,instance,csharp",
    ),
    ErrorFix(
        error_type="InvalidCastException",
        language="csharp",
        pattern=r"InvalidCastException:\s*Unable\s+to\s+cast",
        cause="Invalid type cast between incompatible types.",
        fix_template="Use 'as' operator with null check: var x = obj as MyType; if (x != null). "
                     "Or use 'is' pattern: if (obj is MyType x) { use x }.",
        severity="error",
        tags="cast,type,invalid,csharp",
    ),
    ErrorFix(
        error_type="StackOverflowException",
        language="csharp",
        pattern=r"StackOverflowException",
        cause="Infinite recursion or very deep call stack.",
        fix_template="Add a base case to recursive methods. "
                     "Convert to iterative with explicit stack. "
                     "Check for property getter/setter calling itself.",
        severity="error",
        tags="stack,overflow,recursion,csharp",
    ),
    ErrorFix(
        error_type="ArgumentNullException",
        language="csharp",
        pattern=r"ArgumentNullException:\s*Value\s+cannot\s+be\s+null",
        cause="A null argument was passed to a method that doesn't accept null.",
        fix_template="Validate arguments with ArgumentNullException.ThrowIfNull(). "
                     "Add null checks at method entry. "
                     "Use [NotNull] attribute for compile-time checking.",
        severity="error",
        tags="argument,null,parameter,validation,csharp",
    ),
    ErrorFix(
        error_type="TaskCanceledException",
        language="csharp",
        pattern=r"(TaskCanceledException|OperationCanceledException)",
        cause="An async operation was canceled via CancellationToken or timed out.",
        fix_template="Handle TaskCanceledException in try/catch around async calls. "
                     "Check CancellationToken.IsCancellationRequested before long operations. "
                     "Set appropriate timeouts with CancellationTokenSource.",
        severity="error",
        tags="task,canceled,async,timeout,cancellation,csharp",
    ),
]


# ---------------------------------------------------------------------------
# Markdown seed data
# ---------------------------------------------------------------------------

_PATTERN_DOCS = {
    "clean-code-naming-conventions.md": {
        "title": "Clean Code Naming Conventions",
        "tags": "naming, conventions, clean-code, readability",
        "content": """## Overview

Good naming is the foundation of readable code. Names should reveal intent,
avoid disinformation, and make the code self-documenting.

## Variable Naming

### Use Intention-Revealing Names

Bad: `int d; // elapsed time in days`

Good: `int elapsedTimeInDays;`

### Avoid Disinformation

Don't use `accountList` unless it's actually a List. Prefer `accounts` or
`accountGroup` for non-list containers.

### Use Pronounceable Names

Bad: `genymdhms` (generation date, year, month, day, hour, minute, second)

Good: `generationTimestamp`

## Function Naming

### Use Verb Phrases

Functions do things — name them with verbs:
- `getUserById(id)` not `user(id)`
- `calculateTotal(items)` not `total(items)`
- `isValid()` for boolean returns

### Keep Functions Focused

If you can't name a function without using "and" or "or", it probably does
too much. Split it into separate well-named functions.

## Class Naming

### Use Noun Phrases

Classes represent things — use nouns:
- `UserRepository` not `ManageUsers`
- `PaymentProcessor` not `ProcessPayment`

### Avoid Generic Names

Avoid `Manager`, `Handler`, `Data`, `Info` unless they genuinely describe
the responsibility. Prefer domain-specific names.

## Constants

### Use Screaming Snake Case

In most languages, use `MAX_RETRY_COUNT` not `maxRetryCount` for constants.
This makes constants visually distinct from variables.
""",
    },
    "error-handling-best-practices.md": {
        "title": "Error Handling Best Practices",
        "tags": "error-handling, exceptions, resilience, best-practices",
        "content": """## Overview

Good error handling makes software robust without obscuring the main logic.
Follow these principles across all languages.

## Principle 1: Fail Fast

Validate inputs at the boundary of your system. Don't let invalid data
propagate deep into your code where failures become harder to diagnose.

```python
def process_order(order):
    if not order:
        raise ValueError("Order cannot be None")
    if order.total < 0:
        raise ValueError(f"Invalid order total: {order.total}")
    # ... proceed with valid order
```

## Principle 2: Use Specific Exception Types

Catch the most specific exception type possible. Never use bare `except:`
or `catch (Exception e)` unless you're at the top-level error boundary.

## Principle 3: Don't Swallow Exceptions

Logging an error and continuing as if nothing happened is often worse than
crashing. If you catch an exception, handle it meaningfully:

- Retry the operation (with backoff)
- Return a default value (if safe)
- Re-raise with additional context
- Translate to a domain-specific error

## Principle 4: Provide Context

Include enough context in error messages to diagnose the problem without
access to the source code:

Bad: `Error: invalid input`
Good: `Error: User ID '12345' not found in database 'users_prod'`

## Principle 5: Use Error Boundaries

Create layers where errors are caught, logged, and translated into
appropriate responses for the caller (HTTP 500, exit code 1, etc.).

## Anti-Patterns

### Pokemon Exception Handling
```
try:
    everything()
except:  # Gotta catch 'em all
    pass
```

### Error Code Returns in Exception-Based Languages
Don't return error codes when the language has exceptions. Use the
language's native error mechanism.

### Throwing Exceptions for Flow Control
Exceptions should be exceptional. Don't use try/catch for normal
branching logic.
""",
    },
    "async-patterns.md": {
        "title": "Async Programming Patterns",
        "tags": "async, concurrency, promises, patterns",
        "content": """## Overview

Asynchronous programming enables responsive applications but introduces
complexity. These patterns help manage that complexity.

## Pattern 1: Promise/Future Chaining

Chain operations that depend on each other sequentially:

```javascript
fetchUser(id)
    .then(user => fetchOrders(user.id))
    .then(orders => processOrders(orders))
    .catch(error => handleError(error));
```

## Pattern 2: Parallel Execution

Run independent operations concurrently:

```javascript
const [user, config, permissions] = await Promise.all([
    fetchUser(id),
    fetchConfig(),
    fetchPermissions(id),
]);
```

## Pattern 3: Async Iteration

Process streams of data asynchronously:

```python
async for message in websocket:
    await process_message(message)
```

## Pattern 4: Cancellation

Always support cancellation for long-running async operations:

```csharp
async Task<Data> FetchData(CancellationToken token)
{
    token.ThrowIfCancellationRequested();
    var response = await client.GetAsync(url, token);
    return await response.Content.ReadAsAsync<Data>();
}
```

## Pattern 5: Retry with Backoff

Retry transient failures with exponential backoff:

```python
async def fetch_with_retry(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await fetch(url)
        except TransientError:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
```

## Anti-Patterns

### Fire and Forget
Don't start async operations without awaiting or tracking them.
Uncaught rejections crash Node.js and cause silent failures elsewhere.

### Mixing Callbacks and Promises
Pick one style and stick with it. Mixing leads to lost errors and
tangled control flow.

### Blocking the Event Loop
Never use synchronous I/O in an async context. It defeats the purpose
and blocks all concurrent operations.
""",
    },
}

_ADR_DOCS = {
    "adr-001-use-qdrant-for-vector-store.md": {
        "title": "ADR-001: Use Qdrant for Vector Store",
        "tags": "adr, qdrant, vector-store, architecture",
        "content": """## Status

Accepted

## Context

The AgentChanti knowledge base requires a vector store for semantic search
over code symbols and documentation. We evaluated several options:

1. **Qdrant** — Open-source vector database, runs locally via Docker
2. **ChromaDB** — Lightweight embedded vector store
3. **FAISS** — Facebook's similarity search library (no server)
4. **Pinecone** — Cloud-hosted managed vector database

## Decision

We chose **Qdrant** for the following reasons:

### Advantages
- **Local-first**: Runs as a Docker container, no cloud dependency
- **Rich filtering**: Supports payload-based filtering alongside vector search
- **Snapshot support**: Can export/import collections for offline distribution
- **Production-grade**: Handles millions of vectors with consistent performance
- **REST + gRPC APIs**: Flexible integration options

### Trade-offs
- Requires Docker for local development
- Heavier footprint than embedded solutions like ChromaDB
- Additional operational complexity vs. in-memory FAISS

## Consequences

- All vector operations go through the Qdrant HTTP API on localhost:6333
- The CLI includes `agentchanti kb qdrant start/stop/status` commands
- Pre-built snapshots ship with the CLI for offline-first operation
- Collection naming convention: `local_{project_slug}` for per-project,
  `global_kb` for the shared knowledge base
""",
    },
    "adr-002-tree-sitter-for-ast-parsing.md": {
        "title": "ADR-002: Use Tree-sitter for AST Parsing",
        "tags": "adr, tree-sitter, parsing, ast, architecture",
        "content": """## Status

Accepted

## Context

Building a code graph requires parsing source code into an AST across
multiple languages. We evaluated:

1. **Tree-sitter** — Incremental parsing library with grammars for 100+ languages
2. **Language-specific parsers** — ast (Python), @babel/parser (JS), etc.
3. **ctags/cscope** — Symbol-level indexing without full AST
4. **LSP servers** — Language Server Protocol for per-language intelligence

## Decision

We chose **Tree-sitter** for the following reasons:

### Advantages
- **Multi-language**: One parser framework for Python, JavaScript, TypeScript,
  Java, Go, Rust, C, C++, Ruby, PHP, C# — and growing
- **Incremental parsing**: Re-parses only changed portions, enabling fast
  watch-mode updates
- **Concrete syntax tree**: Preserves all source text, enabling accurate
  line-number tracking and code extraction
- **Widely adopted**: Used by GitHub, Neovim, Zed, and Helix editors

### Trade-offs
- Grammars need per-language installation (`tree-sitter-python`, etc.)
- Less semantic depth than dedicated LSP servers (no type inference)
- Some grammars have edge cases with advanced language features

## Consequences

- The parser module wraps Tree-sitter for all supported languages
- Each language grammar is an optional dependency
- Symbol extraction traverses the Tree-sitter CST to find functions,
  classes, imports, and call sites
- The code graph (NetworkX DiGraph) is built from extracted symbols
""",
    },
}

_DOC_DOCS = {
    "tree-sitter-usage-guide.md": {
        "title": "Tree-sitter Usage Guide",
        "tags": "tree-sitter, parsing, guide, setup",
        "content": """## Overview

Tree-sitter is the core parsing engine for the AgentChanti local knowledge
base. This guide covers setup, usage, and troubleshooting.

## Installation

Install tree-sitter and the language grammars you need:

```bash
pip install tree-sitter
pip install tree-sitter-python tree-sitter-javascript tree-sitter-typescript
pip install tree-sitter-java tree-sitter-go tree-sitter-rust
```

## How It Works

### Parsing Flow
1. Source code is loaded as bytes
2. Tree-sitter parses it into a Concrete Syntax Tree (CST)
3. AgentChanti traverses the CST to extract symbols
4. Symbols are added to the NetworkX code graph

### Symbol Types Extracted
- **Functions/Methods**: name, parameters, return type, docstring, body
- **Classes**: name, base classes, docstring, method list
- **Imports**: module name, imported names
- **Variables**: module-level assignments

## Custom Queries

Tree-sitter supports S-expression queries for targeted extraction:

```scheme
;; Find all function definitions
(function_definition
  name: (identifier) @function.name
  parameters: (parameters) @function.params)
```

## Troubleshooting

### Grammar Not Found
Ensure the language grammar package is installed:
```bash
pip install tree-sitter-{language}
```

### Parse Errors
Tree-sitter is error-tolerant — it produces a partial tree even with
syntax errors. Check for `ERROR` nodes in the tree to find problem areas.

### Performance
For large files (>10K lines), parsing may take >100ms. The incremental
parser helps by only re-parsing changed regions during watch mode.
""",
    },
    "qdrant-local-setup-guide.md": {
        "title": "Qdrant Local Setup Guide",
        "tags": "qdrant, setup, docker, vector-store, guide",
        "content": """## Overview

Qdrant runs locally as a Docker container for the AgentChanti semantic
search layer. This guide covers setup, configuration, and management.

## Prerequisites

- Docker installed and running
- At least 512MB free RAM for the Qdrant container

## Quick Start

```bash
# Start Qdrant via AgentChanti CLI
agentchanti kb qdrant start

# Check status
agentchanti kb qdrant status

# Stop when done
agentchanti kb qdrant stop
```

## Manual Docker Setup

If you prefer manual control:

```bash
docker run -d \\
  --name agentchanti-qdrant \\
  -p 6333:6333 \\
  -v ~/.agentchanti/qdrant:/qdrant/storage \\
  qdrant/qdrant
```

## Collections

AgentChanti creates these collections:

| Collection | Purpose | Vector Size |
|------------|---------|-------------|
| `local_{project}` | Per-project code symbols | 1536 |
| `global_kb` | Shared knowledge base | 1536 |

## Storage

Qdrant data is stored at:
- Per-project: `{project}/.agentchanti/kb/local/qdrant/`
- Global: managed by the Qdrant container's shared volume

## Backup & Restore

```bash
# Create a snapshot
curl -X POST 'http://localhost:6333/collections/global_kb/snapshots'

# List snapshots
curl 'http://localhost:6333/collections/global_kb/snapshots'
```

## Troubleshooting

### Port Already in Use
If port 6333 is occupied, stop the existing container or change the port
in your configuration.

### Container Won't Start
Check Docker logs: `docker logs agentchanti-qdrant`

### Collection Not Found
Ensure you've run indexing first: `agentchanti kb index && agentchanti kb embed`
""",
    },
}

_BEHAVIORAL_DOCS = {
    "code-review-instructions.md": {
        "title": "Code Review Instructions",
        "tags": "code-review, instructions, behavioral, quality",
        "content": """## Overview

When performing code review, follow these structured instructions to
ensure consistent, thorough, and constructive feedback.

## Review Checklist

### 1. Correctness
- Does the code do what it's supposed to do?
- Are edge cases handled?
- Are there potential null/undefined access issues?
- Are error conditions handled appropriately?

### 2. Security
- Is user input validated and sanitized?
- Are there SQL injection or XSS vulnerabilities?
- Are secrets hardcoded?
- Are permissions checked appropriately?

### 3. Performance
- Are there N+1 query patterns?
- Are expensive operations cached when appropriate?
- Are there unnecessary allocations in hot paths?
- Could data structures be more efficient?

### 4. Readability
- Are variable and function names descriptive?
- Is the code self-documenting or well-commented?
- Are functions at a single level of abstraction?
- Is the control flow easy to follow?

### 5. Maintainability
- Does the code follow existing patterns in the codebase?
- Are there duplicated code blocks that should be extracted?
- Is the code testable?
- Are dependencies minimized?

## Giving Feedback

### Be Specific
Bad: "This function is too complex"
Good: "This function has 3 levels of nesting — extract the inner loop into a helper"

### Explain Why
Don't just say what to change — explain why the change matters.

### Distinguish Severity
- **Blocker**: Must fix before merge (bugs, security issues)
- **Suggestion**: Recommended improvement (naming, structure)
- **Nit**: Minor style preference (formatting, comment wording)
""",
    },
    "error-analysis-instructions.md": {
        "title": "Error Analysis Instructions",
        "tags": "error-analysis, debugging, instructions, behavioral",
        "content": """## Overview

When analyzing errors, follow this systematic approach to identify root
causes and suggest effective fixes.

## Step 1: Classify the Error

Determine the error category:
- **Syntax Error**: Code doesn't parse — missing brackets, typos
- **Type Error**: Wrong data type for an operation
- **Runtime Error**: Crash during execution — null access, index bounds
- **Logic Error**: Code runs but produces wrong results
- **Resource Error**: File not found, connection refused, timeout

## Step 2: Read the Full Stack Trace

- Start from the **bottom** of the stack trace (the actual error)
- Work **upward** to find the originating call in user code
- Identify if the error is in user code or library code
- Note the file, line number, and function name

## Step 3: Identify Root Cause

Common root causes:
- **Missing null check**: Object is None/null/nil when accessed
- **Wrong assumption about data**: Expected format differs from actual
- **Race condition**: Concurrent access to shared state
- **State management**: Component lifecycle or state machine error
- **Configuration**: Wrong environment, missing env vars, wrong paths

## Step 4: Suggest a Fix

A good fix suggestion includes:
1. **What to change**: The specific code modification
2. **Why it fixes the issue**: Connect the change to the root cause
3. **How to prevent recurrence**: Tests, type guards, validation

## Step 5: Suggest Preventive Measures

- Add unit tests that reproduce the error
- Add type annotations/guards at the boundary
- Improve error messages for faster future diagnosis
- Consider if similar bugs could exist elsewhere
""",
    },
}


# ---------------------------------------------------------------------------
# Markdown chunk helpers
# ---------------------------------------------------------------------------

def _chunk_markdown(text: str, title: str, min_size: int = 100, max_size: int = 1500) -> list[str]:
    """
    Split markdown text into chunks by heading sections.

    Parameters
    ----------
    text:
        The markdown body (without frontmatter).
    title:
        Title to prepend to every chunk for context.
    min_size:
        Merge sections smaller than this.
    max_size:
        Split sections larger than this.

    Returns
    -------
    list[str]
        List of text chunks.
    """
    import re as _re
    sections: list[str] = []
    current: list[str] = []

    for line in text.split("\n"):
        if _re.match(r"^#{2,3}\s+", line) and current:
            sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current))

    # Merge small sections
    merged: list[str] = []
    buf = ""
    for sec in sections:
        if len(buf) + len(sec) < min_size:
            buf = buf + "\n" + sec if buf else sec
        else:
            if buf:
                merged.append(buf)
            buf = sec
    if buf:
        merged.append(buf)

    # Split oversized chunks
    final: list[str] = []
    for chunk in merged:
        if len(chunk) <= max_size:
            final.append(f"Title: {title}\n\n{chunk.strip()}")
        else:
            # Split at paragraph boundaries
            paragraphs = chunk.split("\n\n")
            sub_buf = ""
            for para in paragraphs:
                if len(sub_buf) + len(para) > max_size and sub_buf:
                    final.append(f"Title: {title}\n\n{sub_buf.strip()}")
                    sub_buf = para
                else:
                    sub_buf = sub_buf + "\n\n" + para if sub_buf else para
            if sub_buf:
                final.append(f"Title: {title}\n\n{sub_buf.strip()}")

    return final


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

def _write_md_file(
    directory: str,
    filename: str,
    title: str,
    category: str,
    tags: str,
    language: str,
    content: str,
) -> str:
    """Write a markdown file with frontmatter.  Returns the absolute path."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    frontmatter = (
        "---\n"
        f'title: "{title}"\n'
        f'category: "{category}"\n'
        f'tags: "{tags}"\n'
        f'language: "{language}"\n'
        f'version: "1.0.0"\n'
        f'created_at: "2025-01-01"\n'
        "---\n\n"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(frontmatter + content)
    return path


# ---------------------------------------------------------------------------
# Main seeder
# ---------------------------------------------------------------------------

def seed(
    embed: bool = True,
    project_root: Optional[str] = None,
    api_client=None,
) -> dict:
    """
    Seed the global knowledge base with sample data.

    Parameters
    ----------
    embed:
        If True, embed markdown documents into the Qdrant ``global_kb``
        collection.  Requires Qdrant running and OpenAI API key.
    project_root:
        Project root for Qdrant storage path.  Defaults to cwd.
    api_client:
        LLM client to use for embedding.

    Returns
    -------
    dict
        Summary with keys: errors_seeded, docs_seeded, chunks_embedded.
    """
    project_root = project_root or os.getcwd()
    summary = {"errors_seeded": 0, "docs_seeded": 0, "chunks_embedded": 0}

    # ── 1. Seed errors.db ───────────────────────────────────────────────
    db_path = _errors_db_path()
    edict = ErrorDict(db_path)
    edict.clear()
    edict.bulk_insert(_ERROR_SEEDS)
    summary["errors_seeded"] = edict.count()
    logger.info("Seeded %d errors into %s", summary["errors_seeded"], db_path)

    # ── 2. Write markdown files ─────────────────────────────────────────
    md_files: list[tuple[str, str, str]] = []  # (path, category, title)

    for filename, meta in _PATTERN_DOCS.items():
        path = _write_md_file(
            os.path.join(_REGISTRY_DIR, "patterns"),
            filename,
            meta["title"],
            "pattern",
            meta["tags"],
            "all",
            meta["content"],
        )
        md_files.append((path, "pattern", meta["title"]))

    for filename, meta in _ADR_DOCS.items():
        path = _write_md_file(
            os.path.join(_REGISTRY_DIR, "adrs"),
            filename,
            meta["title"],
            "adr",
            meta["tags"],
            "all",
            meta["content"],
        )
        md_files.append((path, "adr", meta["title"]))

    for filename, meta in _DOC_DOCS.items():
        path = _write_md_file(
            os.path.join(_REGISTRY_DIR, "docs"),
            filename,
            meta["title"],
            "doc",
            meta["tags"],
            "all",
            meta["content"],
        )
        md_files.append((path, "doc", meta["title"]))

    for filename, meta in _BEHAVIORAL_DOCS.items():
        path = _write_md_file(
            os.path.join(_REGISTRY_DIR, "behavioral"),
            filename,
            meta["title"],
            "behavioral",
            meta["tags"],
            "all",
            meta["content"],
        )
        md_files.append((path, "behavioral", meta["title"]))

    summary["docs_seeded"] = len(md_files)
    logger.info("Wrote %d markdown documents", summary["docs_seeded"])

    # ── 3. Embed into Qdrant/SQLite (optional) ──────────────────────────
    if embed:
        try:
            if api_client is None:
                raise ValueError("api_client required for embedding")
            embedded = _embed_md_files(md_files, project_root, api_client)
            summary["chunks_embedded"] = embedded
        except Exception as exc:
            logger.warning("Embedding skipped: %s", exc)
            summary["chunks_embedded"] = 0

    return summary


def _embed_md_files(
    md_files: list[tuple[str, str, str]],
    project_root: str,
    api_client,
) -> int:
    """
    Embed markdown files into the `global_kb` collection.

    Reuses the embedding helpers from kb.local.embedder.

    Returns the total number of chunks embedded.
    """
    from ..local.embedder import _embed_batch, BATCH_SIZE, make_point_id
    from .store import _get_global_vector_store
    from ...config import Config

    cfg = Config.load()
    embed_model = cfg.EMBEDDING_MODEL or cfg.DEFAULT_MODEL

    # Create a store for the global_kb collection
    store = _get_global_vector_store()
    total_chunks = 0

    for filepath, category, title in md_files:
        with open(filepath, encoding="utf-8") as fh:
            raw = fh.read()

        # Strip frontmatter
        body = raw
        if raw.startswith("---"):
            parts = raw.split("---", 2)
            if len(parts) >= 3:
                body = parts[2]

        # Parse frontmatter for metadata
        meta = _parse_frontmatter(raw)
        tags = [t.strip() for t in meta.get("tags", "").split(",") if t.strip()]
        language = meta.get("language", "all")
        version = meta.get("version", "1.0.0")

        # Get relative path within registry
        rel_path = os.path.relpath(filepath, _GLOBAL_DIR)

        chunks = _chunk_markdown(body, title)
        if not chunks:
            continue

        # Embed in batches
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            try:
                vectors = _embed_batch(api_client, batch, embed_model)
            except RuntimeError as exc:
                logger.warning("Embedding failed for %s: %s", filepath, exc)
                continue

            points = []
            for j, (chunk_text, vector) in enumerate(zip(batch, vectors)):
                point_id = str(uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"global:{rel_path}:{i + j}",
                ))
                payload = {
                    "file": rel_path,
                    "category": category,
                    "title": title,
                    "language": language,
                    "tags": tags,
                    "version": version,
                    "source": "core",
                }
                points.append((point_id, vector, payload))

            store.upsert(points)
            total_chunks += len(batch)

    return total_chunks


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Parse YAML frontmatter from markdown text."""
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    meta: dict[str, str] = {}
    for line in parts[1].strip().split("\n"):
        if ":" in line:
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip().strip('"').strip("'")
    return meta



