# flux-importer

[![Crates.io](https://img.shields.io/crates/v/flux-importer)](https://crates.io/crates/flux-importer)
[![Docs.rs](https://docs.rs/flux-importer/badge.svg)](https://docs.rs/flux-importer)
[![License](https://img.shields.io/crates/l/flux-importer)](LICENSE)

> **Flux bytecode → synthetic MIR.**
>
> The bridge that lets agents compile their own GPU kernels without writing a single line of Rust source.

---

## Table of Contents

1. [What is this crate?](#what-is-this-crate)
2. [Where it sits in the stack](#where-it-sits-in-the-stack)
3. [Quick start](#quick-start)
4. [The Flux instruction set](#the-flux-instruction-set)
5. [Ternary operations: first-class `{-1, 0, +1}`](#ternary-operations-first-class--10-1)
6. [Construct imports (git-native dependencies)](#construct-imports-git-native-dependencies)
7. [Error handling](#error-handling)
8. [Why Flux?](#why-flux)
9. [Further reading](#further-reading)

---

## What is this crate?

If you are coming from a traditional compiler background, think of `flux-importer` as a **front-end plugin** for the cuda-oxide pipeline. It takes a flat array of Flux bytecode bytes and lifts them into *synthetic MIR* — a structured intermediate representation that looks exactly like the output of `mir-importer` (the Rust-source-to-MIR stage).

Why synthetic? Because no Rust source code is involved. The MIR is synthesized directly from bytecode. This means:

- LLMs can emit Flux bytecode instead of fragile stringly-Rust source.
- Higher-level DSLs can target Flux as a portable GPU assembly layer.
- Hand-optimizers can write bytecode directly when they need fine-grained control.

The crate is two things in one:

1. **`FluxToMir`** — a bytecode translator (decoder + SSA-like block builder).
2. **`FluxGpuBuilder`** — a programmatic API for constructing GPU kernels in MIR without touching bytecode at all.

---

## Where it sits in the stack

```text
┌─────────────────────────────────────────────────────────────┐
│                      Agent Intent                           │
│              (LLM prompt / user script / agent plan)        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Flux Bytecode                            │
│              Flat, compact, agent-native                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         │    THIS CRATE             │
         │    flux-importer          │
         │    (translation + builder)│
         └─────────────┬─────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 Synthetic MIR                               │
│  (compatible with cuda-oxide mir-importer output)           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  mir-lower                                  │
│         (MIR → Pliron IR, NVVM dialect)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  llvm-export                                │
│              (Pliron IR → LLVM IR)                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              LLVM IR → PTX                                  │
│         (compiled by nvptx backend)                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              CUDA Driver / Runtime                          │
│                    (execution)                              │
└─────────────────────────────────────────────────────────────┘
```

**Key insight:** From the perspective of `mir-lower`, synthetic MIR and native MIR are indistinguishable. Once we have crossed the `flux-importer` boundary, the rest of the cuda-oxide pipeline operates exactly as it would for code compiled from Rust source. This is deliberate — we did not want to fork the backend. We simply added a new front door.

---

## Quick start

### Translating raw bytecode

The most direct way to use the crate is to hand it a slice of bytes and receive a `MirModule`:

```rust
use flux_importer::{FluxToMir, ImportConfig};

// Bytecode: MOVI R0, 42; HALT
let bytecode = vec![0x01, 0x00, 0x2A, 0x00, 0xFF];

let config = ImportConfig::default();
let module = FluxToMir::translate(&bytecode, &config)?;

// `module` is now synthetic MIR ready for mir-lower.
assert_eq!(module.functions[0].name, "flux_main");
```

### Using the GPU builder

If you prefer a typed API over raw bytes, use `FluxGpuBuilder`. This is especially useful when you are generating kernels procedurally or stitching together imported constructs:

```rust
use flux_importer::FluxGpuBuilder;

// Build a simple map kernel that launches 256 threads per block.
let kernel = FluxGpuBuilder::build_map_kernel("abs_map", 256);

assert_eq!(kernel.name, "abs_map");
assert_eq!(kernel.params.len(), 3);   // input, output, len
assert_eq!(kernel.blocks.len(), 3);   // entry, body, exit
```

### Adding a construct import

Construct imports are git-native dependencies — reusable GPU kernels published as repositories. Think of them as crates.io for kernels:

```rust
use flux_importer::FluxGpuBuilder;

let module = FluxGpuBuilder::new("my_kernel")
    .import_construct(
        "SuperInstance/ternary-attention-kernel",
        "v2.0.0",
        "ternary_attention_forward",
    )
    .build();

assert_eq!(module.imports[0].symbol, "ternary_attention_forward");
```

---

## The Flux instruction set

Flux is a load/store architecture with a compact, fixed-width encoding. Every instruction is easy to decode in a single pass — there is no variable-length metadata to parse, which makes it ideal for LLM generation.

| Opcode | Mnemonic | Operands | Description |
|--------|----------|----------|-------------|
| `0x01` | **MOVI** | `Rd, imm16` | Move 16-bit immediate (sign-extended to `i64`) into general-purpose register `Rd`. |
| `0x02` | **ADD** | `Rd, Rs1, Rs2` | Integer addition: `Rd = Rs1 + Rs2`. |
| `0x03` | **SUB** | `Rd, Rs1, Rs2` | Integer subtraction: `Rd = Rs1 - Rs2`. |
| `0x04` | **MUL** | `Rd, Rs1, Rs2` | Integer multiplication: `Rd = Rs1 * Rs2`. |
| `0x10` | **TADD** | `Rd, Ra, Rb` | Ternary addition on `{-1, 0, +1}` values. |
| `0x11` | **TMUL** | `Rd, Ra, Rb` | Ternary multiplication on `{-1, 0, +1}` values. |
| `0x20` | **THREAD_IDX** | `Rd, dim` | Read CUDA thread index along dimension `dim` (`0=x`, `1=y`, `2=z`). |
| `0x21` | **SYNC_THREADS** | — | Block-level barrier (`__syncthreads()`). |
| `0xFF` | **HALT** | — | End of program / kernel return. |

### Encoding details

- **MOVI** consumes 4 bytes: `[opcode, reg, imm_lo, imm_hi]`.
- **Arithmetic** (`ADD`, `SUB`, `MUL`, `TADD`, `TMUL`) consumes 4 bytes: `[opcode, rd, rs1, rs2]`.
- **THREAD_IDX** consumes 3 bytes: `[opcode, rd, dim]`.
- **SYNC_THREADS** and **HALT** are single-byte instructions.

### A complete example

Here is a minimal kernel that adds two vectors element-wise. In bytecode:

```rust
let bytecode = vec![
    // MOVI R0, 0      // loop counter i
    0x01, 0x00, 0x00, 0x00,
    // THREAD_IDX R1, 0
    0x20, 0x01, 0x00,
    // ADD R2, R0, R1  // global index = i + threadIdx.x
    0x02, 0x02, 0x00, 0x01,
    // SYNC_THREADS
    0x21,
    // HALT
    0xFF,
];

let module = FluxToMir::translate(&bytecode, &ImportConfig::default())?;
```

---

## Ternary operations: first-class `{-1, 0, +1}`

Ternary logic is not an afterthought in Flux — it is a **first-class numeric type** alongside `i32`, `f32`, and pointers. The rationale is simple: many emergent computing patterns (sparse attention, cellular automata, consensus voting, symbolic gradient estimation) naturally map to three-state values rather than full-width integers.

By elevating ternary to the type system, the compiler can:

1. **Generate compact instructions.** A ternary value fits in two bits; eight of them pack into a single byte. The backend can emit custom PTX or tensor-core microcode when it sees `MirType::Ternary`.
2. **Avoid semantic drift.** If you represent ternary states as `{-1, 0, 1}` in `i8`, optimizers may constant-fold or vectorize in ways that break the three-state invariant. A distinct type protects that contract.
3. **Enable hardware specialization.** Future Flux targets (TPUs with ternary MAC units, memristor crossbars, etc.) can pattern-match on `TAdd` and `TMul` directly.

### Ternary MIR operations

| MIR Operation | Semantics |
|---------------|-----------|
| `MirTernaryOp::TAdd` | Balanced ternary addition: the sum of two `{-1, 0, +1}` values is clamped back into the set. |
| `MirTernaryOp::TMul` | Balanced ternary multiplication: sign × sign, with `0` absorbing. |
| `MirTernaryOp::TCompose` | Cascade two ternary values (function composition in the ternary semiring). |
| `MirTernaryOp::TConsensus` | Majority vote across ternary inputs (useful for agent consensus layers). |

In bytecode, `TADD` (`0x10`) and `TMUL` (`0x11`) use the same 4-byte encoding as integer arithmetic. The translator simply emits `MirValue::TernaryOp` instead of `MirValue::BinaryOp`.

---

## Construct imports (git-native dependencies)

A `ConstructImport` is a declaration that a kernel depends on an externally defined GPU capability identified by a Git repository. This is how Flux achieves **composability without a package registry**:

```rust
pub struct ConstructImport {
    pub repo: String,        // e.g. "SuperInstance/ternary-attention-kernel"
    pub version: String,     // e.g. "v2.0.0" or a commit hash
    pub symbol: String,      // The kernel/function name inside that repo
    pub identity_did: Option<String>, // Optional DID for identity verification
}
```

During the `mir-lower` → `llvm-export` phase, the pipeline resolves these imports by cloning (or caching) the referenced repository and linking the declared symbol. Because the identity is rooted in Git history, constructs are **content-addressable and auditable** — you know exactly which kernel code you are running.

Use `FluxGpuBuilder::import_construct` to add them declaratively, or populate `MirModule::imports` directly if you are generating MIR by hand.

---

## Error handling

Translation can fail for five well-defined reasons. Every error carries enough context to pinpoint the offending byte:

| Error Variant | When it happens | Fields |
|---------------|-----------------|--------|
| `InvalidOpcode` | The decoder encounters a byte that is not a recognized opcode. | `offset`, `byte` |
| `UnexpectedEnd` | The bytecode stream ends in the middle of a multi-byte instruction. | `offset`, `expected` (remaining bytes needed) |
| `RegisterOutOfBounds` | A register index exceeds `max_gp_registers` or `max_fp_registers` from `ImportConfig`. | `index`, `max` |
| `InvalidJump` | A control-flow instruction targets an offset outside the bytecode. | `from`, `to` |
| `UnsupportedGpu` | An operation is fundamentally incompatible with GPU lowering (e.g., unbounded recursion). | `op`, `reason` |

All errors implement `std::error::Error` and `Display`, so they integrate cleanly with `anyhow`, `eyre`, or `thiserror` in downstream crates.

### Example: handling translation errors

```rust
use flux_importer::{FluxToMir, ImportConfig, ImportError};

let bad = vec![0xFE]; // 0xFE is not a valid opcode
match FluxToMir::translate(&bad, &ImportConfig::default()) {
    Err(ImportError::InvalidOpcode { offset, byte }) => {
        eprintln!("Byte 0x{byte:02x} at offset {offset} is not a Flux opcode.");
    }
    Ok(_) => panic!("should have failed"),
    Err(e) => eprintln!("Other error: {e}"),
}
```

---

## Why Flux?

Let us step back and look at the landscape.

If you want an LLM or an autonomous agent to compile code for a GPU today, the standard path looks like this:

1. Generate Rust or CUDA C++ source as a string.
2. Hope the string parses.
3. Hope it type-checks.
4. Hope it links.
5. Hope it runs.

Each "hope" is a failure mode. Source generation is brittle; syntax errors are common; and the agent has no visibility into whether the output is valid until the full compiler pipeline rejects it.

Flux inverts this. Bytecode is **structural by construction**. There are no mismatched braces, no missing semicolons, no ambiguous precedence. An agent emitting Flux is emitting a *binary decision tree* that the translator validates in a single linear pass. The feedback loop is immediate and local: invalid opcode at offset 7, expected 2 more bytes, register 300 out of bounds. The agent can correct and retry without ever invoking a C++ parser.

Moreover, Flux is **target-agnostic** at the bytecode layer. The same sequence of `MOVI`, `ADD`, and `TMUL` instructions can be retargeted from CUDA to a future Vulkan-compute backend or a custom ternary accelerator simply by swapping the importer and lowering passes. The agent does not need to learn a new surface syntax for each platform.

In short: **Flux lets agents compile themselves.**

---

## Further reading

- **Full architecture doc:** [`SuperInstance/cuda-oxide/FLUX_TO_PTX.md`](../../SuperInstance/cuda-oxide/FLUX_TO_PTX.md) — the complete Flux → PTX specification, including NVVM lowering rules, memory model details, and the tensor-core dispatch path.
- **cuda-oxide repository:** The downstream compilation pipeline (`mir-lower`, `llvm-export`, PTX emission).
- **flux-core:** Definitions of the Flux bytecode format, opcodes, and ABI (to be published on crates.io).

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---

*If you are a graduate student reading this: study the `translate_instruction` method in `src/lib.rs`. It is the heart of the crate — a classic single-pass decoder. Notice how each arm validates length, decodes operands, and emits MIR in one straight-line sequence. That simplicity is intentional. Complex compilers are impressive; simple ones are maintainable.*
