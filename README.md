# Bootstrapping BFV over cyclotomic subfields — reference implementation

<!-- TODO: replace with the final paper title before submission -->
Artifact accompanying *\<paper title to be determined\>*.

This repository contains an implementation of BFV and of its digit-decomposed variant dBFV over
**cyclotomic subfields** — abelian, non-Galois-closed number fields `L` sitting inside a
cyclotomic field `M` — together with the benchmarks reported in the paper. Working in a subfield
rather than in the full cyclotomic field is what buys the slot counts and the amortized costs in
the tables: the ring of integers `O_L` has no power-of-two conductor to exploit, so the
implementation replaces the usual negacyclic NTT with transforms between a *period basis* and a
*CRT basis* of `O_L/q`, and carries out bootstrapping with homomorphic CRT transforms in that
basis.

## Repository structure

```
.
├── Cargo.toml                  workspace root: members, shared dep versions, release profile
├── rust-toolchain.toml         pinned nightly toolchain
├── rustfmt.toml
├── cyclosubfield-dbfv/         the implementation (library crate)
├── cyclosubfield-benchmarks/   the benchmarks (binary crate)
│   ├── benchmarks.md           the measurements reported in the paper
│   └── run-benchmark.sh        runs the full suite with the key store on an SSD
└── handwritten-reference/      the original hand-written prototype (not built; see below)
```

The two crates form a single Cargo workspace, so one `cargo build` covers both and a single
`Cargo.lock` pins every dependency.

### `cyclosubfield-dbfv/` — the implementation

| Module      | Contents                                                                            |
| :---------- | :---------------------------------------------------------------------------------- |
| `hal`       | the tensor-based hardware abstraction layer the whole crate is written against, with a CPU backend (`hal::cpu`), the NTTs (`hal::ntt`) and Montgomery arithmetic (`hal::mont`) |
| `field`     | hard-coded descriptors for the supported number fields, each recording the conductor `m`, the Galois generators and the operator-norm estimates |
| `ring`      | the RNS ring `R_q = O_L/q` and the period-basis ↔ CRT-basis transforms               |
| `bfv`       | the BFV scheme, including dynamic scaling (mod-switching to smaller RNS bases)       |
| `dbfv`      | the digit-decomposed dBFV scheme                                                     |
| `bootstrap` | bootstrapping: homomorphic CRT/linear transforms (`bootstrap::homcrt`), digit extraction (`bootstrap::digitex`) and Galois-ring arithmetic (`bootstrap::galoisring`) |
| `keystore`  | disk backing for Galois keys, for the parameter sets whose key material exceeds RAM  |

### `cyclosubfield-benchmarks/` — the benchmarks

A single binary that measures addition, multiplication, bootstrapping and the
coefficients-to-slots transform, and prints the result tables in Markdown:

| Module               | Scheme benchmarked                                                       |
| :------------------- | :----------------------------------------------------------------------- |
| `cyclotomic_subrings`| cyclotomic-subring BFV and dBFV, i.e. this paper's construction, via `cyclosubfield-dbfv` |
| `cyclotomic_bfv`     | BFV over a full (composite-conductor) cyclotomic field, as the baseline, via [`fheanor`](https://crates.io/crates/fheanor) |
| `tfhe`               | TFHE, as a second baseline, via [`tfhe-rs`](https://crates.io/crates/tfhe) |

### `handwritten-reference/` — the original prototype

The hand-written code the implementation is derived from (see *Provenance* below). It is kept in
the repository as a reference for the mathematics, but should not be considered an artifact.
It may contain implementation bugs, inefficient code and does not compile as given. We only provide
it, since it was used as reference when generating the actual implementation using Claude Code.

## Requirements

- **Toolchain.** A nightly Rust toolchain; the exact one is pinned in `rust-toolchain.toml`
  (`nightly-2026-02-05`) and `rustup` will fetch it automatically. Nightly is needed for the
  `mapped_lock_guards` feature and for the `profile-rustflags` Cargo feature.
- **CPU.** The release profile compiles with `-Ctarget-cpu=native` for maximum efficiency.
  Binaries built this way are not portable between machines.
- **Memory.** The unit tests and the smaller parameter sets run on an ordinary workstation. The
  large parameter sets do not: their Galois key material runs to tens of gigabytes, more than
  fits in RAM. See *Running the benchmarks* below.
- **Disk.** For those large parameter sets, roughly 100 GB of free space on a **fast NVMe** device
  for the key store. The access pattern (each key read once per dBFV digit, in a fixed order, in
  contiguous multi-megabyte chunks) is what makes streaming from disk viable; a SATA SSD cannot
  feed the transform fast enough. See the module documentation of `cyclosubfield_dbfv::keystore`.

## Building

```sh
cargo build --release
```

Use `--release` for anything but the unit tests — the debug profile is orders of magnitude slower,
and all reported numbers were taken with the release profile defined in the root `Cargo.toml`
(`lto = "fat"`, `codegen-units = 1`, `-Ctarget-cpu=native`).

## Running the tests

```sh
cargo test -p cyclosubfield-dbfv
```

87 tests, covering the ring arithmetic, the basis transforms, the BFV and dBFV operations, digit
extraction and bootstrapping at small parameter sets. Two further tests exercise full-size
bootstrapping over `field_36864()` and `field_49152()`; they take a very long time and are
`#[ignore]`d by default:

```sh
cargo test -p cyclosubfield-dbfv --release -- --ignored --nocapture
```

## Running the benchmarks

For the small and medium parameter sets, run the binary directly:

```sh
cargo run --release -p cyclosubfield-benchmarks
```

It prints each measurement as it is taken and, at the end, the Markdown tables that
`cyclosubfield-benchmarks/benchmarks.md` records. **The full suite takes many hours** — a single
dBFV bootstrapping at the largest parameter set is measured in tens of minutes — so expect to
leave it running, and note that the numbers in `benchmarks.md` were taken on one machine and will
not reproduce exactly on another.

### Large parameter sets: the key store and the memory cgroup

At the large parameter sets the Galois keys do not fit in RAM, and the implementation memory-maps
them out of a file instead. Setting `DBFV_KEY_STORE_DIR` to a directory on a fast NVMe device
turns this on:

```sh
DBFV_KEY_STORE_DIR=/mnt/nvme/dbfv cargo run --release -p cyclosubfield-benchmarks
```

`cyclosubfield-benchmarks/run-benchmark.sh` is the wrapper actually used to take the reported
numbers. On top of pointing the key store at an NVMe device, it runs the benchmark inside a
`systemd` scope with a memory limit:

```sh
./cyclosubfield-benchmarks/run-benchmark.sh
```

The cgroup limit is not a nicety. The key store's pages are clean and file-backed, so without a
limit the kernel's global reclaim prefers swapping out *other* processes' anonymous pages over
dropping them, and the rest of the machine grinds to a halt. Inside a cgroup the pressure stays
confined to the benchmark, which re-reads its own pages from the SSD. `MemorySwapMax=0` keeps the
benchmark's own resident working set — the transform's baby-step buffer — out of swap, where it
would be catastrophic. Override the defaults with the `DBFV_KEY_STORE_DIR`, `MEM_HIGH` and
`MEM_MAX` environment variables.

### The circuit cache

Generating the linear transforms, digit-extraction circuits and hypercube structures for a
parameter set is itself expensive, so they are cached as JSON under
`cyclosubfield-benchmarks/cache/` and reused across runs. The directory is git-ignored and is
recreated on demand; deleting it costs setup time on the next run but nothing else.

## Provenance: which parts are AI-generated

The paper's implementation was produced with substantial help from
[Claude Code](https://claude.com/claude-code), using models Opus 4.8, Opus 5 and Fable 5, and the split is as follows.

- **`handwritten-reference/` — written by hand.** The original prototype: the mathematics, the
  algorithms and the field descriptors were all worked out and written by the authors. This is the
  human-authored artifact the rest of the implementation derives from.

- **`cyclosubfield-dbfv/` — generated by Claude Code.** The implementation in this crate was generated by
  Claude Code from the hand-written prototype above, under the authors' direction and review. The
  restructuring around the `hal` tensor abstraction, the RNS handling, the disk-backed key store
  and the test suite are the product of that process. The mathematical content is the prototype's;
  the code that realizes it here is not hand-written.

- **`cyclosubfield-benchmarks/` — written by hand, with occasional AI assistance.** The benchmark
  harness, the choice of parameter sets and the baseline comparisons were written by the authors.
  A few passages were touched up or filled in with AI assistance, so we do not claim that every
  line is hand-written.

- **This README** was drafted by Claude Code from the repository contents and edited by the
  authors.

Claude Code was used as an implementation tool throughout, not as a source of the mathematics: the
constructions, parameter choices and security arguments in the paper are the authors' own, and the
generated code was reviewed and tested against the hand-written reference.
