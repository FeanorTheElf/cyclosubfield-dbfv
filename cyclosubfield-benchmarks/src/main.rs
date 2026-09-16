//! Benchmarks for cyclotomic-subring BFV/dBFV (as implemented in [`cyclosubfield_dbfv`]), against
//! cyclotomic BFV (via `fheanor`) and TFHE (via `tfhe-rs`). See the repository README.
// rings are named after the literature (`P` plaintext, `C` ciphertext, `H` hypercube), as in
// `cyclosubfield_dbfv`
#![allow(non_snake_case)]

use std::collections::BTreeMap;

use feanor_tracing::DelayedLogger;

use crate::cyclotomic_bfv::benchmark_cyclotomic_bfv;
use crate::cyclotomic_subrings::{benchmark_cyclotomic_subring_bfv, benchmark_cyclotomic_subring_dbfv};
use crate::tfhe::benchmark_tfhe;

mod cyclotomic_bfv;
mod cyclotomic_subrings;
mod tfhe;

/// Where the generated linear transforms, digit-extraction circuits and hypercube structures are
/// cached between runs. Anchored at the crate directory rather than the working directory, so
/// that the cache is shared no matter where in the workspace `cargo run` was invoked from.
pub const CACHE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/cache");

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
enum DType {
    Uint(usize),
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
enum Benchmark {
    Bootstrap,
    Add,
    Mul,
    LinTransform,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
enum Scheme {
    CycloSubringBFV,
    CycloRingBFV,
    CycloSubringDBFV,
    TFHE,
    /// reserved for a radix-CKKS comparison, which is not part of this artifact
    #[allow(dead_code)]
    RadixCKKS,
}

/// The noise measured during one benchmark. Whether the decryption gave the expected result
/// and the noise budget are tracked independently: the noise may overflow during a
/// lower-precision intermediate computation (as it happens inside bootstrapping), while the
/// resulting ciphertext still has a significant budget left. Hence the budget is reported
/// also for measurements whose decryption failed.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct NoiseInfo {
    /// whether decrypting the result gave the expected value; if `false`, the noise
    /// overflowed at some point during the operation
    decryption_correct: bool,
    budget: NoiseBudget,
}

/// Noise budgets are signed, since an operation may well leave a negative budget - this
/// means the noise has grown past the point where decryption is guaranteed to work.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
enum NoiseBudget {
    /// bits of noise budget left on the resulting ciphertext
    Left(i64),
    /// bits of noise budget consumed by the operation
    Consumed(i64),
    NotApplicable,
}

impl NoiseInfo {
    fn new(decryption_correct: bool, budget: NoiseBudget) -> Self { NoiseInfo { decryption_correct, budget } }

    fn not_applicable() -> Self { NoiseInfo { decryption_correct: true, budget: NoiseBudget::NotApplicable } }
}

impl std::fmt::Debug for NoiseInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.budget {
            NoiseBudget::Left(bits) => write!(f, "{} bits left", bits)?,
            NoiseBudget::Consumed(bits) => write!(f, "{} bits consumed", bits)?,
            NoiseBudget::NotApplicable => write!(f, "n/a")?,
        }
        if !self.decryption_correct {
            write!(f, " (decryption failed)")?;
        }
        return Ok(());
    }
}

impl DType {
    fn name(&self) -> String {
        match self {
            DType::Uint(1) => format!("Bit"),
            DType::Uint(x) => format!("Uint{}", x),
        }
    }
}

impl Scheme {
    fn name(&self) -> &'static str {
        match self {
            Scheme::CycloSubringBFV => "Cyclotomic-Subring BFV",
            Scheme::CycloRingBFV => "Cyclotomic BFV",
            Scheme::CycloSubringDBFV => "Cyclotomic-Subring DBFV",
            Scheme::TFHE => "TFHE",
            Scheme::RadixCKKS => "Radix CKKS",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Measurement {
    time_ms: u64,
    slots: usize,
    noise: NoiseInfo,
}

/// Identifies one row of the result tables; the same scheme is benchmarked with
/// multiple parameter sets, hence the parameter string is part of the key.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct RowKey {
    dtype: DType,
    scheme: Scheme,
    params: String,
}

pub struct Reporter {
    results: BTreeMap<RowKey, BTreeMap<Benchmark, Measurement>>,
}

impl Reporter {
    fn new() -> Self { Reporter { results: BTreeMap::new() } }

    fn report(
        &mut self,
        benchmark: Benchmark,
        dtype: DType,
        scheme: Scheme,
        params: &str,
        time_ms: u64,
        slots: usize,
        noise_budget: NoiseInfo,
    ) {
        println!(
            "{:?} {:?}, {:?}, {}, time: {} ms, slots: {}, noise: {:?}",
            benchmark, dtype, scheme, params, time_ms, slots, noise_budget
        );
        let key = RowKey { dtype, scheme, params: params.to_owned() };
        self.results.entry(key).or_default().insert(benchmark, Measurement { time_ms, slots, noise: noise_budget });
    }

    /// Prints one markdown table per [`DType`], with a row for each
    /// (scheme, parameter set) that was benchmarked with this dtype, followed by
    /// the separate table of [`Benchmark::LinTransform`] (see
    /// [`Reporter::finish_lin_transform()`]).
    fn finish(&self) {
        let dtypes = self.results.keys().map(|key| key.dtype).collect::<std::collections::BTreeSet<_>>();
        let mut any_overflow = false;
        for dtype in dtypes {
            println!();
            println!("## {}", dtype.name());
            println!();
            let header = [
                "Scheme",
                "Parameters",
                "Slots",
                "Add",
                "Mul",
                "Bootstrap",
                "Mul/Bootstrap",
                "Amortized Add",
                "Amortized Mul",
            ];
            let mut rows = Vec::new();
            for (key, measurements) in self.results.iter().filter(|(key, _)| key.dtype == dtype) {
                let add = measurements.get(&Benchmark::Add);
                let mul = measurements.get(&Benchmark::Mul);
                let bootstrap = measurements.get(&Benchmark::Bootstrap);
                // a parameter set for which only the linear transform was benchmarked has no
                // row here; it appears in the linear-transform table instead
                if add.is_none() && mul.is_none() && bootstrap.is_none() {
                    continue;
                }
                any_overflow |= [add, mul, bootstrap].into_iter().flatten().any(|m| !m.noise.decryption_correct);
                let slots = mul.or(add).or(bootstrap).map(|m| m.slots).unwrap_or(1).max(1);

                // number of multiplications that fit between two bootstrappings; if the
                // multiplication already includes a bootstrapping (as for TFHE), there is
                // no additional bootstrapping to amortize
                let muls_per_bootstrap = match (mul.map(|m| m.noise.budget), bootstrap.map(|m| m.noise.budget)) {
                    (Some(NoiseBudget::NotApplicable), _) | (_, None) => Some(f64::INFINITY),
                    (Some(NoiseBudget::Consumed(consumed)), _) if consumed <= 0 => Some(f64::INFINITY),
                    (Some(NoiseBudget::Consumed(consumed)), Some(NoiseBudget::Left(left))) => {
                        Some((left / consumed) as f64)
                    }
                    _ => None,
                };

                let amortized_add = add.map(|m| m.time_ms as f64 / slots as f64);
                let amortized_mul = mul.zip(muls_per_bootstrap).map(|(m, muls)| {
                    let bootstrap_share = if muls.is_finite() && muls > 0. {
                        bootstrap.map_or(0., |b| b.time_ms as f64) / muls
                    } else if muls <= 0. {
                        f64::INFINITY
                    } else {
                        0.
                    };
                    (m.time_ms as f64 + bootstrap_share) / slots as f64
                });

                rows.push(vec![
                    key.scheme.name().to_owned(),
                    key.params.clone(),
                    format!("{}", slots),
                    fmt_measurement(add),
                    fmt_measurement(mul),
                    fmt_measurement(bootstrap),
                    match muls_per_bootstrap {
                        None => "?".to_owned(),
                        Some(muls) if muls.is_infinite() => "∞".to_owned(),
                        Some(muls) => format!("{}", muls),
                    },
                    amortized_add.map_or("—".to_owned(), fmt_duration),
                    amortized_mul.map_or("—".to_owned(), fmt_duration),
                ]);
            }
            print_markdown_table(&header, &rows);
        }
        any_overflow |= self.finish_lin_transform();
        if any_overflow {
            println!();
            println!(
                "`(!)` = decryption failed, i.e. the noise overflowed; a noise budget given for \
                such a measurement still refers to the resulting ciphertext, and may well be \
                positive if the overflow happened during a lower-precision intermediate \
                computation"
            );
        }
    }

    /// Prints the table of the linear-transform (coefficients-to-slots) benchmark, and returns
    /// whether any of its measurements overflowed. Unlike add/mul/bootstrap, the linear
    /// transform is not an operation one builds circuits from — it is a building block of
    /// bootstrapping, and is measured to show where the bootstrapping time goes — so it gets a
    /// table of its own instead of a column in the tables above. The plaintext modulus is a
    /// column here rather than the table key, since the same transform is benchmarked for very
    /// few dtypes.
    fn finish_lin_transform(&self) -> bool {
        let measurements = self
            .results
            .iter()
            .filter_map(|(key, measurements)| {
                measurements.get(&Benchmark::LinTransform).map(|measurement| (key, measurement))
            })
            .collect::<Vec<_>>();
        if measurements.is_empty() {
            return false;
        }
        println!();
        println!("## Linear Transform (Coefficients-to-Slots)");
        println!();
        let header = ["Scheme", "Parameters", "Type", "Slots", "Time", "Amortized", "Noise Growth"];
        let mut any_overflow = false;
        let mut rows = Vec::new();
        for (key, measurement) in measurements {
            any_overflow |= !measurement.noise.decryption_correct;
            let slots = measurement.slots.max(1);
            rows.push(vec![
                key.scheme.name().to_owned(),
                key.params.clone(),
                key.dtype.name(),
                format!("{}", slots),
                fmt_measurement(Some(measurement)),
                fmt_duration(measurement.time_ms as f64 / slots as f64),
                match measurement.noise.budget {
                    NoiseBudget::Consumed(bits) => format!("{} bits", bits),
                    NoiseBudget::Left(bits) => format!("{} bits left", bits),
                    NoiseBudget::NotApplicable => "—".to_owned(),
                },
            ]);
        }
        print_markdown_table(&header, &rows);
        return any_overflow;
    }
}

fn fmt_measurement(measurement: Option<&Measurement>) -> String {
    match measurement {
        None => "—".to_owned(),
        Some(measurement) => {
            let time = fmt_duration(measurement.time_ms as f64);
            if !measurement.noise.decryption_correct { format!("{} (!)", time) } else { time }
        }
    }
}

fn fmt_duration(time_ms: f64) -> String {
    if !time_ms.is_finite() {
        return "—".to_owned();
    }
    if time_ms >= 60000. {
        format!("{:.1} min", time_ms / 60000.)
    } else if time_ms >= 1000. {
        format!("{:.2} s", time_ms / 1000.)
    } else if time_ms >= 1. {
        format!("{:.1} ms", time_ms)
    } else if time_ms >= 0.001 {
        format!("{:.1} µs", time_ms * 1000.)
    } else {
        format!("{:.0} ns", time_ms * 1000000.)
    }
}

/// The first two columns are left-aligned, all others right-aligned.
fn print_markdown_table(header: &[&str], rows: &[Vec<String>]) {
    let mut widths = header.iter().map(|entry| entry.chars().count()).collect::<Vec<_>>();
    for row in rows {
        for (i, cell) in row.iter().enumerate() {
            widths[i] = widths[i].max(cell.chars().count());
        }
    }
    let pad = |cell: &str, i: usize| {
        let fill = " ".repeat(widths[i] - cell.chars().count());
        if i < 2 { format!("{}{}", cell, fill) } else { format!("{}{}", fill, cell) }
    };
    let render = |cells: Vec<String>| println!("| {} |", cells.join(" | "));

    render(header.iter().enumerate().map(|(i, entry)| pad(entry, i)).collect());
    render(
        widths
            .iter()
            .enumerate()
            .map(
                |(i, width)| {
                    if i < 2 { format!(":{}", "-".repeat(width - 1)) } else { format!("{}:", "-".repeat(width - 1)) }
                },
            )
            .collect(),
    );
    for row in rows {
        render(row.iter().enumerate().map(|(i, cell)| pad(cell, i)).collect());
    }
}

fn main() {
    DelayedLogger::init_test();
    let mut reporter = Reporter::new();
    benchmark_cyclotomic_bfv(&mut reporter);
    benchmark_cyclotomic_subring_bfv(&mut reporter);
    benchmark_cyclotomic_subring_dbfv(&mut reporter);
    benchmark_tfhe(&mut reporter);
    reporter.finish();
}
