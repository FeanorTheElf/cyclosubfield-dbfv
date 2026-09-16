//! Backing storage on disk for key material that does not fit in RAM.
//!
//! The homomorphic linear transforms ([`crate::bootstrap::homcrt`]) need one Galois key per
//! baby step, and at the large parameter sets the key material is far bigger than the machine's
//! memory: at `field_65536()` with `log2 α = 1720` a single key holds
//! `16 digits · 2 · 31 primes · 65536` coefficients, i.e. roughly half a gigabyte, and the
//! transform needs of the order of a hundred of them.
//!
//! The access pattern is what makes an SSD viable here. The baby step phase of
//! [`crate::bootstrap::homcrt`] reads every key exactly once per dBFV digit, in a fixed order
//! and in contiguous multi-megabyte chunks, and the keys are never written after generation.
//! The giant step phase touches only a single key (repeatedly, so it stays cached) but rereads
//! the `babys` buffer once per giant step — which is why *only* the keys belong on disk and the
//! ciphertext buffers must stay in RAM. Streaming one key takes an order of magnitude less time
//! than the arithmetic that consumes it, so the transfers largely hide behind the computation.
//!
//! A [`KeyStore`] owns one sparse file and hands out page aligned, memory mapped slices of it;
//! [`crate::hal::cpu::CpuTensor::offload()`] moves a tensor into such a slice. Since the pages
//! are file backed and clean after [`KeyStore::store()`] has flushed them, the kernel can
//! reclaim them without writing anything to swap, and fault them back in from the file on the
//! next access.
//!
//! Put the file on the fastest available device — the point of the exercise is that the device
//! can feed the transform faster than it consumes data, which a SATA SSD cannot do. The file is
//! deleted when the store is dropped.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use memmap2::{Advice, MmapMut, MmapOptions};
use tracing::{info, warn};

/// The environment variable naming the directory that [`KeyStore::from_env()`] puts its backing
/// file in.
pub const KEY_STORE_DIR_VAR: &str = "DBFV_KEY_STORE_DIR";

/// Regions are handed out at this alignment, which must be a multiple of the page size for
/// [`MmapOptions::offset()`] to accept it, and which should be large enough that neighbouring
/// regions do not share readahead.
const REGION_ALIGN: u64 = 2 * 1024 * 1024;

fn align_up(x: u64, align: u64) -> u64 { return x.div_ceil(align) * align; }

/// A file that backs memory mapped tensors; see the module documentation. The file is created
/// with a unique name in the given directory, unlinked immediately, and therefore reclaimed by
/// the operating system when the store is dropped (or when the process dies).
pub struct KeyStore {
    file: File,
    /// the next free offset, and the length the file has been grown to
    cursor: Mutex<(u64, u64)>,
    stored: AtomicU64,
    /// kept for diagnostics only; the file itself is already unlinked
    path: PathBuf,
}

impl KeyStore {
    /// Creates a store backed by a file in `dir`.
    pub fn new(dir: impl AsRef<Path>) -> io::Result<Self> {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;
        // the name is unique per store, so that several stores (and several processes) can
        // share a directory without colliding
        let path =
            dir.join(format!("dbfv-keystore-{}-{}.bin", std::process::id(), COUNTER.fetch_add(1, Ordering::Relaxed)));
        let file = OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&path)?;
        // the file is only ever reachable through this handle, so it cannot be left behind
        std::fs::remove_file(&path)?;
        info!("key store backed by {}", path.display());
        return Ok(Self { file, cursor: Mutex::new((0, 0)), stored: AtomicU64::new(0), path });
    }

    /// Creates a store in the directory named by [`KEY_STORE_DIR_VAR`], or returns `None` if
    /// that variable is unset — in which case the caller should keep its key material in RAM.
    pub fn from_env() -> io::Result<Option<Self>> {
        return match std::env::var(KEY_STORE_DIR_VAR) {
            Ok(dir) if !dir.is_empty() => Self::new(dir).map(Some),
            _ => Ok(None),
        };
    }

    /// Copies `data` into the backing file and returns a mapping of it. The mapping is at least
    /// `size_of_val(data)` bytes long and page aligned, hence aligned for `T`.
    pub fn store<T: Copy>(&self, data: &[T]) -> io::Result<MmapMut> {
        let len = std::mem::size_of_val(data) as u64;
        let offset = {
            let mut cursor = self.cursor.lock().unwrap();
            let (next, allocated) = *cursor;
            let offset = align_up(next, REGION_ALIGN);
            let end = offset + len;
            // grow generously so that a run of stores does not call `set_len` every time
            if end > allocated {
                let grown = align_up(end + end / 8, REGION_ALIGN);
                self.file.set_len(grown)?;
                *cursor = (end, grown);
            } else {
                *cursor = (end, allocated);
            }
            offset
        };

        let mut map = unsafe { MmapOptions::new().offset(offset).len(len as usize).map_mut(&self.file)? };
        // `T: Copy` has no padding invariants that a byte-wise copy could violate, and the
        // mapping is exactly `len` bytes long
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, len as usize) };
        map.copy_from_slice(bytes);
        // write the pages back now, so that the kernel may later drop them without any I/O on
        // the critical path, and so that a store does not leave the page cache dirty
        map.flush()?;
        // the consumer reads each region front to back, once
        if let Err(e) = map.advise(Advice::Sequential) {
            warn!("could not advise the key store mapping: {}", e);
        }
        self.stored.fetch_add(len, Ordering::Relaxed);
        return Ok(map);
    }

    /// The number of bytes handed to [`KeyStore::store()`] so far.
    pub fn stored_bytes(&self) -> u64 { return self.stored.load(Ordering::Relaxed); }

    /// The path the backing file was created under. The file is already unlinked, so this is
    /// only useful for diagnostics.
    pub fn path(&self) -> &Path { return &self.path; }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hal::cpu::{CpuBackend, CpuTensor};
    use crate::hal::{Backend, TensorOwned, TensorRef};

    #[test]
    fn test_store_roundtrip() {
        let store = KeyStore::new(std::env::temp_dir()).unwrap();
        let data = (0..4 * 3 * 2 * 5).map(|i| i as i64 * 7 - 11).collect::<Vec<_>>();
        let tensor = CpuTensor::new(data.clone(), [4, 3, 2, 5]);
        let mapped = tensor.offload(&store).unwrap();
        assert!(mapped.is_mapped());
        assert_eq!(mapped.data(), &data[..]);
        assert_eq!([4, 3, 2, 5], TensorRef::shape(mapped.as_ref()));
        // offloading twice is a no-op, and the store accounts for what it holds
        let mapped = mapped.offload(&store).unwrap();
        assert_eq!(mapped.data(), &data[..]);
        assert_eq!(std::mem::size_of_val(&data[..]) as u64, store.stored_bytes());
    }

    #[test]
    fn test_regions_are_independent_and_writable() {
        let store = KeyStore::new(std::env::temp_dir()).unwrap();
        let a = CpuTensor::new(vec![1i64; 1024], [1, 1, 1, 1024]).offload(&store).unwrap();
        let mut b = CpuTensor::new(vec![2i64; 1024], [1, 1, 1, 1024]).offload(&store).unwrap();
        // a mapped tensor stays writable through the ordinary view API, and the regions are
        // aligned and disjoint, so writing one does not disturb the other
        CpuBackend.hal_assign(b.as_mut(), CpuTensor::new(vec![3i64; 1024], [1, 1, 1, 1024]).as_ref());
        assert!(b.data().iter().all(|x| *x == 3));
        assert!(a.data().iter().all(|x| *x == 1));
        // cloning a mapped tensor detaches it from the store
        let c = b.clone();
        assert!(!c.is_mapped());
        assert_eq!(c.data(), b.data());
    }
}
