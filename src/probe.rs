//! Opcode probing and hardware execution logic.

use std::time::Instant;
use crate::cpu_state::{SnapshotBuffer};
use crate::emitter;
use crate::jit_page::JitPage;

#[cfg(target_os = "macos")]
mod accelerate {
    use crate::crucible::{CblasOrder, CblasTranspose};
    use std::os::raw::{c_float, c_int};

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        pub fn cblas_sgemm(
            order:   CblasOrder,
            trans_a: CblasTranspose,
            trans_b: CblasTranspose,
            m: c_int, n: c_int, k: c_int,
            alpha: c_float,
            a: *const c_float, lda: c_int,
            b: *const c_float, ldb: c_int,
            beta: c_float,
            c: *mut c_float, ldc: c_int,
        );
    }

    /// Wake the SME hardware via a cheap Accelerate call.
    pub fn wake_hardware() {
        let a = [1.0f32];
        let b = [1.0f32];
        let mut c = [0.0f32];
        unsafe {
            cblas_sgemm(
                CblasOrder::RowMajor, CblasTranspose::NoTrans, CblasTranspose::NoTrans,
                1, 1, 1, 1.0, a.as_ptr(), 1, b.as_ptr(), 1, 0.0, c.as_mut_ptr(), 1,
            );
        }
    }
}

pub struct SharedMemory<T> {
    ptr: *mut T,
}

impl<T> SharedMemory<T> {
    pub fn new() -> Self {
        let size = std::mem::size_of::<T>();
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(), size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANON | libc::MAP_SHARED,
                -1, 0,
            )
        };
        assert!(ptr != libc::MAP_FAILED, "mmap failed");
        unsafe { std::ptr::write_bytes(ptr, 0, size); }
        Self { ptr: ptr as *mut T }
    }
    pub fn as_mut_ptr(&self) -> *mut T { self.ptr }
}

impl<T> Drop for SharedMemory<T> {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.ptr as *mut libc::c_void, std::mem::size_of::<T>()); }
    }
}

#[derive(Debug, Clone)]
pub struct ProbeResult {
    pub faulted: bool,
    pub timed_out: bool,
}

impl ProbeResult {
    pub fn status(&self) -> &'static str {
        if self.timed_out { "TIMEOUT" }
        else if self.faulted { "SIGILL/SEGV" }
        else { "ok" }
    }
}

pub struct Probe {
    page: JitPage,
    pub timeout_micros: u64,
}

impl Probe {
    pub fn new() -> Self {
        let page = JitPage::alloc(1024 * 1024).expect("failed to alloc JIT page");
        Probe { page, timeout_micros: 5_000 }
    }

    pub fn run_block_with_overrides(&self, opcodes: &[u32], gpr_overrides: &[(u8, u64)], streaming: bool) -> ProbeResult {
        let buf_pre = SharedMemory::<SnapshotBuffer>::new();
        let buf_post = SharedMemory::<SnapshotBuffer>::new();

        self.page.make_writable();
        let mut off = emitter::emit_prelude(&self.page, buf_pre.as_mut_ptr() as *mut u8, streaming, gpr_overrides, false);
        for &op in opcodes {
            self.page.write_instruction(off, op);
            off += 4;
        }
        emitter::emit_postlude(&self.page, off, buf_post.as_mut_ptr() as *mut u8, buf_pre.as_mut_ptr() as *mut u8, streaming, false);
        self.page.make_executable();

        let mut faulted = false;
        let mut timed_out = false;

        unsafe {
            let pid = libc::fork();
            if pid == 0 {
                libc::signal(libc::SIGILL, libc::SIG_DFL);
                libc::signal(libc::SIGSEGV, libc::SIG_DFL);
                libc::signal(libc::SIGBUS, libc::SIG_DFL);
                #[cfg(target_os = "macos")]
                accelerate::wake_hardware();
                self.page.call_void();
                libc::_exit(0);
            } else if pid > 0 {
                let mut status: libc::c_int = 0;
                let start = Instant::now();
                let timeout = std::time::Duration::from_micros(self.timeout_micros);
                loop {
                    let ret = libc::waitpid(pid, &mut status, libc::WNOHANG);
                    if ret == pid { break; }
                    if start.elapsed() > timeout {
                        libc::kill(pid, libc::SIGKILL);
                        libc::waitpid(pid, &mut status, 0);
                        timed_out = true;
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_micros(100));
                }
                if !timed_out && libc::WIFSIGNALED(status) {
                    faulted = true;
                }
            } else { panic!("fork failed"); }
        }
        ProbeResult { faulted, timed_out }
    }
}
