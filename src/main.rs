//! Gate runner — tiny dispatcher for active research gates.
//!
//! All shared infrastructure lives in the library crate (`sme_jit_core`).
//! Historical gates have been retired; only the current research front
//! (Gate 26: predicated memory) is wired up here.

use sme_jit_core::emitter::build_gate26_page;
use sme_jit_core::signal_handler::install_sigill_handler;

/// Current active research gate.
fn gate_26() {
    println!("══════════════════════════════════════════════════════════════");
    println!("  Gate 26: Predicated Memory & Generation — Edge Bounds");
    println!("══════════════════════════════════════════════════════════════");
    println!();

    let limit = 20_usize;
    let guard_val = -1.0f32;

    let src: Vec<f32> = (0..limit).map(|i| i as f32).collect();
    let mut dst = vec![guard_val; 32];

    println!("  [1] Building predicated copy kernel (limit={})...", limit);
    let page = build_gate26_page(limit).expect("Failed to build gate 26 page");

    println!("  [2] Executing copy...");
    // SAFETY: page contains valid SVE kernel, src and dst pointers are valid.
    unsafe {
        page.call_with_args(src.as_ptr() as u64, dst.as_mut_ptr() as u64);
    }

    println!("  [3] Verifying results...");
    let mut errors = 0;
    for i in 0..limit {
        if dst[i] != src[i] {
            println!("      [✗] Mismatch at index {}: expected {}, got {}", i, src[i], dst[i]);
            errors += 1;
        }
    }

    let mut guard_violations = 0;
    for i in limit..32 {
        if dst[i] != guard_val {
            println!("      [✗] Guard violation at index {}: expected {}, got {}", i, guard_val, dst[i]);
            guard_violations += 1;
        }
    }

    if errors == 0 && guard_violations == 0 {
        println!("  ████████████████████████████████████████████████████████████");
        println!("  █                                                          █");
        println!("  █   🛡️  GATE 26 — PREDICATED MEMORY SUCCESS  🛡️           █");
        println!("  █                                                          █");
        println!("  █   Copied: {}/20 elements correctly                    █", limit);
        println!("  █   Guard:  12/12 elements untouched                       █");
        println!("  █                                                          █");
        println!("  █   SVE WHILELT generated correct masks for 20 elements.   █");
        println!("  █                                                          █");
        println!("  ████████████████████████████████████████████████████████████");
    } else {
        println!("  [!] Gate 26 FAILED: {} errors, {} guard violations", errors, guard_violations);
    }

    println!();
    println!("✓ gate 26 complete\n");
}

fn main() {
    install_sigill_handler();

    let args: Vec<String> = std::env::args().collect();
    if args.contains(&"gate26".to_string()) {
        gate_26();
    } else if args.contains(&"all".to_string()) {
        println!("Historical gates are disabled by default. Run specifically (e.g., cargo run -- gate26).");
    } else {
        println!("sme-jit-core gate runner");
        println!("Usage: cargo run --release -- [gate26]");
        println!();
        println!("Running latest research (Gate 26)...");
        gate_26();
    }
}
