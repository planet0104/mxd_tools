//! 真实 `game_preview` 首平台探针（子进程 + stdout 日志断言）。
//!
//! ```powershell
//! cargo test --release --test game_preview_first_platform -- --nocapture
//! ```

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn game_preview_bin() -> PathBuf {
    manifest_dir().join("target/release/game_preview.exe")
}

fn build_game_preview() {
    let status = Command::new("cargo")
        .current_dir(manifest_dir())
        .args(["build", "--release", "--bin", "game_preview"])
        .status()
        .expect("cargo build game_preview");
    assert!(status.success(), "cargo build --release --bin game_preview failed");
}

#[test]
fn game_preview_leaves_first_platform() {
    build_game_preview();
    let bin = game_preview_bin();
    assert!(bin.exists(), "missing {}", bin.display());

    let mut child = Command::new(&bin)
        .args(["--probe", "first_platform", "--quiet"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn game_preview");

    let stdout = child.stdout.take().expect("stdout");
    let stderr = child.stderr.take().expect("stderr");

    let stdout_lines: std::sync::Arc<std::sync::Mutex<Vec<String>>> =
        std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let stderr_lines: std::sync::Arc<std::sync::Mutex<Vec<String>>> =
        std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));

    let out_cap = stdout_lines.clone();
    let out_thread = std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            eprintln!("[game_preview stdout] {line}");
            out_cap.lock().unwrap().push(line);
        }
    });

    let err_cap = stderr_lines.clone();
    let err_thread = std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            eprintln!("[game_preview stderr] {line}");
            err_cap.lock().unwrap().push(line);
        }
    });

    let deadline = Instant::now() + Duration::from_secs(180);
    let status = loop {
        if let Some(status) = child.try_wait().expect("try_wait") {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            panic!("game_preview --probe first_platform timed out after 180s");
        }
        std::thread::sleep(Duration::from_millis(200));
    };

    out_thread.join().unwrap();
    err_thread.join().unwrap();

    let stdout_all = stdout_lines.lock().unwrap().join("\n");
    assert!(
        status.success(),
        "game_preview exited with {status}; stdout:\n{stdout_all}"
    );
    assert!(
        stdout_all.contains("PREVIEW_DONE verdict=PASS probe=first_platform"),
        "expected PASS line in stdout, got:\n{stdout_all}"
    );
}
