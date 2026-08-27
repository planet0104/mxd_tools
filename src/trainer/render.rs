//! 训练/预览共用的离屏渲染读回（须在 macroquad 主线程调用）。

use image::RgbImage;
use macroquad::prelude::RenderTarget;

use crate::game::view::{self, GameViewAssets};
use crate::game::GameSim;
use crate::trainer::profile::RenderStepTiming;

pub const GL_WARMUP_FRAMES: usize = 2;

pub async fn capture_render_rgb(
    assets: &GameViewAssets,
    sim: &GameSim,
    rt: &RenderTarget,
) -> RgbImage {
    for _ in 0..GL_WARMUP_FRAMES {
        view::draw_to_render_target(assets, sim, rt);
        macroquad::prelude::next_frame().await;
    }
    view::draw_to_render_target(assets, sim, rt);
    macroquad::prelude::next_frame().await;
    view::render_target_to_rgb(rt)
}

/// NEAT 训练 eval：离屏 draw + 读回，不 `next_frame`（headless 全速，不等 vsync）。
pub fn capture_render_rgb_headless(
    assets: &GameViewAssets,
    sim: &GameSim,
    rt: &RenderTarget,
) -> RgbImage {
    capture_render_rgb_headless_timed(assets, sim, rt).0
}

/// 带分步计时的 headless 截帧（`--profile`）。
pub fn capture_render_rgb_headless_timed(
    assets: &GameViewAssets,
    sim: &GameSim,
    rt: &RenderTarget,
) -> (RgbImage, RenderStepTiming) {
    use std::time::Instant;
    let t0 = Instant::now();
    view::draw_to_render_target(assets, sim, rt);
    let draw_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let t1 = Instant::now();
    let rgb = view::render_target_to_rgb(rt);
    let readback_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    (
        rgb,
        RenderStepTiming {
            draw_ms,
            present_ms: 0.0,
            readback_ms,
            total_ms,
        },
    )
}

/// 兼容旧调用；等同 [`capture_render_rgb_headless`]。
pub async fn capture_render_rgb_fast(
    assets: &GameViewAssets,
    sim: &GameSim,
    rt: &RenderTarget,
) -> RgbImage {
    capture_render_rgb_headless(assets, sim, rt)
}

/// 兼容旧调用；等同 [`capture_render_rgb_headless_timed`]。
pub async fn capture_render_rgb_timed(
    assets: &GameViewAssets,
    sim: &GameSim,
    rt: &RenderTarget,
) -> (RgbImage, RenderStepTiming) {
    capture_render_rgb_headless_timed(assets, sim, rt)
}

/// 窗口模式：将当前 sim 状态绘制到屏幕（训练 `--visible` 调试用）。
pub async fn present_training_frame(
    assets: &GameViewAssets,
    sim: &GameSim,
    hud_line: Option<&str>,
) {
    use macroquad::prelude::*;
    clear_background(Color::new(0.05, 0.05, 0.08, 1.0));
    view::begin_logical_viewport();
    view::draw_content(assets, sim);
    set_default_camera();
    if let Some(line) = hud_line {
        draw_text(line, 12.0, 22.0, 18.0, WHITE);
    }
    next_frame().await;
}
