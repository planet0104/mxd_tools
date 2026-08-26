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

/// NEAT 训练 eval：单次绘制 + 一次 present（headless swap_interval=0），不阻塞等 YOLO。
pub async fn capture_render_rgb_fast(
    assets: &GameViewAssets,
    sim: &GameSim,
    rt: &RenderTarget,
) -> RgbImage {
    capture_render_rgb_timed(assets, sim, rt).await.0
}

/// 带分步计时的 fast 截帧（`--profile`）。
pub async fn capture_render_rgb_timed(
    assets: &GameViewAssets,
    sim: &GameSim,
    rt: &RenderTarget,
) -> (RgbImage, RenderStepTiming) {
    use std::time::Instant;
    let t0 = Instant::now();
    view::draw_to_render_target(assets, sim, rt);
    let draw_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let t1 = Instant::now();
    macroquad::prelude::next_frame().await;
    let present_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let t2 = Instant::now();
    let rgb = view::render_target_to_rgb(rt);
    let readback_ms = t2.elapsed().as_secs_f64() * 1000.0;
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    (
        rgb,
        RenderStepTiming {
            draw_ms,
            present_ms,
            readback_ms,
            total_ms,
        },
    )
}
