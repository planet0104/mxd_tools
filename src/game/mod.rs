pub mod action;
pub mod agent;
pub mod bot_harness;
pub mod camera;
pub mod config;
pub mod fitness_core;
pub mod fitness {
    pub use super::fitness_core::*;
}
pub mod headless_vision;
pub mod human_pace;
pub mod input;
pub mod macro_action;
pub mod map;
pub mod movement_gate;
pub mod npc;
pub mod observation;
pub mod rule_bot;
pub mod self_anchor;
pub mod sim;
pub mod sim_observation;
pub mod types;
pub mod view;
pub mod vision;
pub mod vision_worker;
pub mod visual_progress;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

pub use action::{action_to_input, input_label, Action};
pub use agent::{AgentController, VisionWorkerTiming};
pub use bot_harness::{
    assert_yolo_probes, assert_yolo_probes_with, build_parallel_episode_jobs,
    build_parallel_probe_jobs, default_parallel_episode_seeds, default_probe_seeds,
    evaluate_first_platform_report, format_first_platform_preview_done, probe_duration_secs,
    run_all_yolo_probes, run_episode, run_parallel_probe_pool, run_parallel_probe_subprocess,
    run_probe_seeds, run_yolo_probes, BotProbeConfig, EpisodeReport, FirstPlatformReport,
    FirstPlatformTracker, ParallelProbeReport, ProbeDriver, SpawnJumpReport, YoloProbeSet,
    YoloProbeSummary, DEFAULT_PARALLEL_JOBS, FIRST_PLATFORM_PROBE_TICKS,
};
pub use camera::WorldCamera;
pub use config::{GameSimConfig, VisionAnchorConfig, VisionPaceConfig};
pub use fitness::{
    FitnessPreviewDiag, FitnessShapingConfig, TrainingFitness, IDLE_FORFEIT_GRACE_TICKS,
    STAGNATION_TICKS,
};
pub use headless_vision::{default_yolo_model_path, DeferredCaptureVision, HeadlessVisionEnv};
pub use human_pace::HumanPace;
pub use input::InputFrame;
pub use macro_action::{MacroAction, MacroRunner, MACRO_ACTION_COUNT};
pub use map::{ClimbDir, ClimbHint, GameMap, Portal};
pub use movement_gate::{MovementGate, MovementGateCtx};
pub use npc::NpcPlayerState;
pub use observation::{
    inject_proprioception, obs_climb_grab_ready, obs_climb_hint, obs_enemy_in_attack_range,
    obs_farm_band_enemies, obs_floor_ahead, obs_floor_ahead_connected, obs_floor_drop_ahead,
    obs_floor_underfoot, obs_has_drop, obs_has_ladder_or_rope_signal, obs_has_nearby_platform_enemy,
    obs_has_platform_enemy, obs_has_same_level_enemy, obs_jump_allowed,
    obs_nearest_same_level_enemy_px, obs_step_up_dx, VisionObservation,
    OBS_DIM, OBS_DROP_SLOTS, OBS_DROP_START, OBS_ENEMY_SLOTS, OBS_ENEMY_START, OBS_FLOOR_SLOTS,
    OBS_FLOOR_START, OBS_LADDER_SLOTS, OBS_LADDER_START, OBS_PROPRIO, OBS_PROPRIO_START,
    OBS_ROPE_SLOTS, OBS_ROPE_START, OBS_SLOT_DIM, VISION_CONF_THRESH,
};
pub use rule_bot::{visit_key, RuleBot, RuleBotCtx, VisionSenseState};
pub use self_anchor::{apply_anchor_jitter, episode_anchor_offset};
pub use sim::{EngageHint, GameModal, GameSim, GameState, GroundTruth, MobState, PlayerState};
pub use sim_observation::observation_from_sim;
pub use types::{
    DropKind, MobAnim, PlayerAnim, ATTACK_DURATION, DEFAULT_PLAYER_NAME, LOGIC_DT, LOGIC_HZ,
    NAME_TAG_BG_ALPHA, NAME_TAG_FONT_SIZE, NAME_TAG_GAP_BELOW_FEET, NAME_TAG_PAD_X, NAME_TAG_PAD_Y,
    TRAINING_NPC_COUNT, TRAINING_NPC_SPRITES, WINDOW_H, WINDOW_W, WORLD_VIEW_H,
};
pub use vision::{
    assert_training_frame, filter_detections, SimVisionSnapshot, VisionPipeline, VisionStep,
};
pub use visual_progress::{LocationNode, LoopKind, VisualMotionEstimator, VisualProgressMonitor};

pub fn assets_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets")
}

pub fn default_map_path() -> PathBuf {
    assets_root().join("maps/50001/map_50001_platforms.json")
}

pub fn load_default_map() -> Result<GameMap> {
    GameMap::load(&default_map_path())
}

pub fn ui_layout_path() -> PathBuf {
    assets_root().join("ui_game/ui_layout.json")
}

pub fn player_sprite_dir() -> PathBuf {
    assets_root().join("player/默认男新手")
}

pub fn player_sprite_dir_named(name: &str) -> PathBuf {
    assets_root().join("player").join(name)
}

pub fn mob_sprite_dir(mob_id: u32) -> PathBuf {
    assets_root().join(format!("mobs/{}", types::mob_dir_name(mob_id)))
}

pub fn ui_texture_path(name: &str) -> PathBuf {
    assets_root().join("ui_game").join(name)
}

pub fn portal_sprite_dir() -> PathBuf {
    let root = assets_root().join("portals");
    if let Ok(entries) = std::fs::read_dir(&root) {
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
                if name.starts_with("pv_") {
                    return p;
                }
            }
        }
    }
    root.join("pv_可见传送门")
}

pub fn drop_texture_path(kind: types::DropKind) -> PathBuf {
    let root = assets_root().join("drops");
    match kind {
        types::DropKind::Meso => root.join("金币/meso_00.png"),
        types::DropKind::RedPotion => find_potion_icon(&root),
    }
}

fn find_potion_icon(drops: &Path) -> PathBuf {
    if let Ok(entries) = std::fs::read_dir(drops.join("药水")) {
        for e in entries.flatten() {
            let p = e.path();
            if p.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .contains("2000000")
            {
                return p;
            }
        }
    }
    drops.join("药水/2000000_红色药水.png")
}

#[derive(Debug, serde::Deserialize)]
pub struct UiLayoutFile {
    pub window: [u32; 2],
    pub world_height: u32,
    pub widgets: UiWidgets,
    pub dynamic_overlay: DynamicOverlay,
    pub inventory_window: InventoryWindow,
}

#[derive(Debug, serde::Deserialize)]
pub struct UiWidgets {
    pub minimap: UiRect,
    pub panel: UiRect,
    pub keyboard: UiRect,
    #[serde(default)]
    pub quest: Option<UiRect>,
    #[serde(default)]
    pub float_buttons: Option<UiRect>,
}

#[derive(Debug, serde::Deserialize)]
pub struct UiRect {
    pub file: String,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Debug, serde::Deserialize)]
pub struct OverlayRect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Debug, serde::Deserialize)]
pub struct DynamicOverlay {
    pub hp_bar: OverlayRect,
    pub mp_bar: OverlayRect,
    #[serde(default)]
    pub player_name: Option<OverlayRect>,
    #[serde(default)]
    pub hotbar_slots: Vec<HotbarSlot>,
}

#[derive(Debug, serde::Deserialize)]
pub struct HotbarSlot {
    pub slot: u32,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Debug, serde::Deserialize)]
pub struct InventoryWindow {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub cols: u32,
    pub rows: u32,
    pub slot_size: f32,
}

pub fn load_ui_layout() -> Result<UiLayoutFile> {
    let path = ui_layout_path();
    let text = std::fs::read_to_string(&path).with_context(|| format!("读取 {path:?}"))?;
    serde_json::from_str(&text).context("解析 ui_layout.json")
}
