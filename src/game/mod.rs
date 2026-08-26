pub mod action;
pub mod camera;
pub mod config;
pub mod fitness;
pub mod input;
pub mod map;
pub mod npc;
pub mod observation;
pub mod sim;
pub mod types;
pub mod view;
pub mod vision;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

pub use action::{action_to_input, actions_from_bits, Action};
pub use camera::WorldCamera;
pub use config::{GameSimConfig, TrainingPaceConfig};
pub use fitness::{FitnessShapingConfig, TrainingFitness};
pub use input::InputFrame;
pub use map::{GameMap, Portal};
pub use npc::NpcPlayerState;
pub use observation::{VisionObservation, NEAT_CONF_THRESH, OBS_DIM};
pub use sim::{GameModal, GameSim, GameState, GroundTruth, MobState, PlayerState};
pub use types::{
    DropKind, MobAnim, PlayerAnim, ATTACK_DURATION, DEFAULT_PLAYER_NAME, LOGIC_DT, LOGIC_HZ,
    NAME_TAG_BG_ALPHA, NAME_TAG_FONT_SIZE, NAME_TAG_GAP_BELOW_FEET, NAME_TAG_PAD_X, NAME_TAG_PAD_Y,
    TRAINING_NPC_SPRITES, WINDOW_H, WINDOW_W, WORLD_VIEW_H,
};
pub use vision::{assert_training_frame, filter_detections, VisionPipeline, VisionStep};

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
            if p.file_name().and_then(|s| s.to_str()).unwrap_or("").contains("2000000") {
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
