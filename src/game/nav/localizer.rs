use super::super::map::GameMap;
use super::map_graph::MapGraph;
use super::types::{LocState, PlatformNodeId};

#[derive(Debug, Clone, Default)]
pub struct Localizer {
    pub state: LocState,
    initialized: bool,
}

impl Localizer {
    pub fn reset(&mut self, x: f32, y: f32, node_id: PlatformNodeId) {
        self.state = LocState {
            world_x: x,
            world_y: y,
            confidence: 255,
            node_id,
            on_ground: true,
            climbing: false,
        };
        self.initialized = true;
    }

    /// 用视觉里程计更新位置；低置信度时冻结 node_id。
    pub fn tick(
        &mut self,
        map: &GameMap,
        graph: &MapGraph,
        est_x: f32,
        est_y: f32,
        confidence: u8,
        on_ground: bool,
        climbing: bool,
        min_conf: u8,
    ) {
        if !self.initialized {
            self.state.world_x = est_x;
            self.state.world_y = est_y;
            self.state.confidence = confidence;
            self.initialized = true;
        } else {
            self.state.world_x = est_x;
            self.state.world_y = est_y;
            self.state.confidence = confidence;
        }
        self.state.on_ground = on_ground;
        self.state.climbing = climbing;
        if confidence >= min_conf {
            if let Some(id) = graph.node_at(map, self.state.world_x, self.state.world_y) {
                self.state.node_id = id;
            } else if on_ground && !climbing {
                // est y 漂到台面下：仅落地时用几何兜底。爬绳中途勿吸到上下层同 x 台。
                if let Some(id) =
                    graph.node_at_by_xy(self.state.world_x, self.state.world_y, 120.0)
                {
                    self.state.node_id = id;
                }
            }
        }
    }

    pub fn low_confidence(&self, min_conf: u8) -> bool {
        self.state.confidence < min_conf
    }
}
