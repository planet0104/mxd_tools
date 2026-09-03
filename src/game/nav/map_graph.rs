use std::collections::{HashMap, HashSet};

use super::super::map::{GameMap, WalkAhead};
use super::types::{EdgeKind, GraphEdge, PlatformNode, PlatformNodeId};

#[derive(Debug, Clone)]
pub struct MapGraph {
    pub nodes: HashMap<PlatformNodeId, PlatformNode>,
    pub edges: Vec<GraphEdge>,
    adj: HashMap<PlatformNodeId, Vec<usize>>,
}

impl MapGraph {
    pub fn build(map: &GameMap) -> Self {
        let mut nodes = HashMap::new();
        for p in &map.platforms {
            let x_min = p.x1.min(p.x2);
            let x_max = p.x1.max(p.x2);
            if x_max - x_min < 4.0 {
                continue;
            }
            let y = (p.y1 + p.y2) * 0.5;
            nodes.insert(
                p.id,
                PlatformNode {
                    id: p.id,
                    x_min,
                    x_max,
                    y,
                    layer: p.layer,
                    group: p.group,
                    prev: p.prev,
                    next: p.next,
                },
            );
        }

        let mut edges = Vec::new();
        let mut push_edge = |kind: EdgeKind, from: u32, to: u32, rope_x: Option<f32>, target_x: f32| {
            if from == 0 || to == 0 || from == to {
                return;
            }
            if !nodes.contains_key(&from) || !nodes.contains_key(&to) {
                return;
            }
            let cost = match kind {
                EdgeKind::Walk => 1,
                EdgeKind::ClimbUp => 2,
                EdgeKind::ClimbDown => 3,
                EdgeKind::Fall => 4,
                // 窄台台阶极易空跳；探索代价高于多段 Walk，避免出生点死磕 StepUp。
                EdgeKind::StepUp => 8,
            };
            edges.push(GraphEdge {
                kind,
                from,
                to,
                rope_x,
                target_x,
                cost,
            });
        };

        for node in nodes.values() {
            if node.prev != 0 {
                // 目标点必须落在目的节点上，否则 goto 走到本台边缘永远进不了 prev。
                let tx = nodes
                    .get(&node.prev)
                    .map(|p| p.x_max - 6.0)
                    .unwrap_or(node.x_min + 6.0);
                push_edge(EdgeKind::Walk, node.id, node.prev, None, tx);
            }
            if node.next != 0 {
                let tx = nodes
                    .get(&node.next)
                    .map(|n| n.x_min + 6.0)
                    .unwrap_or(node.x_max - 6.0);
                push_edge(EdgeKind::Walk, node.id, node.next, None, tx);
            }
        }

        for node in nodes.values() {
            let fh = Some((node.layer, node.group));
            for (side, probe_x, to_x) in [
                (SideProbe::Left, node.x_min + 6.0, node.x_min - 8.0),
                (SideProbe::Right, node.x_max - 6.0, node.x_max + 8.0),
            ] {
                let _ = side;
                match map.walk_ahead(probe_x, node.y, to_x, fh) {
                    WalkAhead::Fall => {
                        // 落地探测必须在台外 to_x，用 probe_x 会落到本台，Fall 边永远建不出来。
                        if let Some(land) = map.land_at(to_x, node.y - 4.0, node.y + 160.0) {
                            // 微落差（同链邻台）交给 Walk，只建真正下到更低层的 Fall。
                            if land.id != 0
                                && land.id != node.id
                                && land.y >= node.y + 20.0
                            {
                                push_edge(
                                    EdgeKind::Fall,
                                    node.id,
                                    land.id,
                                    None,
                                    to_x,
                                );
                            }
                        }
                    }
                    _ => {}
                }
            }

            let mid_x = (node.x_min + node.x_max) * 0.5;
            if let Some(dx) = map.nearest_step_up_dx(mid_x, node.y) {
                let target_x = mid_x + dx;
                if let Some(up) = map.stand_at(target_x, node.y - 80.0, 80.0) {
                    if up.id != node.id && up.y < node.y - 12.0 {
                        push_edge(EdgeKind::StepUp, node.id, up.id, None, target_x);
                    }
                }
            }
        }

        for r in &map.ropes {
            let top = r.y1.min(r.y2);
            let bot = r.y1.max(r.y2);
            let rx = r.x;
            if let Some(bottom) = map.stand_at(rx, bot + 20.0, 40.0) {
                if let Some(top_stand) = map.stand_at_climb_exit(rx, top) {
                    if bottom.id != top_stand.id {
                        push_edge(
                            EdgeKind::ClimbUp,
                            bottom.id,
                            top_stand.id,
                            Some(rx),
                            rx,
                        );
                        push_edge(
                            EdgeKind::ClimbDown,
                            top_stand.id,
                            bottom.id,
                            Some(rx),
                            rx,
                        );
                    }
                }
            }
        }

        let mut adj: HashMap<PlatformNodeId, Vec<usize>> = HashMap::new();
        for (i, e) in edges.iter().enumerate() {
            adj.entry(e.from).or_default().push(i);
        }

        Self { nodes, edges, adj }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn node_at(&self, map: &GameMap, x: f32, y: f32) -> Option<PlatformNodeId> {
        map.stand_at(x, y, 80.0)
            .map(|s| s.id)
            .filter(|id| self.nodes.contains_key(id))
            .or_else(|| {
                map.strict_stand_at(x, y)
                    .map(|s| s.id)
                    .filter(|id| self.nodes.contains_key(id))
            })
    }

    /// 仅按图节点几何：x 落在台面内且 |y-台面| 最小。
    /// 供落地时 OCR y 漂移兜底；爬绳/滞空勿用（会误吸到上下层同 x 台）。
    pub fn node_at_by_xy(&self, x: f32, y: f32, max_dy: f32) -> Option<PlatformNodeId> {
        let mut best: Option<(f32, PlatformNodeId)> = None;
        for n in self.nodes.values() {
            if x < n.x_min - 4.0 || x > n.x_max + 4.0 {
                continue;
            }
            let dy = (n.y - y).abs();
            if dy > max_dy {
                continue;
            }
            if best.map(|(d, _)| dy < d).unwrap_or(true) {
                best = Some((dy, n.id));
            }
        }
        best.map(|(_, id)| id)
    }

    pub fn get(&self, id: PlatformNodeId) -> Option<&PlatformNode> {
        self.nodes.get(&id)
    }

    pub fn is_patrol_platform(&self, id: PlatformNodeId) -> bool {
        self.get(id)
            .map(|n| n.is_patrol_worthy())
            .unwrap_or(false)
    }

    /// 从起点可达、且宽度足够刷怪巡逻的平台节点。
    pub fn patrol_reachable_nodes(
        &self,
        from: PlatformNodeId,
        blocked: &HashMap<(PlatformNodeId, EdgeKind, PlatformNodeId), u32>,
    ) -> HashSet<PlatformNodeId> {
        self.reachable_nodes(from, blocked)
            .into_iter()
            .filter(|&id| self.is_patrol_platform(id))
            .collect()
    }

    fn path_via_hubs(
        &self,
        from: PlatformNodeId,
        to: PlatformNodeId,
        hubs: &[PlatformNodeId],
        blocked: &HashMap<(PlatformNodeId, EdgeKind, PlatformNodeId), u32>,
    ) -> Option<Vec<usize>> {
        if let Some(path) = self.path_between(from, to, blocked) {
            return Some(path);
        }
        for &hub in hubs {
            let (Some(mut path), Some(tail)) = (
                self.path_between(from, hub, blocked),
                self.path_between(hub, to, blocked),
            ) else {
                continue;
            };
            path.extend(tail);
            return Some(path);
        }
        None
    }

    /// 随机顺序 + 固定种子，串联最短路径，覆盖从起点可达的**大平台**（小落脚台跳过）。
    pub fn build_patrol_route(
        &self,
        start: PlatformNodeId,
        seed: u64,
    ) -> Vec<usize> {
        use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

        let blocked = HashMap::new();
        let patrol_nodes = self.patrol_reachable_nodes(start, &blocked);
        if patrol_nodes.is_empty() {
            return Vec::new();
        }

        let cur_start = if patrol_nodes.contains(&start) {
            start
        } else {
            *patrol_nodes.iter().min().unwrap()
        };

        let mut nodes: Vec<PlatformNodeId> = patrol_nodes
            .iter()
            .copied()
            .filter(|&id| id != cur_start)
            .collect();
        nodes.sort_unstable();
        let mut rng = StdRng::seed_from_u64(seed);
        nodes.shuffle(&mut rng);

        let hub_pool: Vec<PlatformNodeId> = patrol_nodes.iter().copied().collect();
        let mut route = Vec::new();
        for target in nodes {
            let Some(path) = self.path_via_hubs(cur_start, target, &hub_pool, &blocked) else {
                continue;
            };
            route.extend(path);
        }
        if !route.is_empty() {
            let last_to = self.edges[*route.last().unwrap()].to;
            if last_to != cur_start {
                if let Some(back) = self.path_via_hubs(last_to, cur_start, &hub_pool, &blocked) {
                    route.extend(back);
                }
            }
        }
        route
    }

    pub fn nearest_unvisited_path(
        &self,
        from: PlatformNodeId,
        visited: &HashSet<PlatformNodeId>,
        blocked: &HashMap<(PlatformNodeId, EdgeKind, PlatformNodeId), u32>,
    ) -> Option<&GraphEdge> {
        let path = self.path_to_nearest_unvisited(from, visited, blocked, 1.0)?;
        path.first().map(|&i| &self.edges[i])
    }

    /// 按边代价找最近未访问节点（StepUp 很贵，优先 Walk/Climb）。
    pub fn path_to_nearest_unvisited(
        &self,
        from: PlatformNodeId,
        visited: &HashSet<PlatformNodeId>,
        blocked: &HashMap<(PlatformNodeId, EdgeKind, PlatformNodeId), u32>,
        prefer_dir: f32,
    ) -> Option<Vec<usize>> {
        self.path_to_nearest_unvisited_with(from, visited, blocked, |_| true, prefer_dir)
    }

    /// `accept_y(to_y)` 过滤目标台高度。
    /// `prefer_dir`：>0 偏好向右，<0 偏好向左；等代价与侧向罚分都跟此方向，不再写死偏右。
    pub fn path_to_nearest_unvisited_with(
        &self,
        from: PlatformNodeId,
        visited: &HashSet<PlatformNodeId>,
        blocked: &HashMap<(PlatformNodeId, EdgeKind, PlatformNodeId), u32>,
        accept_y: impl Fn(f32) -> bool,
        prefer_dir: f32,
    ) -> Option<Vec<usize>> {
        if !self.nodes.contains_key(&from) {
            return None;
        }
        let prefer_right = prefer_dir >= 0.0;
        let mut best: HashMap<PlatformNodeId, u32> = HashMap::new();
        let mut parent: HashMap<PlatformNodeId, (PlatformNodeId, usize)> = HashMap::new();
        let mut heap = std::collections::BinaryHeap::new();
        best.insert(from, 0);
        heap.push(std::cmp::Reverse((0u32, from)));
        let mut target = None;
        let mut target_cost = u32::MAX;
        let mut target_mid_x = if prefer_right {
            f32::MIN
        } else {
            f32::MAX
        };

        while let Some(std::cmp::Reverse((cost, cur))) = heap.pop() {
            if best.get(&cur).copied().unwrap_or(u32::MAX) < cost {
                continue;
            }
            if cur != from && !visited.contains(&cur) {
                let to_y = self.nodes.get(&cur).map(|n| n.y).unwrap_or(0.0);
                let mid_x = self.node_mid_x(cur);
                let better_tie = if prefer_right {
                    mid_x > target_mid_x
                } else {
                    mid_x < target_mid_x
                };
                if accept_y(to_y) && (cost < target_cost || (cost == target_cost && better_tie)) {
                    target = Some(cur);
                    target_cost = cost;
                    target_mid_x = mid_x;
                }
                continue;
            }
            if cost >= target_cost {
                continue;
            }
            let Some(indices) = self.adj.get(&cur) else {
                continue;
            };
            for &idx in indices {
                let e = &self.edges[idx];
                if blocked.contains_key(&(e.from, e.kind, e.to)) {
                    continue;
                }
                let mut step = e.cost.max(1);
                if e.kind == EdgeKind::StepUp {
                    if let Some(d) = self.nodes.get(&e.to) {
                        if d.x_max - d.x_min < 48.0 {
                            step = step.saturating_add(4);
                        }
                    }
                }
                let cur_x = self.node_mid_x(cur);
                let to_x = self.node_mid_x(e.to);
                // 逆着当前巡逻方向的探索边加罚，使方向粘滞；不再写死罚左侧。
                let against = if prefer_right {
                    to_x + 40.0 < cur_x
                } else {
                    to_x > cur_x + 40.0
                };
                let against_soft = if prefer_right {
                    to_x + 80.0 < cur_x
                } else {
                    to_x > cur_x + 80.0
                };
                if against {
                    step = step.saturating_add(5);
                } else if against_soft {
                    step = step.saturating_add(2);
                }
                let next = cost.saturating_add(step);
                if next < best.get(&e.to).copied().unwrap_or(u32::MAX) {
                    best.insert(e.to, next);
                    parent.insert(e.to, (cur, idx));
                    heap.push(std::cmp::Reverse((next, e.to)));
                }
            }
        }

        let target = target?;
        let mut path = Vec::new();
        let mut cur = target;
        while cur != from {
            let (prev, idx) = parent.get(&cur)?;
            path.push(*idx);
            cur = *prev;
        }
        path.reverse();
        Some(path)
    }

    fn node_mid_x(&self, id: PlatformNodeId) -> f32 {
        self.nodes
            .get(&id)
            .map(|n| (n.x_min + n.x_max) * 0.5)
            .unwrap_or(0.0)
    }

    /// BFS 最短路径（边索引序列）。
    pub fn path_between(
        &self,
        from: PlatformNodeId,
        to: PlatformNodeId,
        blocked: &HashMap<(PlatformNodeId, EdgeKind, PlatformNodeId), u32>,
    ) -> Option<Vec<usize>> {
        if from == to {
            return Some(Vec::new());
        }
        if !self.nodes.contains_key(&from) || !self.nodes.contains_key(&to) {
            return None;
        }
        let mut queue = std::collections::VecDeque::new();
        let mut seen = HashSet::new();
        let mut parent: HashMap<PlatformNodeId, (PlatformNodeId, usize)> = HashMap::new();
        queue.push_back(from);
        seen.insert(from);

        while let Some(cur) = queue.pop_front() {
            if cur == to {
                break;
            }
            let Some(indices) = self.adj.get(&cur) else {
                continue;
            };
            for &idx in indices {
                let e = &self.edges[idx];
                if blocked.contains_key(&(e.from, e.kind, e.to)) {
                    continue;
                }
                if seen.insert(e.to) {
                    parent.insert(e.to, (cur, idx));
                    queue.push_back(e.to);
                }
            }
        }

        if !parent.contains_key(&to) && from != to {
            return None;
        }

        let mut path = Vec::new();
        let mut cur = to;
        while cur != from {
            let (prev, idx) = parent.get(&cur)?;
            path.push(*idx);
            cur = *prev;
        }
        path.reverse();
        Some(path)
    }

    pub fn reachable_nodes(
        &self,
        from: PlatformNodeId,
        blocked: &HashMap<(PlatformNodeId, EdgeKind, PlatformNodeId), u32>,
    ) -> HashSet<PlatformNodeId> {
        let mut seen = HashSet::new();
        if !self.nodes.contains_key(&from) {
            return seen;
        }
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(from);
        seen.insert(from);
        while let Some(cur) = queue.pop_front() {
            let Some(indices) = self.adj.get(&cur) else {
                continue;
            };
            for &idx in indices {
                let e = &self.edges[idx];
                if blocked.contains_key(&(e.from, e.kind, e.to)) {
                    continue;
                }
                if seen.insert(e.to) {
                    queue.push_back(e.to);
                }
            }
        }
        seen
    }

    /// BFS 找不到未访问节点时，优先选跨平台边（落点/绳梯/台阶）。
    /// 先选落点为巡逻级宽台的边，避免 farm 清空后反复跳 39px 小台。
    /// 同优先级下偏好更靠右的落点。
    pub fn prefer_cross_platform_edge(
        &self,
        from: PlatformNodeId,
        blocked: &HashMap<(PlatformNodeId, EdgeKind, PlatformNodeId), u32>,
        prefer_dir: f32,
    ) -> Option<&GraphEdge> {
        let indices = self.adj.get(&from)?;
        let kinds = [
            EdgeKind::ClimbUp,
            EdgeKind::Fall,
            EdgeKind::ClimbDown,
            EdgeKind::StepUp,
        ];
        let from_mid = self.node_mid_x(from);
        let prefer_right = prefer_dir >= 0.0;
        for prefer_wide in [true, false] {
            let mut best_idx: Option<usize> = None;
            let mut best_key: Option<(u32, u32, f32)> = None;
            for (ki, kind) in kinds.iter().enumerate() {
                for &idx in indices {
                    let e = &self.edges[idx];
                    if e.kind != *kind || blocked.contains_key(&(e.from, e.kind, e.to)) {
                        continue;
                    }
                    if prefer_wide && !self.is_patrol_platform(e.to) {
                        continue;
                    }
                    let to_x = self.node_mid_x(e.to);
                    let forward = if prefer_right {
                        to_x + 24.0 >= from_mid
                    } else {
                        to_x - 24.0 <= from_mid
                    };
                    let dir_pen = if forward { 0u32 } else { 1u32 };
                    // 同向时选更远一点的开荒，避免立刻折返；不再用 -to_x 写死偏右。
                    let reach = if prefer_right { to_x } else { -to_x };
                    let key = (ki as u32, dir_pen, -reach);
                    if best_key.map(|bk| key < bk).unwrap_or(true) {
                        best_key = Some(key);
                        best_idx = Some(idx);
                    }
                }
            }
            if let Some(idx) = best_idx {
                return Some(&self.edges[idx]);
            }
        }
        None
    }

    /// 按边代价找最近的指定种类边（含通向它的路径）。
    pub fn path_to_nearest_kind(
        &self,
        from: PlatformNodeId,
        kind: EdgeKind,
        blocked: &HashMap<(PlatformNodeId, EdgeKind, PlatformNodeId), u32>,
    ) -> Option<Vec<usize>> {
        if !self.nodes.contains_key(&from) {
            return None;
        }
        let mut best: HashMap<PlatformNodeId, u32> = HashMap::new();
        let mut parent: HashMap<PlatformNodeId, (PlatformNodeId, usize)> = HashMap::new();
        let mut heap = std::collections::BinaryHeap::new();
        best.insert(from, 0);
        heap.push(std::cmp::Reverse((0u32, from)));
        let mut best_path: Option<(u32, Vec<usize>)> = None;

        while let Some(std::cmp::Reverse((cost, cur))) = heap.pop() {
            if best.get(&cur).copied().unwrap_or(u32::MAX) < cost {
                continue;
            }
            if best_path.as_ref().is_some_and(|(bc, _)| cost >= *bc) {
                continue;
            }
            let Some(indices) = self.adj.get(&cur) else {
                continue;
            };
            for &idx in indices {
                let e = &self.edges[idx];
                if blocked.contains_key(&(e.from, e.kind, e.to)) {
                    continue;
                }
                if e.kind == kind {
                    let mut path = Vec::new();
                    let mut node = e.from;
                    let mut ok = true;
                    while node != from {
                        let Some(&(prev, edge_idx)) = parent.get(&node) else {
                            ok = false;
                            break;
                        };
                        path.push(edge_idx);
                        node = prev;
                    }
                    if !ok {
                        continue;
                    }
                    path.reverse();
                    path.push(idx);
                    let total = cost.saturating_add(e.cost.max(1));
                    if best_path
                        .as_ref()
                        .map(|(bc, _)| total < *bc)
                        .unwrap_or(true)
                    {
                        best_path = Some((total, path));
                    }
                    continue;
                }
                let mut step = e.cost.max(1);
                if e.kind == EdgeKind::StepUp {
                    if let Some(d) = self.nodes.get(&e.to) {
                        if d.x_max - d.x_min < 48.0 {
                            step = step.saturating_add(4);
                        }
                    }
                }
                let next = cost.saturating_add(step);
                if next < best.get(&e.to).copied().unwrap_or(u32::MAX) {
                    best.insert(e.to, next);
                    parent.insert(e.to, (cur, idx));
                    heap.push(std::cmp::Reverse((next, e.to)));
                }
            }
        }
        best_path.map(|(_, p)| p)
    }

    pub fn edge_to_subgoal(&self, edge: &GraphEdge) -> super::types::SubGoal {
        use super::types::{Side, SubGoal};
        match edge.kind {
            EdgeKind::Walk => SubGoal::GoTo { x: edge.target_x },
            EdgeKind::Fall => SubGoal::WalkOff {
                side: if edge.target_x <= self.nodes[&edge.from].x_min + 8.0 {
                    Side::Left
                } else {
                    Side::Right
                },
            },
            EdgeKind::ClimbUp => SubGoal::ClimbUp {
                rope_x: edge.rope_x.unwrap_or(edge.target_x),
            },
            EdgeKind::ClimbDown => SubGoal::ClimbDown {
                rope_x: edge.rope_x.unwrap_or(edge.target_x),
            },
            EdgeKind::StepUp => SubGoal::StepUp {
                target_x: edge.target_x,
            },
        }
    }
}

enum SideProbe {
    Left,
    Right,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::load_default_map;

    #[test]
    fn spawn_unvisited_prefers_walk_over_narrow_stepup() {
        let map = load_default_map().expect("map");
        let g = MapGraph::build(&map);
        let mut visited = HashSet::new();
        visited.insert(46);
        let blocked = HashMap::new();
        let path = g
            .path_to_nearest_unvisited(46, &visited, &blocked, 1.0)
            .expect("path");
        let first = &g.edges[path[0]];
        eprintln!(
            "from 46 first {:?} ->{} cost path_len={}",
            first.kind,
            first.to,
            path.len()
        );
        assert_eq!(
            first.kind,
            EdgeKind::Walk,
            "should Walk to 38/50 before StepUp to narrow 15, got {:?}",
            first.kind
        );
    }

    #[test]
    #[ignore]
    fn dump_spawn_cluster_exits() {
        let map = load_default_map().expect("map");
        let g = MapGraph::build(&map);
        for id in [15u32, 19, 46, 50, 38, 41, 12, 16] {
            let Some(n) = g.get(id) else { continue };
            eprintln!(
                "node {id} x={:.0}..{:.0} y={:.0} w={:.0}",
                n.x_min,
                n.x_max,
                n.y,
                n.x_max - n.x_min
            );
            for e in &g.edges {
                if e.from != id {
                    continue;
                }
                let ty = g.get(e.to).map(|d| d.y).unwrap_or(0.0);
                eprintln!(
                    "  {:?} ->{} tx={:.0} rope={:?} to_y={:.0}",
                    e.kind, e.to, e.target_x, e.rope_x, ty
                );
            }
        }
        for id in [47u32, 57, 58, 52, 56, 48] {
            let Some(n) = g.get(id) else { continue };
            eprintln!("right {id} x={:.0}..{:.0} y={:.0}", n.x_min, n.x_max, n.y);
            for e in &g.edges {
                if e.from == id && matches!(e.kind, EdgeKind::ClimbUp | EdgeKind::StepUp) {
                    eprintln!("  {:?} ->{} rope={:?}", e.kind, e.to, e.rope_x);
                }
            }
        }
    }

    #[test]
    fn walk_edge_target_lies_on_destination_node() {
        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        let e = graph
            .edges
            .iter()
            .find(|e| e.from == 46 && e.to == 50 && e.kind == EdgeKind::Walk)
            .expect("46-walk->50");
        let dest = graph.get(50).expect("node 50");
        assert!(
            e.target_x >= dest.x_min - 1.0 && e.target_x <= dest.x_max + 1.0,
            "tx={:.0} not on dest [{:.0},{:.0}]",
            e.target_x,
            dest.x_min,
            dest.x_max
        );
    }

    #[test]
    fn spawn_platform_has_exit_edge() {
        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        let (sx, sy) = map.default_spawn();
        let id = graph.node_at(&map, sx, sy).expect("spawn node");
        let blocked = std::collections::HashMap::new();
        let edge = graph
            .prefer_cross_platform_edge(id, &blocked, 1.0)
            .or_else(|| graph.nearest_unvisited_path(id, &Default::default(), &blocked));
        assert!(
            edge.is_some(),
            "spawn node {id} should have cross-platform or explore edge"
        );
    }

    #[test]
    fn path_to_unvisited_skips_satisfied_walk_hops() {
        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        let (sx, sy) = map.default_spawn();
        let from = graph.node_at(&map, sx, sy).expect("spawn");
        let blocked = HashMap::new();
        let mut visited = HashSet::new();
        visited.insert(from);
        let path = graph
            .path_to_nearest_unvisited(from, &visited, &blocked, 1.0)
            .expect("path");
        assert!(!path.is_empty());
    }

    #[test]
    fn node_at_by_xy_recovers_when_ocr_y_drifts_below_platform() {
        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        let n38 = graph.get(38).expect("node 38");
        assert_eq!(graph.node_at(&map, n38.x_min + 2.0, n38.y), Some(38));
        // 严格 stand_at 在 y 漂到台下时会空；几何兜底仍认 46。
        assert_eq!(
            graph.node_at_by_xy(251.0, n38.y + 65.0, 120.0),
            Some(46),
            "x=251 is on node 46 even if y drifts"
        );
        // node_at 本身不再做松散吸附（避免爬绳误吸）。
        assert!(graph.node_at(&map, 251.0, n38.y + 65.0).is_none());
    }

    #[test]
    fn platform_midpoint_snaps_to_node() {
        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        let seg = map
            .platforms
            .iter()
            .find(|p| (p.x1 - p.x2).abs() >= 40.0)
            .expect("wide segment");
        let x = (seg.x1 + seg.x2) * 0.5;
        let y = (seg.y1 + seg.y2) * 0.5;
        let id = graph.node_at(&map, x, y).expect("platform node");
        assert!(graph.get(id).is_some());
    }

    #[test]
    fn patrol_route_skips_narrow_landing_platforms() {
        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        let narrow: Vec<_> = graph
            .nodes
            .values()
            .filter(|n| !n.is_patrol_worthy())
            .collect();
        assert!(
            !narrow.is_empty(),
            "default map should have some narrow landing segments"
        );
        let (sx, sy) = map.default_spawn();
        let start = graph.node_at(&map, sx, sy).expect("spawn");
        let blocked: HashMap<(PlatformNodeId, EdgeKind, PlatformNodeId), u32> = HashMap::new();
        let all = graph.reachable_nodes(start, &blocked);
        let patrol = graph.patrol_reachable_nodes(start, &blocked);
        assert!(
            patrol.len() < all.len(),
            "patrol set should exclude narrow platforms"
        );
    }

    #[test]
    fn patrol_route_covers_patrol_worthy_nodes() {
        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        let (sx, sy) = map.default_spawn();
        let start = graph.node_at(&map, sx, sy).expect("spawn");
        let route = graph.build_patrol_route(start, 42);
        assert!(!route.is_empty(), "patrol route should not be empty");

        let blocked: HashMap<(PlatformNodeId, EdgeKind, PlatformNodeId), u32> = HashMap::new();
        let patrol_reachable = graph.patrol_reachable_nodes(start, &blocked);

        let mut touched_patrol = HashSet::new();
        for &idx in &route {
            let e = &graph.edges[idx];
            if graph.is_patrol_platform(e.from) {
                touched_patrol.insert(e.from);
            }
            if graph.is_patrol_platform(e.to) {
                touched_patrol.insert(e.to);
            }
        }
        assert_eq!(
            touched_patrol.len(),
            patrol_reachable.len(),
            "patrol route should touch all {} patrol-worthy nodes, got {}",
            patrol_reachable.len(),
            touched_patrol.len()
        );
    }

    #[test]
    fn fall_edges_exist_from_small_ledges_132_135() {
        let map = load_default_map().expect("map");
        let g = MapGraph::build(&map);
        // 132 左端应落到大台；135 至少一侧有显著落差 Fall（微落差邻台不建边）。
        let falls_132: Vec<_> = g
            .edges
            .iter()
            .filter(|e| e.from == 132 && e.kind == EdgeKind::Fall)
            .map(|e| (e.to, e.target_x))
            .collect();
        eprintln!("node 132 falls={falls_132:?}");
        assert!(
            falls_132.iter().any(|(to, _)| *to == 113 || *to == 118),
            "132 should Fall to big floor, got {falls_132:?}"
        );
        let falls_135: Vec<_> = g
            .edges
            .iter()
            .filter(|e| e.from == 135 && e.kind == EdgeKind::Fall)
            .map(|e| (e.to, e.target_x))
            .collect();
        eprintln!("node 135 falls={falls_135:?}");
        assert!(
            !falls_135.is_empty()
                && falls_135
                    .iter()
                    .all(|(to, _)| *to != 134 && *to != 133),
            "135 should have real Fall (not micro onto 133/134), got {falls_135:?}"
        );
    }

    #[test]
    fn path_between_adjacent_walk_nodes() {
        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        let blocked = HashMap::new();
        // node 60/62 are prev/next on default map lower platform
        if graph.nodes.contains_key(&60) && graph.nodes.contains_key(&62) {
            let path = graph.path_between(60, 62, &blocked).expect("path 60->62");
            assert!(!path.is_empty());
            assert_eq!(graph.edges[path[0]].from, 60);
            assert_eq!(graph.edges[path.last().copied().unwrap()].to, 62);
        }
    }
}
