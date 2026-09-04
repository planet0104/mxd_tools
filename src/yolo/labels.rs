/// 与 dataset/.../generated/yolo/data.yaml 对齐（31 类 = 原 21 + 血条10%～100%）。
pub const CLASS_NAMES: [&str; 31] = [
    "地板",
    "梯子",
    "绳子",
    "入口",
    "出口",
    "花蘑菇",
    "蓝蜗牛",
    "绿蜗牛",
    "红蜗牛",
    "树怪",
    "玩家",
    "金币",
    "药水",
    "武器",
    "装备",
    "材料",
    "小地图",
    "任务窗",
    "浮动按钮",
    "面板",
    "键盘",
    "血条10%",
    "血条20%",
    "血条30%",
    "血条40%",
    "血条50%",
    "血条60%",
    "血条70%",
    "血条80%",
    "血条90%",
    "血条100%",
];

/// 血条类在 CLASS_NAMES 中的起始下标（含）。
pub const HP_BAR_CLASS_FIRST: usize = 21;
/// 血条类在 CLASS_NAMES 中的结束下标（含）。
pub const HP_BAR_CLASS_LAST: usize = 30;

pub fn class_name(id: usize) -> &'static str {
    CLASS_NAMES.get(id).copied().unwrap_or("未知")
}

/// 若检测框是血条类，返回估计血量比例（10%→0.1 … 100%→1.0）。
pub fn hp_ratio_from_class_id(class_id: usize) -> Option<f32> {
    if (HP_BAR_CLASS_FIRST..=HP_BAR_CLASS_LAST).contains(&class_id) {
        let step = class_id - HP_BAR_CLASS_FIRST + 1;
        Some((step as f32) * 0.1)
    } else {
        None
    }
}

/// 从标签名解析血量比例（如「血条70%」→ 0.7）。
pub fn hp_ratio_from_label(label: &str) -> Option<f32> {
    let rest = label.strip_prefix("血条")?.strip_suffix('%')?;
    let pct: u32 = rest.parse().ok()?;
    if (10..=100).contains(&pct) && pct % 10 == 0 {
        Some(pct as f32 / 100.0)
    } else {
        None
    }
}
