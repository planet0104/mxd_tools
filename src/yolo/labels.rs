/// 与 dataset/.../generated/yolo/data.yaml 对齐（21 类）。
pub const CLASS_NAMES: [&str; 21] = [
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
];

pub fn class_name(id: usize) -> &'static str {
    CLASS_NAMES.get(id).copied().unwrap_or("未知")
}
