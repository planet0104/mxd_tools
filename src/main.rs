#[cfg(windows)]
mod keyboard_input;
#[cfg(windows)]
mod no_activate;

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

use eframe::egui;
use mxd_tools::capture;
use mxd_tools::locate;
use mxd_tools::map_api;
use mxd_tools::ocr;
use mxd_tools::paths::{maps_dir, minimap_shots_dir, workspace_root};

/// 加载系统自带中文字体，避免 egui 默认字体缺字显示为方框/乱码。
fn setup_cjk_fonts(ctx: &egui::Context) {
    let candidates: &[(&str, u32)] = &[
        (r"C:\Windows\Fonts\msyh.ttc", 0),   // 微软雅黑
        (r"C:\Windows\Fonts\msyhbd.ttc", 0),
        (r"C:\Windows\Fonts\simhei.ttf", 0), // 黑体
        (r"C:\Windows\Fonts\simsun.ttc", 0), // 宋体
        (r"C:\Windows\Fonts\msjh.ttc", 0),   // 微软正黑（部分系统）
    ];

    let mut chosen: Option<(Vec<u8>, u32, String)> = None;
    for &(path, index) in candidates {
        if Path::new(path).is_file() {
            if let Ok(data) = std::fs::read(path) {
                chosen = Some((
                    data,
                    index,
                    Path::new(path)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("cjk")
                        .to_string(),
                ));
                break;
            }
        }
    }

    let Some((data, index, name)) = chosen else {
        return;
    };

    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        name.clone(),
        std::sync::Arc::new(egui::FontData {
            font: std::borrow::Cow::Owned(data),
            index,
            tweak: egui::FontTweak::default(),
        }),
    );

    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        if let Some(fonts_for_family) = fonts.families.get_mut(&family) {
            fonts_for_family.insert(0, name.clone());
        }
    }

    ctx.set_fonts(fonts);
}

enum JobKind {
    CaptureAndOcr,
    SaveMap,
    SaveFromMinimap,
    Locate,
}

enum JobResult {
    Log(String),
    /// OCR / 识别得到的地图名，写入输入框
    MapName(String),
    Done,
}

struct App {
    root: PathBuf,
    map_name: String,
    log: String,
    busy: bool,
    tx: Option<Sender<JobResult>>,
    rx: Receiver<JobResult>,
}

impl App {
    fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            root: workspace_root(),
            map_name: String::new(),
            log: "就绪。先点「截取小地图并识别」，OCR 结果会填入下方地图名。\n".into(),
            busy: false,
            tx: Some(tx),
            rx,
        }
    }

    fn append_log(&mut self, text: impl AsRef<str>) {
        self.log.push_str(text.as_ref());
        if !self.log.ends_with('\n') {
            self.log.push('\n');
        }
    }

    fn spawn_job(&mut self, kind: JobKind) {
        if self.busy {
            self.append_log("请等待当前任务完成");
            return;
        }
        let Some(tx0) = &self.tx else {
            return;
        };
        let tx = tx0.clone();
        let root = self.root.clone();
        let map_name = self.map_name.clone();
        self.busy = true;
        self.append_log("任务开始…");
        thread::spawn(move || {
            match kind {
                JobKind::CaptureAndOcr => {
                    match capture_and_ocr(&root) {
                        Ok((path, query, street, name)) => {
                            let _ = tx.send(JobResult::MapName(query.clone()));
                            let _ = tx.send(JobResult::Log(format!(
                                "已截取 {}\nOCR：一级地图 {street} / 二级地图 {name}\n已填入地图名：{query}",
                                path.display()
                            )));
                        }
                        Err(e) => {
                            let _ = tx.send(JobResult::Log(format!("失败：{e}")));
                        }
                    }
                }
                JobKind::SaveMap => {
                    if map_name.trim().is_empty() {
                        let _ = tx.send(JobResult::Log(
                            "失败：地图名为空，请先「截取小地图并识别」或手动填写".into(),
                        ));
                    } else {
                        match map_api::save_map(&map_name, &maps_dir(&root)) {
                            Ok((id, path, label)) => {
                                let _ = tx.send(JobResult::Log(format!(
                                    "地图ID {id}\n已保存 {}\n{label}",
                                    path.display()
                                )));
                            }
                            Err(e) => {
                                let _ = tx.send(JobResult::Log(format!("失败：{e}")));
                            }
                        }
                    }
                }
                JobKind::SaveFromMinimap => {
                    match locate::save_from_minimap(&root) {
                        Ok((street, name, id, path)) => {
                            let query = format!("{street}-{name}");
                            let _ = tx.send(JobResult::MapName(query));
                            let _ = tx.send(JobResult::Log(format!(
                                "一级地图 {street}\n二级地图 {name}\n地图ID {id}\n已保存 {}",
                                path.display()
                            )));
                        }
                        Err(e) => {
                            let _ = tx.send(JobResult::Log(format!("失败：{e}")));
                        }
                    }
                }
                JobKind::Locate => {
                    match locate::locate_player(&root) {
                        Ok(msg) => {
                            if let Some(query) = parse_map_query_from_locate_log(&msg) {
                                let _ = tx.send(JobResult::MapName(query));
                            }
                            let _ = tx.send(JobResult::Log(msg));
                        }
                        Err(e) => {
                            let _ = tx.send(JobResult::Log(format!("失败：{e}")));
                        }
                    }
                }
            }
            let _ = tx.send(JobResult::Done);
        });
    }

    fn poll_jobs(&mut self) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                JobResult::Log(s) => self.append_log(s),
                JobResult::MapName(name) => {
                    self.map_name = name;
                }
                JobResult::Done => {
                    self.busy = false;
                    self.append_log("任务结束");
                }
            }
        }
    }

    #[cfg(windows)]
    fn send_key(&mut self, modifiers: &[keyboard_input::Modifier], key: keyboard_input::Key) {
        match keyboard_input::tap(modifiers, key) {
            Ok(()) => self.append_log(format!("已发送按键：{}", key.label())),
            Err(e) => self.append_log(format!("发送按键失败：{e}")),
        }
    }

    #[cfg(windows)]
    fn send_combo(
        &mut self,
        label: &str,
        modifiers: &[keyboard_input::Modifier],
        keys: &[keyboard_input::Key],
    ) {
        match keyboard_input::tap_keys(modifiers, keys) {
            Ok(()) => self.append_log(format!("已发送组合键：{label}")),
            Err(e) => self.append_log(format!("发送组合键失败：{e}")),
        }
    }
}

fn capture_and_ocr(root: &Path) -> Result<(PathBuf, String, String, String), String> {
    let path = capture::capture_minimap(&minimap_shots_dir(root))?;
    let img = image::open(&path)
        .map_err(|e| e.to_string())?
        .to_rgb8();
    let (street, name) = ocr::read_map_names(&img)?;
    let query = format!("{street}-{name}");
    Ok((path, query, street, name))
}

fn parse_map_query_from_locate_log(msg: &str) -> Option<String> {
    let mut street = None;
    let mut name = None;
    for line in msg.lines() {
        if let Some(rest) = line.strip_prefix("一级地图 ") {
            street = Some(rest.trim().to_string());
        }
        if let Some(rest) = line.strip_prefix("二级地图 ") {
            name = Some(rest.trim().to_string());
        }
    }
    match (street, name) {
        (Some(s), Some(n)) => Some(format!("{s}-{n}")),
        _ => None,
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        self.poll_jobs();
        if self.busy {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }

        #[cfg(windows)]
        no_activate::sync_from_handle(frame);

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("冒险岛经典版工具");
            ui.label(format!("输出目录：{}", self.root.display()));
            ui.label("自动查找正在运行的 Maplestory_Classic.exe 截图识别");

            #[cfg(windows)]
            {
                let mut no_steal = no_activate::enabled();
                if ui
                    .checkbox(&mut no_steal, "点击不夺取焦点（类似屏幕键盘）")
                    .on_hover_text("开启后点本窗口按钮不会抢走其他程序焦点，便于向目标窗口发键")
                    .changed()
                {
                    no_activate::set_enabled(no_steal);
                    no_activate::sync_from_handle(frame);
                }
                if no_steal {
                    ui.label("先点击目标窗口（如记事本），再点下方按键即可输入到该窗口。");
                }
            }

            ui.separator();

            ui.add_enabled_ui(!self.busy, |ui| {
                if ui
                    .button("截取小地图并识别")
                    .on_hover_text("从游戏截取左上角小地图，OCR 后填入下方地图名")
                    .clicked()
                {
                    self.spawn_job(JobKind::CaptureAndOcr);
                }
            });

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.label("地图名");
                ui.add(
                    egui::TextEdit::singleline(&mut self.map_name)
                        .desired_width(280.0)
                        .hint_text("截取识别后自动填充，也可手动修改"),
                );
            });
            ui.label("识别结果可用于「下载完整地图」；也可手动改名后再下载。");

            ui.add_space(8.0);
            ui.horizontal_wrapped(|ui| {
                ui.add_enabled_ui(!self.busy, |ui| {
                    if ui.button("下载完整地图").clicked() {
                        self.spawn_job(JobKind::SaveMap);
                    }
                    if ui.button("从小地图下载").clicked() {
                        self.spawn_job(JobKind::SaveFromMinimap);
                    }
                    if ui.button("定位玩家").clicked() {
                        self.spawn_job(JobKind::Locate);
                    }
                });
            });

            if self.busy {
                ui.label("执行中…");
            }

            #[cfg(windows)]
            {
                ui.separator();
                ui.heading("键盘输入测试");
                ui.label("使用 SendInput 向当前前台窗口注入按键（无需驱动 / 测试模式）。");
                ui.label("测试按键（请先让目标窗口获得焦点）：");
                ui.horizontal_wrapped(|ui| {
                    use keyboard_input::Key;
                    let keys = [
                        Key::A,
                        Key::W,
                        Key::S,
                        Key::D,
                        Key::Space,
                        Key::Enter,
                        Key::Esc,
                        Key::Tab,
                    ];
                    for key in keys {
                        if ui.button(key.label()).clicked() {
                            self.send_key(&[], key);
                        }
                    }
                });
                ui.horizontal(|ui| {
                    use keyboard_input::Key;
                    ui.label("方向键：");
                    for key in [Key::Left, Key::Right, Key::Up, Key::Down] {
                        if ui.button(key.label()).clicked() {
                            self.send_key(&[], key);
                        }
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    use keyboard_input::{Key, Modifier};
                    if ui.button("Ctrl+C").clicked() {
                        self.send_combo("Ctrl+C", &[Modifier::LeftCtrl], &[Key::C]);
                    }
                    if ui.button("Shift+A").clicked() {
                        self.send_combo("Shift+A", &[Modifier::LeftShift], &[Key::A]);
                    }
                    if ui.button("W+A 同按").clicked() {
                        self.send_combo("W+A", &[], &[Key::W, Key::A]);
                    }
                });
            }

            ui.separator();
            ui.label("日志");
            egui::ScrollArea::vertical()
                .stick_to_bottom(true)
                .max_height(360.0)
                .show(ui, |ui| {
                    ui.add(
                        egui::TextEdit::multiline(&mut self.log)
                            .desired_width(f32::INFINITY)
                            .desired_rows(18)
                            .font(egui::TextStyle::Monospace),
                    );
                });
        });
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([760.0, 720.0])
            .with_title("冒险岛经典版工具"),
        ..Default::default()
    };
    eframe::run_native(
        "mxd_tools",
        options,
        Box::new(|cc| {
            setup_cjk_fonts(&cc.egui_ctx);
            #[cfg(windows)]
            no_activate::sync_from_handle(cc);
            Ok(Box::new(App::new()))
        }),
    )
}
