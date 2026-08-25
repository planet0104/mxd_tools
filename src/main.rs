#[cfg(windows)]
mod keyboard_input;
#[cfg(windows)]
mod no_activate;

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

use eframe::egui;
use mxd_tools::map_api;
use mxd_tools::paths::{maps_dir, workspace_root};

/// 加载系统自带中文字体，避免 egui 默认字体缺字显示为方框/乱码。
fn setup_cjk_fonts(ctx: &egui::Context) {
    let candidates: &[(&str, u32)] = &[
        (r"C:\Windows\Fonts\msyh.ttc", 0),
        (r"C:\Windows\Fonts\msyhbd.ttc", 0),
        (r"C:\Windows\Fonts\simhei.ttf", 0),
        (r"C:\Windows\Fonts\simsun.ttc", 0),
        (r"C:\Windows\Fonts\msjh.ttc", 0),
    ];

    let mut chosen: Option<(Vec<u8>, u32, String)> = None;
    for &(path, index) in candidates {
        if std::path::Path::new(path).is_file() {
            if let Ok(data) = std::fs::read(path) {
                chosen = Some((
                    data,
                    index,
                    std::path::Path::new(path)
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

enum JobResult {
    Log(String),
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
            map_name: "彩虹岛-南港西郊平原".into(),
            log: "就绪。地图名可填中文或数字 ID。\n· 提取小地图与完整图：从网络下载资源\n".into(),
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

    fn spawn_extract_map(&mut self) {
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
            if map_name.trim().is_empty() {
                let _ = tx.send(JobResult::Log("失败：请先填写地图名或地图 ID".into()));
            } else {
                match map_api::extract_map_by_name(&map_name, &maps_dir(&root)) {
                    Ok((id, mini, full, label)) => {
                        let _ = tx.send(JobResult::Log(format!(
                            "地图ID {id}\n{label}\n小地图 {}\n完整图 {}",
                            mini.display(),
                            full.display()
                        )));
                    }
                    Err(e) => {
                        let _ = tx.send(JobResult::Log(format!("失败：{e}")));
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
            ui.label(format!("工作目录：{}", self.root.display()));

            #[cfg(windows)]
            {
                let mut no_steal = no_activate::enabled();
                if ui
                    .checkbox(&mut no_steal, "点击不夺取焦点（类似屏幕键盘）")
                    .on_hover_text("开启后点本窗口按钮不会抢走其他程序焦点")
                    .changed()
                {
                    no_activate::set_enabled(no_steal);
                    no_activate::sync_from_handle(frame);
                }
            }

            ui.separator();
            ui.horizontal(|ui| {
                ui.label("地图名");
                ui.add(
                    egui::TextEdit::singleline(&mut self.map_name)
                        .desired_width(320.0)
                        .hint_text("例：彩虹岛-南港西郊平原 或 50001"),
                );
            });

            ui.add_space(8.0);
            ui.horizontal_wrapped(|ui| {
                ui.add_enabled_ui(!self.busy, |ui| {
                    if ui
                        .button("提取小地图与完整图")
                        .on_hover_text("网络下载 minimap + render 到 maps/")
                        .clicked()
                    {
                        self.spawn_extract_map();
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
                ui.label("SendInput 注入到当前前台窗口。");
                ui.horizontal_wrapped(|ui| {
                    use keyboard_input::Key;
                    for key in [
                        Key::A,
                        Key::W,
                        Key::S,
                        Key::D,
                        Key::Space,
                        Key::Enter,
                        Key::Esc,
                        Key::Tab,
                    ] {
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
                .max_height(320.0)
                .show(ui, |ui| {
                    ui.add(
                        egui::TextEdit::multiline(&mut self.log)
                            .desired_width(f32::INFINITY)
                            .desired_rows(16)
                            .font(egui::TextStyle::Monospace),
                    );
                });
        });
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([820.0, 760.0])
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
