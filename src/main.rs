#[cfg(windows)]
mod firmware_flash;
#[cfg(windows)]
mod keyboard_input;
#[cfg(windows)]
mod live_nav;
#[cfg(windows)]
mod live_nav_diag;
#[cfg(windows)]
mod no_activate;
#[cfg(windows)]
mod usb_hid;
#[cfg(windows)]
mod win_capture;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
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
    #[cfg(windows)]
    Nav(live_nav::LiveNavEvent),
}

struct App {
    root: PathBuf,
    map_name: String,
    log: String,
    busy: bool,
    tx: Option<Sender<JobResult>>,
    rx: Receiver<JobResult>,
    #[cfg(windows)]
    nav_running: bool,
    #[cfg(windows)]
    nav_stop: Option<Arc<AtomicBool>>,
    #[cfg(windows)]
    nav_status: String,
    #[cfg(windows)]
    capture_target: win_capture::CaptureTarget,
    #[cfg(windows)]
    capture_detect: String,
    #[cfg(windows)]
    kb_backend: keyboard_input::KeyboardBackend,
    #[cfg(windows)]
    usb_port: String,
    #[cfg(windows)]
    kb_session: Option<keyboard_input::KeyboardSession>,
    #[cfg(windows)]
    kb_status: String,
}

impl App {
    fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            root: workspace_root(),
            map_name: "彩虹岛-南港西郊平原".into(),
            log: "就绪。地图名可填中文或数字 ID。\n\
· 提取小地图与完整图：从网络下载资源\n\
· 寻路：可选复刻版 mini_game 或正式「冒险岛怀旧服」窗口，再点「开始寻路」\n\
· 键盘默认走 RP2040 USB 虚拟键盘（可在下方切换为 SendInput）\n"
                .into(),
            busy: false,
            tx: Some(tx),
            rx,
            #[cfg(windows)]
            nav_running: false,
            #[cfg(windows)]
            nav_stop: None,
            #[cfg(windows)]
            nav_status: "寻路未运行".into(),
            #[cfg(windows)]
            capture_target: win_capture::CaptureTarget::MiniGame,
            #[cfg(windows)]
            capture_detect: "尚未检测窗口".into(),
            #[cfg(windows)]
            kb_backend: keyboard_input::KeyboardBackend::UsbHid,
            #[cfg(windows)]
            usb_port: String::new(),
            #[cfg(windows)]
            kb_session: None,
            #[cfg(windows)]
            kb_status: "未连接".into(),
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
                #[cfg(windows)]
                JobResult::Nav(ev) => match ev {
                    live_nav::LiveNavEvent::Log(s) => {
                        let line = format!("[寻路] {s}");
                        eprintln!("{line}");
                        self.append_log(line);
                    }
                    live_nav::LiveNavEvent::Status(s) => {
                        eprintln!("[寻路状态] {s}");
                        self.nav_status = s;
                    }
                    live_nav::LiveNavEvent::Stopped { reason } => {
                        self.nav_running = false;
                        self.nav_stop = None;
                        self.nav_status = reason.clone();
                        let line = format!("[寻路] {reason}");
                        eprintln!("{line}");
                        self.append_log(line);
                        // 寻路释放串口后，若仍选 USB 可重新连接供测试
                        if self.kb_backend == keyboard_input::KeyboardBackend::UsbHid {
                            self.ensure_kb_session();
                        }
                    }
                },
            }
        }
    }

    #[cfg(windows)]
    fn kb_config(&self) -> keyboard_input::KeyboardConfig {
        keyboard_input::KeyboardConfig {
            backend: self.kb_backend,
            usb_port: self.usb_port.clone(),
        }
    }

    #[cfg(windows)]
    fn drop_kb_session(&mut self) {
        if let Some(mut s) = self.kb_session.take() {
            let _ = s.clear_all();
        }
        if self.kb_backend == keyboard_input::KeyboardBackend::UsbHid {
            self.kb_status = "未连接（串口已释放）".into();
        }
    }

    #[cfg(windows)]
    fn ensure_kb_session(&mut self) {
        if self.nav_running {
            self.kb_status = "寻路占用中".into();
            return;
        }
        if self.kb_backend == keyboard_input::KeyboardBackend::SendInput {
            self.kb_session = None;
            self.kb_status = "SendInput（无需连接）".into();
            return;
        }
        match keyboard_input::KeyboardSession::open(&self.kb_config()) {
            Ok(s) => {
                self.kb_status = format!("已连接 {}", s.describe());
                self.append_log(format!("键盘：{}", self.kb_status));
                self.kb_session = Some(s);
            }
            Err(e) => {
                self.kb_session = None;
                self.kb_status = format!("连接失败：{e}");
                self.append_log(format!("键盘连接失败：{e}"));
            }
        }
    }

    #[cfg(windows)]
    fn detect_capture_window(&mut self) {
        let target = self.capture_target;
        match win_capture::find_game_window(target) {
            Some(w) => {
                let msg = format!("已找到：{}", w.short_desc());
                self.capture_detect = msg.clone();
                self.append_log(format!("[窗口] {msg}"));
            }
            None => {
                let candidates = win_capture::list_candidate_windows(target);
                let msg = if candidates.is_empty() {
                    match target {
                        win_capture::CaptureTarget::MiniGame => {
                            format!("未找到「{}」；请先 cargo run --bin mini_game", target.title_hint())
                        }
                        win_capture::CaptureTarget::ClassicClient => format!(
                            "未找到「{}」/Maplestory_Classic；请确认已启动正式客户端",
                            target.title_hint()
                        ),
                    }
                } else {
                    format!(
                        "候选 {} 个但未选中最佳匹配：{}",
                        candidates.len(),
                        candidates
                            .iter()
                            .map(|w| w.short_desc())
                            .collect::<Vec<_>>()
                            .join(" | ")
                    )
                };
                self.capture_detect = msg.clone();
                self.append_log(format!("[窗口] {msg}"));
            }
        }
    }

    #[cfg(windows)]
    fn start_nav(&mut self) {
        if self.nav_running {
            self.append_log("寻路已在运行");
            return;
        }
        let Some(tx0) = self.tx.clone() else {
            return;
        };
        // USB 串口同一时间只能被一方占用
        self.drop_kb_session();
        let kb = self.kb_config();
        let capture = self.capture_target;
        let tx = tx0;
        let stop = Arc::new(AtomicBool::new(false));
        self.nav_stop = Some(stop.clone());
        self.nav_running = true;
        self.nav_status = "启动中…".into();
        self.append_log(format!(
            "[寻路] 启动后台线程…（目标：{}；键盘：{}）",
            capture.label(),
            kb.backend.label()
        ));
        thread::Builder::new()
            .name("live-nav".into())
            .spawn(move || {
                let (nav_tx, nav_rx) = mpsc::channel();
                let bridge = tx.clone();
                let bridge_thread = thread::spawn(move || {
                    while let Ok(ev) = nav_rx.recv() {
                        let stop_msg = matches!(ev, live_nav::LiveNavEvent::Stopped { .. });
                        let _ = bridge.send(JobResult::Nav(ev));
                        if stop_msg {
                            break;
                        }
                    }
                });
                live_nav::run_live_nav(
                    stop,
                    nav_tx,
                    live_nav::LiveNavConfig { kb, capture },
                );
                let _ = bridge_thread.join();
            })
            .expect("spawn live-nav");
    }

    #[cfg(windows)]
    fn stop_nav(&mut self) {
        if let Some(stop) = &self.nav_stop {
            stop.store(true, Ordering::SeqCst);
            self.append_log("[寻路] 正在停止…");
            self.nav_status = "正在停止…".into();
        } else {
            self.append_log("寻路未在运行");
        }
    }

    #[cfg(windows)]
    fn send_key(&mut self, modifiers: &[keyboard_input::Modifier], key: keyboard_input::Key) {
        if self.nav_running {
            self.append_log("寻路运行中，请先结束寻路再测键");
            return;
        }
        if self.kb_backend == keyboard_input::KeyboardBackend::UsbHid && self.kb_session.is_none() {
            self.ensure_kb_session();
        }
        let result = if let Some(session) = self.kb_session.as_mut() {
            session.tap(modifiers, key)
        } else {
            keyboard_input::tap(&self.kb_config(), modifiers, key)
        };
        match result {
            Ok(()) => self.append_log(format!(
                "已发送按键：{}（{}）",
                key.label(),
                self.kb_backend.label()
            )),
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
        if self.nav_running {
            self.append_log("寻路运行中，请先结束寻路再测键");
            return;
        }
        if self.kb_backend == keyboard_input::KeyboardBackend::UsbHid && self.kb_session.is_none() {
            self.ensure_kb_session();
        }
        let result = if let Some(session) = self.kb_session.as_mut() {
            session.tap_keys(modifiers, keys)
        } else {
            keyboard_input::tap_keys(&self.kb_config(), modifiers, keys)
        };
        match result {
            Ok(()) => self.append_log(format!(
                "已发送组合键：{label}（{}）",
                self.kb_backend.label()
            )),
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
        if self.nav_running {
            ctx.request_repaint_after(std::time::Duration::from_millis(200));
        }

        #[cfg(windows)]
        no_activate::sync_from_handle(frame);

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("mxd经典版工具");
            ui.label(format!("工作目录：{}", self.root.display()));

            #[cfg(windows)]
            {
                let mut no_steal = no_activate::enabled();
                if ui
                    .checkbox(&mut no_steal, "点击不夺取焦点（类似屏幕键盘）")
                    .on_hover_text("开启后点本窗口按钮不会抢走其他程序焦点；寻路时请保持小游戏在前台")
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
                ui.heading("NavBot 实时寻路");
                ui.label(
                    "附着游戏窗口：实时截图 → YOLO+SelfTracker → NavBot → 键盘注入。可切换复刻版 / 正式客户端。",
                );
                ui.add_enabled_ui(!self.nav_running, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("截图目标");
                        let mut changed = false;
                        changed |= ui
                            .radio_value(
                                &mut self.capture_target,
                                win_capture::CaptureTarget::MiniGame,
                                win_capture::CaptureTarget::MiniGame.label(),
                            )
                            .changed();
                        changed |= ui
                            .radio_value(
                                &mut self.capture_target,
                                win_capture::CaptureTarget::ClassicClient,
                                win_capture::CaptureTarget::ClassicClient.label(),
                            )
                            .changed();
                        if changed {
                            self.capture_detect = "目标已切换，请点「检测窗口」".into();
                        }
                        if ui.button("检测窗口").clicked() {
                            self.detect_capture_window();
                        }
                    });
                });
                ui.label(format!("窗口：{}", self.capture_detect));
                ui.label(format!("状态：{}", self.nav_status));
                ui.horizontal(|ui| {
                    ui.add_enabled_ui(!self.nav_running, |ui| {
                        if ui
                            .button("开始寻路")
                            .on_hover_text("先选择截图目标并确保对应窗口已打开")
                            .clicked()
                        {
                            self.start_nav();
                        }
                    });
                    ui.add_enabled_ui(self.nav_running, |ui| {
                        if ui.button("结束寻路").clicked() {
                            self.stop_nav();
                        }
                    });
                });
            }

            #[cfg(windows)]
            {
                ui.separator();
                ui.heading("键盘注入方式");
                ui.horizontal(|ui| {
                    let mut changed = false;
                    changed |= ui
                        .radio_value(
                            &mut self.kb_backend,
                            keyboard_input::KeyboardBackend::UsbHid,
                            keyboard_input::KeyboardBackend::UsbHid.label(),
                        )
                        .changed();
                    changed |= ui
                        .radio_value(
                            &mut self.kb_backend,
                            keyboard_input::KeyboardBackend::SendInput,
                            keyboard_input::KeyboardBackend::SendInput.label(),
                        )
                        .changed();
                    if changed {
                        self.kb_session = None;
                        if self.kb_backend == keyboard_input::KeyboardBackend::UsbHid {
                            self.ensure_kb_session();
                        } else {
                            self.kb_status = "SendInput（无需连接）".into();
                        }
                    }
                });

                if self.kb_backend == keyboard_input::KeyboardBackend::UsbHid {
                    ui.horizontal(|ui| {
                        ui.label("串口");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.usb_port)
                                .desired_width(120.0)
                                .hint_text("空=自动 VID/PID"),
                        );
                        if ui.button("刷新口").clicked() {
                            let ports = usb_hid::list_ports_for_ui();
                            if ports.is_empty() {
                                self.append_log("未枚举到串口");
                            } else {
                                let line = ports
                                    .iter()
                                    .map(|(n, hit)| {
                                        if *hit {
                                            format!("{n}*")
                                        } else {
                                            n.clone()
                                        }
                                    })
                                    .collect::<Vec<_>>()
                                    .join(", ");
                                self.append_log(format!("串口列表（* = RP2040）：{line}"));
                                if self.usb_port.is_empty() {
                                    if let Some((name, _)) = ports.iter().find(|(_, hit)| *hit) {
                                        self.usb_port = name.clone();
                                    }
                                }
                            }
                        }
                        if ui
                            .add_enabled(!self.nav_running, egui::Button::new("连接/重连"))
                            .clicked()
                        {
                            self.kb_session = None;
                            self.ensure_kb_session();
                        }
                        if ui
                            .add_enabled(
                                !self.nav_running && self.kb_session.is_some(),
                                egui::Button::new("Ping"),
                            )
                            .clicked()
                        {
                            if let Some(s) = self.kb_session.as_mut() {
                                match s.ping() {
                                    Ok(()) => self.append_log("USB Ping OK (pong)"),
                                    Err(e) => self.append_log(format!("USB Ping 失败：{e}")),
                                }
                            }
                        }
                    });
                    ui.label(format!("USB 状态：{}", self.kb_status));
                }

                ui.separator();
                ui.heading("键盘输入测试");
                ui.label(match self.kb_backend {
                    keyboard_input::KeyboardBackend::UsbHid => {
                        "通过 RP2040 USB HID 注入到当前前台窗口（请先连接设备，并把焦点放在小游戏上）。"
                    }
                    keyboard_input::KeyboardBackend::SendInput => {
                        "SendInput 注入到当前前台窗口。"
                    }
                });
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
                    if ui.button("J 攻击").clicked() {
                        self.send_key(&[], Key::J);
                    }
                    if ui.button("Z 拾取").clicked() {
                        self.send_key(&[], Key::Z);
                    }
                });
            }

            #[cfg(windows)]
            {
                ui.separator();
                ui.heading("RP2040 固件（USB 虚拟键盘）");
                ui.label(format!(
                    "已嵌入 UF2：{}（约 {:.1} KB）。烧写前请按住 BOOTSEL 插入 Pico，直到出现 RPI-RP2 盘。",
                    firmware_flash::EMBEDDED_UF2_NAME,
                    firmware_flash::embedded_size_kb()
                ));
                let bootsel = firmware_flash::find_bootsel_drives();
                if bootsel.is_empty() {
                    ui.colored_label(
                        egui::Color32::from_rgb(180, 120, 40),
                        "当前未检测到 RPI-RP2 烧录盘",
                    );
                } else {
                    let list = bootsel
                        .iter()
                        .map(|p| p.display().to_string())
                        .collect::<Vec<_>>()
                        .join(", ");
                    ui.colored_label(
                        egui::Color32::from_rgb(40, 140, 70),
                        format!("已检测到烧录盘：{list}"),
                    );
                }
                ui.horizontal_wrapped(|ui| {
                    if ui
                        .button("刷新检测")
                        .on_hover_text("重新扫描 RPI-RP2 / INFO_UF2.TXT")
                        .clicked()
                    {
                        let drives = firmware_flash::find_bootsel_drives();
                        if drives.is_empty() {
                            self.append_log("固件：未检测到 RPI-RP2（请 BOOTSEL 模式插入）");
                        } else {
                            self.append_log(format!(
                                "固件：检测到 {}",
                                drives
                                    .iter()
                                    .map(|p| p.display().to_string())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            ));
                        }
                    }
                    if ui
                        .button("自动烧写到 RPI-RP2")
                        .on_hover_text("将嵌入的 UF2 拷到 BOOTSEL 盘；写入后设备会重启")
                        .clicked()
                    {
                        self.drop_kb_session();
                        match firmware_flash::flash_embedded_to_bootsel() {
                            Ok(msg) => self.append_log(format!("固件：{msg}")),
                            Err(e) => self.append_log(format!("固件烧写失败：{e}")),
                        }
                    }
                    if ui
                        .button("固件另存为…")
                        .on_hover_text("导出 UF2 到本地，可手动拖进 RPI-RP2 盘")
                        .clicked()
                    {
                        if let Some(path) = rfd::FileDialog::new()
                            .set_title("保存 RP2040 UF2 固件")
                            .set_file_name(firmware_flash::EMBEDDED_UF2_NAME)
                            .add_filter("UF2", &["uf2"])
                            .save_file()
                        {
                            match firmware_flash::save_uf2_to(&path) {
                                Ok(()) => self.append_log(format!(
                                    "固件已另存为：{}\n可手动拷贝到 RPI-RP2 盘完成烧写。",
                                    path.display()
                                )),
                                Err(e) => self.append_log(format!("固件另存为失败：{e}")),
                            }
                        }
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

    #[cfg(windows)]
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        if let Some(stop) = &self.nav_stop {
            stop.store(true, Ordering::SeqCst);
        }
        // 先松开 USB/SendInput 按键，再关串口（固件侧仍有断开清键兜底）
        self.drop_kb_session();
    }
}

fn load_app_icon() -> egui::IconData {
    let bytes = include_bytes!("../assets/app_icon.png");
    let image = image::load_from_memory(bytes)
        .expect("加载 app_icon.png")
        .into_rgba8();
    let (width, height) = image.dimensions();
    egui::IconData {
        rgba: image.into_raw(),
        width,
        height,
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([820.0, 820.0])
            .with_title("mxd经典版工具")
            .with_icon(load_app_icon()),
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
