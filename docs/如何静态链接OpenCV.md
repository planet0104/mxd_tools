# Rust + 静态 OpenCV 对接说明

本仓库定位逻辑调用 **OpenCV**（`matchTemplate` / `resize` / mask，以及画布→大图的 **SIFT + FLANN + findHomography**），与 `scripts/locate_from_screencaps.py` 对齐。

Rust crate features：`imgproc`、`imgcodecs`、`features2d`、`calib3d`、`flann`。链接库需包含对应模块（见下 `OPENCV_LINK_LIBS`）。

为与 **静态链接的 ONNX Runtime（`ort`）** 共存、实现「单 exe 部署」，默认使用：

- vcpkg triplet：`x64-windows-static-md`（OpenCV 静态进 exe，CRT 用 `/MD`）
- **不要**再开 `rustflags = +crt-static`（`/MT` 会与 pyke 预编译 ort 冲突）

## 一次性安装（Windows）

1. 安装 LLVM（bindgen 需要 libclang）  
   `winget install -e --id LLVM.LLVM`

2. 安装静态 OpenCV（久，可能 1～3 小时）  
   ```powershell
   cd mxd_tools
   powershell -ExecutionPolicy Bypass -File scripts/setup_opencv_static.ps1
   ```
   默认安装 `opencv4:x64-windows-static-md`。

3. 核对 `.cargo/config.toml`  
   - 路径指向 `%USERPROFILE%\vcpkg\installed\x64-windows-static-md\...`  
   - `OPENCV_MSVC_CRT=dynamic`  
   - `OPENCV_LINK_LIBS` 与 `scripts/list_opencv_libs.ps1` 一致（注意 zlib 可能是 `zlib` 而非旧 triplet 的 `zs`）  
   - 无 `+crt-static` rustflags

4. 编译  
   ```powershell
   cargo build --release
   cargo run --release --bin validate_screen_caps -- `
     --caps screen_caps/彩虹岛-南港西郊平原 `
     --minimap assets/maps/50001/map_50001_minimap.png `
     --full assets/maps/50001/map_50001_render_cn.png
   ```

YOLO 推理见 `docs/如何在Rust中做YOLO推理.md`（ORT 同样静态链进 exe）。

## App 按钮

「验证截图定位（OpenCV）」走同一套实现，输出到 `tmp/screen_cap_locate/`。算法说明见 `docs/如何从截图定位玩家.md`。

## 注意

- 静态包体积大，首次 vcpkg 编译很慢，属正常。  
- 若链接报缺库，用 `list_opencv_libs.ps1` 补全 `OPENCV_LINK_LIBS`。  
- 旧方案 `x64-windows-static`（`/MT`）仍可用，但无法直接静态链接 pyke ort；仅 OpenCV、不要 YOLO 时才考虑。  
- 开发机临时想动态链接 OpenCV：改用 `x64-windows` triplet，并设置 `VCPKGRS_DYNAMIC=1`（目标机需带 DLL）。
