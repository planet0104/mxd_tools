# Rust + 静态 OpenCV 对接说明

本仓库 **YOLO 预处理** 使用 OpenCV（`resize` 等）。安装方式与 YOLO 推理相同，见 `docs/如何在Rust中做YOLO推理.md`。

Rust crate features：`imgproc`、`imgcodecs`。链接库需包含对应模块（见下 `OPENCV_LINK_LIBS`）。

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

4. 编译验证  
   ```powershell
   cargo build --release --bin game_preview
   cargo run --release --bin game_preview
   ```

玩家自身定位见 `SelfTracker`（YOLO「玩家」框 + 运动残差跟踪）。YOLO 推理细节见 `docs/如何在Rust中做YOLO推理.md`。

## 注意

- 静态包体积大，首次 vcpkg 编译很慢，属正常。  
- 若链接报缺库，用 `list_opencv_libs.ps1` 补全 `OPENCV_LINK_LIBS`。  
- 旧方案 `x64-windows-static`（`/MT`）仍可用，但无法直接静态链接 pyke ort；仅 OpenCV、不要 YOLO 时才考虑。  
- 开发机临时想动态链接 OpenCV：改用 `x64-windows` triplet，并设置 `VCPKGRS_DYNAMIC=1`（目标机需带 DLL）。
