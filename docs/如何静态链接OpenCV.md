# Rust + 静态 OpenCV 对接说明

本仓库定位逻辑调用 **OpenCV**（`matchTemplate` / `resize` / mask，以及画布→大图的 **SIFT + FLANN + findHomography**），与 `scripts/locate_from_screencaps.py` 对齐。

Rust crate features：`imgproc`、`imgcodecs`、`features2d`、`calib3d`、`flann`。链接库需包含对应模块（见下 `OPENCV_LINK_LIBS`）。

## 一次性安装（Windows）

1. 安装 LLVM（bindgen 需要 libclang）  
   `winget install -e --id LLVM.LLVM`

2. 安装静态 OpenCV（久，可能 1～3 小时）  
   ```powershell
   cd mxd_tools
   powershell -ExecutionPolicy Bypass -File scripts/setup_opencv_static.ps1
   ```

3. 核对 `.cargo/config.toml`  
   - `OPENCV_INCLUDE_PATHS` / `OPENCV_LINK_PATHS` 指向  
     `%USERPROFILE%\vcpkg\installed\x64-windows-static\...`  
   - `OPENCV_LINK_LIBS` 与 `scripts/list_opencv_libs.ps1` 列出的库名一致（版本后缀可能是 `4`）  
   - 定位相关模块至少包含：`opencv_imgcodecs4`、`opencv_features2d4`、`opencv_calib3d4`、`opencv_flann4`、`opencv_imgproc4`、`opencv_core4` 及编解码依赖  
   - `OPENCV_MSVC_CRT=static` + `crt-static`：目标机一般**不需要** `opencv_world*.dll`

4. 编译  
   ```powershell
   cargo build --release
   cargo run --release --bin validate_screen_caps -- `
     --caps screen_caps/彩虹岛-南港西郊平原 `
     --minimap assets/maps/50001/map_50001_minimap.png `
     --full assets/maps/50001/map_50001_render_cn.png
   ```

## App 按钮

「验证截图定位（OpenCV）」走同一套实现，输出到 `tmp/screen_cap_locate/`。算法说明见 `docs/如何从截图定位玩家.md`。

## 注意

- 静态包体积大，首次 vcpkg 编译很慢，属正常。  
- 若链接报缺库，用 `list_opencv_libs.ps1` 补全 `OPENCV_LINK_LIBS`。  
- 开发机临时想动态链接：改用 `x64-windows` triplet，并设置 `VCPKGRS_DYNAMIC=1`（目标机需带 DLL）。
