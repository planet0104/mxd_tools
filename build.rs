// 嵌入的 UF2 变更时触发重编译。
// 重新生成：powershell -File scripts/build_rp2040_uf2.ps1

fn main() {
    println!("cargo:rerun-if-changed=firmware/mxd-usb-hid.uf2");
    println!("cargo:rerun-if-changed=scripts/build_rp2040_uf2.ps1");
    println!("cargo:rerun-if-changed=assets/app_icon.ico");
    println!("cargo:rerun-if-changed=assets/app_icon.png");

    let uf2 = std::path::Path::new("firmware/mxd-usb-hid.uf2");
    if !uf2.is_file() {
        println!(
            "cargo:warning=缺少 firmware/mxd-usb-hid.uf2。请先运行: powershell -File scripts/build_rp2040_uf2.ps1"
        );
    }

    #[cfg(windows)]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/app_icon.ico");
        if let Err(e) = res.compile() {
            println!("cargo:warning=嵌入 exe 图标失败: {e}");
        }
    }
}
