# Anime Denoise Vulkan

在本机处理动漫图像中的鱼鳞状纹理。默认推荐档为 60% 预缩放、Real-ESRGAN Anime Vulkan 超分和 L0 0.001 平滑。项目基于 [CXP-2024/gpt-anime-denoise](https://github.com/CXP-2024/gpt-anime-denoise)，保留原网页界面的对比、批量队列、保护笔刷和磁性套索。

## Windows 安装

1. 从 [GitHub Releases](https://github.com/hhsbjbsj/anime-denoise-vulkan/releases/latest) 下载 `AnimeDenoiseVulkan-1.0.0-windows-x64.msi`。
2. 双击安装。在桌面或开始菜单打开 **Anime Denoise Vulkan**，浏览器会打开本机界面。
3. 将图片拖进页面，选择「推荐」。处理结果可从页面下载，也会保存到 `%LOCALAPPDATA%\AnimeDenoiseVulkan\outputs`。点击页面右上角「结束」可关闭后台。

需要 Windows 10/11 x64 和支持 Vulkan 的显卡驱动。MSI 内含 Python、NumPy、Pillow、Vulkan 推理程序与动漫模型，安装和处理可离线完成。安装包目前未签名。可在 Windows「已安装的应用」中卸载，已生成的图片保留在输出目录。

## Vulkan 和内存优化

- 按可用 Windows 提交额度和预缩放后的尺寸，在 64 与 112 分块之间选择。Vulkan 报告内存分配失败时会回退到 64。
- CPU L0 平滑复用 FFT 缓冲区，限制 Windows 下空闲的 BLAS 线程。
- 一台 GTX 1660 上，同一组五张约 1.57 MP 图片以推荐档测试，Vulkan 平均从 8.36 秒降至 7.26 秒。Vulkan 进程峰值提交内存从 735 MiB 增至 961 MiB，输出 RGB 通道平均绝对差为 0.37/255。速度取决于设备、驱动和当时的负载。

## 从源码运行

安装 Python 3.11 或更新版，运行 `python setup_env.py` 安装所需依赖，再用虚拟环境的 Python 运行 `denoise_app.py`。命令行选项见 `python denoise.py --help`。

源码版如需启用 Vulkan，请将 [Real-ESRGAN Windows 官方包](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.5.0) 解压到 `.denoise-tools/realesrgan-ncnn-vulkan/`，让可执行文件和 `models/` 同级。没有该目录时，源码版使用 PyTorch 路径。

## 构建 MSI

在 Windows x64 上安装 Python 和 [.NET SDK](https://dotnet.microsoft.com/download)。运行 `dotnet tool install wix --version 5.0.2 --tool-path build/wix`，再运行 `python packaging/build_windows.py`。构建脚本会校验下载的依赖，安装包和 SHA-256 校验文件输出到 `dist/`。

## 许可与致谢

应用代码采用 [MIT](LICENSE)，原项目作者与贡献者见 [上游致谢](https://github.com/CXP-2024/gpt-anime-denoise#致谢)。本项目基于上游提交 [41bc9b4](https://github.com/CXP-2024/gpt-anime-denoise/commit/41bc9b4)。安装包的第三方组件许可随包提供，来源见 [第三方声明](packaging/THIRD_PARTY.md)。
