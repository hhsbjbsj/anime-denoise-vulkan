# Third-party software in the Windows installer

- [CPython 3.13.17](https://www.python.org/downloads/release/python-31317/), with the PSF and bundled component notices in `runtime/LICENSE.txt`.
- [NumPy 2.4.6](https://pypi.org/project/numpy/2.4.6/) and [Pillow 13.0.0](https://pypi.org/project/pillow/13.0.0/), with their wheel license notices in `runtime/Lib/site-packages/*dist-info/`.
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.5.0), the `realesrgan-x4plus-anime` model and its [ncnn Vulkan inference program](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/tree/v0.2.0), both BSD-3-Clause. Their notices, plus notices for [ncnn](https://github.com/Tencent/ncnn), [glslang](https://github.com/KhronosGroup/glslang), and [stb](https://github.com/nothings/stb), are in `licenses/`.
- Microsoft Visual C++ OpenMP runtime `vcomp140.dll` is the release DLL shipped in the official Real-ESRGAN Windows package. See [Microsoft's redistributable component list](https://learn.microsoft.com/en-us/visualstudio/releases/2022/redistribution).
