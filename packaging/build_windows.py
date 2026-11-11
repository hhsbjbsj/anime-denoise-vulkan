"""Build the offline Windows x64 MSI with WiX 5.0.2."""

import hashlib
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = ROOT / "build" / "downloads"
VERSION = "1.0.0"

ARCHIVES = {
    "python.zip": (
        "https://www.python.org/ftp/python/3.13.17/python-3.13.17-embed-amd64.zip",
        "9e35319bffedab8b36ad44d8f377cb4f1733bcec257a0b6d04d94f01486ddb38",
    ),
    "realesrgan.zip": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
        "realesrgan-ncnn-vulkan-20220424-windows.zip",
        "abc02804e17982a3be33675e4d471e91ea374e65b70167abc09e31acb412802d",
    ),
    "numpy.whl": (
        "https://files.pythonhosted.org/packages/b5/cd/9cc4dc876fb065d5c220aae4d5e14826b2715331bb7618ce1fb07a679d99/"
        "numpy-2.4.6-cp313-cp313-win_amd64.whl",
        "c4fc99836233ea196540b17ab0983aff60ed07941751930f5f4d05bc3b3b7359",
    ),
    "pillow.whl": (
        "https://files.pythonhosted.org/packages/d7/ed/1b16fb66a37e1b9e41370b93b2c614c7da761a5b2a41dc333f7af4639962/"
        "pillow-13.0.0-cp313-cp313-win_amd64.whl",
        "9dda0788a9932776a75d8bcc020076edc308ce2508b51f6739a45e3158b3fc42",
    ),
}

LICENSES = {
    "real-esrgan.txt": "xinntao/Real-ESRGAN/v0.2.5.0/LICENSE",
    "real-esrgan-ncnn-vulkan.txt": "xinntao/Real-ESRGAN-ncnn-vulkan/v0.2.0/LICENSE",
    "ncnn.txt": "Tencent/ncnn/6125c9f47cd14b589de0521350668cf9d3d37e3c/LICENSE.txt",
    "glslang.txt": "KhronosGroup/glslang/4afd69177258d0636f78d2c4efb823ab6382a187/LICENSE.txt",
    "libwebp.txt": "webmproject/libwebp/8ea81561d2fdd382da60f57958741a7c23a18eb6/COPYING",
    "libwebp-patents.txt": "webmproject/libwebp/8ea81561d2fdd382da60f57958741a7c23a18eb6/PATENTS",
}


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def download(url, target):
    temporary = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as file:
        shutil.copyfileobj(response, file)
    temporary.replace(target)


def archive(name):
    url, digest = ARCHIVES[name]
    path = DOWNLOADS / name
    if not path.is_file() or sha256(path) != digest:
        download(url, path)
    if sha256(path) != digest:
        raise ValueError(f"SHA-256 mismatch: {name}")
    return ZipFile(path)


def stage(payload):
    for name in ("denoise.py", "denoise_app.py", "LICENSE", "README.md"):
        shutil.copy2(ROOT / name, payload)
    shutil.copytree(ROOT / "app", payload / "app")
    shutil.copy2(ROOT / "packaging/start_windows.py", payload)
    shutil.copy2(ROOT / "packaging/THIRD_PARTY.md", payload)

    runtime = payload / "runtime"
    with archive("python.zip") as zf:
        zf.extractall(runtime)
    (runtime / "python313._pth").write_text("python313.zip\n.\nLib/site-packages\n", encoding="ascii")
    for wheel in ("numpy.whl", "pillow.whl"):
        with archive(wheel) as zf:
            zf.extractall(runtime / "Lib/site-packages")

    vulkan = payload / ".denoise-tools" / "realesrgan-ncnn-vulkan"
    with archive("realesrgan.zip") as zf:
        zf.extractall(vulkan, (
            "realesrgan-ncnn-vulkan.exe",
            "vcomp140.dll",
            "models/realesrgan-x4plus-anime.bin",
            "models/realesrgan-x4plus-anime.param",
        ))
    licenses = payload / "licenses"
    licenses.mkdir()
    for name, source in LICENSES.items():
        download("https://raw.githubusercontent.com/" + source, licenses / name)
    with urllib.request.urlopen(
        "https://raw.githubusercontent.com/xinntao/Real-ESRGAN-ncnn-vulkan/v0.2.0/src/stb_image.h",
        timeout=60,
    ) as response:
        source = response.read().decode("utf-8")
    (licenses / "stb.txt").write_text(
        source[source.index("ALTERNATIVE A - MIT License"):source.index("ALTERNATIVE B - Public Domain")],
        encoding="utf-8",
    )


def main():
    wix = ROOT / "build/wix/wix.exe"
    if not wix.is_file():
        raise SystemExit("Install WiX as described in README.md")
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    output = ROOT / f"dist/AnimeDenoiseVulkan-{VERSION}-windows-x64.msi"
    output.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="payload-", dir=ROOT / "build") as directory:
        payload = Path(directory)
        stage(payload)
        subprocess.run([
            str(wix), "build", "-arch", "x64", "-d", f"Version={VERSION}", "-d", f"Payload={payload}",
            "-pdbtype", "none", "-o", str(output), str(ROOT / "packaging/windows.wxs"),
        ], check=True)
    output.with_suffix(".msi.sha256").write_text(f"{sha256(output)}  {output.name}\n", encoding="ascii")
    print(output)


if __name__ == "__main__":
    main()
