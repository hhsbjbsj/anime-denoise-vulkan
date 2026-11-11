"""One-click dependency install for gpt-anime-denoise. Needs only the Python standard library.

    python setup_env.py                 # detect the graphics card, show the plan, ask, install into .venv
    python setup_env.py --kind cpu      # force a kind: auto / cuda / rocm / mps / cpu
    python setup_env.py --yes           # don't ask

What it picks:
- NVIDIA: the CUDA build of PyTorch. Only the graphics driver is needed, not the CUDA Toolkit, because the wheels
  carry their own CUDA runtime. It takes the newest build on download.pytorch.org that the driver can run.
- AMD (Windows): AMD's ROCm 7.2.1 for Windows wheels, for the cards AMD lists (RX 7000 / 9000, Radeon PRO W7000 /
  AI PRO, Ryzen AI Max). They exist only for Python 3.12; when none is installed, uv downloads a private copy into
  .denoise-tools/ without touching PATH, the registry or the user profile.
- Apple Silicon (macOS): the ordinary PyPI wheel of PyTorch already carries Metal (MPS), so nothing extra is
  downloaded. An Intel Mac, or a Python running under Rosetta, has no Metal device and gets the CPU build.
- Anything else: the CPU build.
Everything goes into .venv next to this file, then torch is imported and asked for the GPU to check the result.
"""
import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
TOOLS = ROOT / ".denoise-tools"
WINDOWS = os.name == "nt"
MACOS = sys.platform == "darwin"
PYTHON = VENV / ("Scripts/python.exe" if WINDOWS else "bin/python")
NO_WINDOW = subprocess.CREATE_NO_WINDOW if WINDOWS else 0

TORCH_INDEX = "https://download.pytorch.org/whl"
CUDA_FALLBACK = [134, 132, 130, 126]  # used only when the index page can't be read
ROCM_BASE = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1"
ROCM_SDK = ["rocm_sdk_core-7.2.1-py3-none-win_amd64.whl", "rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl",
            "rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl", "rocm-7.2.1.tar.gz"]
# torchvision comes from the same place as torch: spandrel needs it, and the newest one on PyPI would replace
# this torch with a CPU build of another version.
ROCM_TORCH = ["torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl", "torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"]
ROCM_GPU = re.compile(r"RX\s*[79]\d{3}|Radeon\s*(AI\s*)?PRO\s*(W7\d{3}|R9\d{3})|Radeon\s*80[56]0S", re.I)
DRIVER_PAGE = {"cuda": "https://www.nvidia.com/Download/index.aspx",
               "rocm": "https://www.amd.com/en/support/download/drivers.html"}
SIZE = {"cuda": "约 2–3 GB", "rocm": "约 2.2 GB（另下约 20 MB 的 Python 3.12）",
        "mps": "约 300 MB", "cpu": "约 200 MB"}
NAME = {"cuda": "NVIDIA CUDA", "rocm": "AMD ROCm 7.2.1", "mps": "Apple Metal (MPS)", "cpu": "CPU"}

CHECK = r"""
import json, torch, spandrel, numpy, PIL
ok = torch.cuda.is_available()
mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
print("CHECK" + json.dumps({"torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
                            "cuda": torch.version.cuda, "mps": mps,
                            "gpu": torch.cuda.get_device_name(0) if ok else None}))
"""


def capture(cmd, timeout=60):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8",
                              errors="replace", creationflags=NO_WINDOW).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def run(cmd, env=None):
    print("  > " + " ".join(str(c) for c in cmd[:5]) + (" ..." if len(cmd) > 5 else ""), flush=True)
    return subprocess.run([str(c) for c in cmd], env=env).returncode == 0


def pip(*args):
    return run([PYTHON, "-m", "pip", "install", "--disable-pip-version-check", *args])


def apple_silicon():
    """True on an arm64 Mac; False on Windows, Linux or an Intel Mac."""
    return MACOS and platform.machine() == "arm64"


def gpu_names():
    if MACOS:
        out = capture(["system_profiler", "SPDisplaysDataType"])
        found = [x.split(":", 1)[1].strip() for x in out.splitlines()
                 if x.strip().startswith(("Chipset Model", "Chip:"))]
        return [x for x in found if x][:4]
    if WINDOWS:
        out = capture(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"])
    elif shutil.which("lspci"):
        out = "\n".join(x.split(": ", 1)[-1] for x in capture(["lspci"]).splitlines() if "VGA" in x or "3D" in x)
    else:
        out = ""
    return [x.strip() for x in out.splitlines() if x.strip()]


def detect():
    """Apple Silicon first (macOS has no CUDA), then NVIDIA over AMD, because a computer with both renders on NVIDIA."""
    names = gpu_names()
    if apple_silicon():
        return {"kind": "mps", "gpu": names[0] if names else "Apple Silicon", "names": names,
                "note": "Apple GPU，用 Metal 加速；PyTorch 的普通 wheel 自带 MPS"}
    if MACOS:  # Intel Mac: MPS support there is thin and these conv stacks gain little from it
        return {"kind": "cpu", "gpu": None, "names": names, "note": "Intel Mac 没有可用的 MPS，使用 CPU"}
    smi = capture(["nvidia-smi"]) if shutil.which("nvidia-smi") else ""
    nvidia = next((n for n in names if "nvidia" in n.lower()), None)
    if smi or nvidia:
        m = re.search(r"CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)", smi)
        driver = f"{m[1]}.{m[2]}" if m else None
        if driver:
            note = f"驱动最高支持 CUDA {driver}"
        elif not smi:  # an NVIDIA card but no nvidia-smi output: usually the driver is not installed yet
            note = "没有检测到 NVIDIA 驱动：环境照样装，装好驱动后才能用显卡"
        else:  # nvidia-smi ran, but its header has a layout this script doesn't know
            note = "无法读取驱动支持的 CUDA 版本：环境照样装，先装最新的 CUDA 版，装完会检查能不能用上显卡"
        return {"kind": "cuda", "gpu": nvidia or "NVIDIA", "driver_cuda": driver, "names": names, "note": note}
    amd = next((n for n in names if ROCM_GPU.search(n)), None)
    if amd and WINDOWS:
        return {"kind": "rocm", "gpu": amd, "names": names, "note": "需要较新的 AMD 驱动（AMD 为 ROCm 7.2.1 指定 26.2.2）"}
    other = next((n for n in names if re.search(r"radeon|arc|intel", n, re.I)), None)
    note = f"{other} 不在支持列表里，使用 CPU" if other else "没有可用的独立显卡，使用 CPU"
    return {"kind": "cpu", "gpu": None, "names": names, "note": note}


def cuda_builds(py_tag, driver_cuda):
    """CUDA builds of torch for this Python and OS, newest first, starting at the newest the driver can run."""
    platform = "win_amd64" if WINDOWS else "linux_x86_64"
    try:
        page = urllib.request.urlopen(f"{TORCH_INDEX}/torch/", timeout=60).read().decode("utf-8", "replace")
        found = {int(x) for x in re.findall(rf"torch-[\d.]+(?:%2B|\+)cu(\d+)-{py_tag}-{py_tag}-{platform}\.whl", page)}
    except OSError:
        found = set()
    builds = sorted(found or CUDA_FALLBACK, reverse=True)
    if driver_cuda:
        major, minor = map(int, driver_cuda.split("."))
        fits = [b for b in builds if b <= major * 10 + minor]
        builds = fits + [b for b in builds if b not in fits] if fits else sorted(builds)  # too old: oldest first
    return [f"cu{b}" for b in builds]


def venv_version():
    return capture([str(PYTHON), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"]).strip() or None


def fresh_venv_if(wrong):
    if VENV.exists() and wrong:
        print(f"现有 .venv 的 Python 版本不对（{venv_version()}），重建。", flush=True)
        shutil.rmtree(VENV)


def python312_venv():
    """.venv on Python 3.12: an installed 3.12 if there is one, otherwise a private copy downloaded by uv."""
    fresh_venv_if(venv_version() != "3.12")
    if PYTHON.exists():
        return
    found = capture(["py", "-3.12", "-c", "import sys; print(sys.executable)"]).strip() if shutil.which("py") else ""
    if found and Path(found).exists():
        print(f"使用已安装的 Python 3.12：{found}", flush=True)
        if not run([found, "-m", "venv", VENV]):
            raise SystemExit("创建虚拟环境失败。")
        return
    uv = TOOLS / "uv" / "bin" / "uv.exe"
    if not uv.exists():
        print("下载 uv，用来获取独立的 Python 3.12（放在 .denoise-tools，不改系统）...", flush=True)
        if not run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--target", TOOLS / "uv", "uv"]):
            raise SystemExit("uv 下载失败。")
    env = {**os.environ, "UV_CACHE_DIR": str(TOOLS / "uv-cache"), "UV_PYTHON_INSTALL_DIR": str(TOOLS / "python"),
           "UV_PYTHON_INSTALL_BIN": "0", "UV_PYTHON_INSTALL_REGISTRY": "0", "UV_NO_CONFIG": "1"}
    print("下载 Python 3.12 并创建 .venv ...", flush=True)
    if not run([uv, "venv", "--seed", "--python", "3.12", "--python-preference", "only-managed", VENV], env):
        raise SystemExit("Python 3.12 下载失败。")


def other_torch(index):
    """The installed torch's version when it is a build for another index, else None.

    pip counts torch 2.x+cpu as already satisfying a request for the CUDA build, so install() must replace it.
    Ask torch for its version: pip records PyPI's wheels without the build tag (the Windows one, a CPU build, is
    plain 2.x to pip but 2.x+cpu to torch). pip's record is the fallback when torch does not import.
    """
    have = (capture([str(PYTHON), "-c", "import torch; print(torch.__version__)"], timeout=300).strip()
            or capture([str(PYTHON), "-c", "import importlib.metadata as m; print(m.version('torch'))"]).strip())
    return have if have and "+" in have and not have.endswith("+" + index) else None


def install(kind, info):
    """Installs into .venv; returns the PyTorch index used ("cpu" when every CUDA build failed), or None."""
    index = None
    if kind == "rocm":
        if not WINDOWS:
            raise SystemExit("这里的 ROCm 安装只针对 Windows。Linux 请按 https://pytorch.org 的 ROCm 安装命令来装。")
        python312_venv()
        print("\n安装 AMD ROCm 运行库（约 1.4 GB）...", flush=True)
        if not pip("--no-cache-dir", *[f"{ROCM_BASE}/{f}" for f in ROCM_SDK]):
            raise SystemExit("ROCm 运行库安装失败，见上面的信息。")
        print("\n安装 ROCm 版 PyTorch（约 820 MB）...", flush=True)
        if not pip("--no-cache-dir", *[f"{ROCM_BASE}/{f}" for f in ROCM_TORCH]):
            raise SystemExit("ROCm 版 PyTorch 安装失败，见上面的信息。")
    else:
        if not PYTHON.exists():
            print(f"创建 {VENV} ...", flush=True)
            if not run([sys.executable, "-m", "venv", VENV]):
                raise SystemExit("创建虚拟环境失败。")
        tag = "cp" + (venv_version() or "").replace(".", "")
        if kind == "cuda":
            indexes = cuda_builds(tag, info.get("driver_cuda")) + ["cpu"]
            print("可用的 CUDA 版本（按优先顺序）：" + ", ".join(indexes[:-1]), flush=True)
        elif kind == "mps":
            indexes = [None]  # the PyPI wheel already carries Metal on macOS, no extra index needed
        else:
            indexes = ["cpu"]
        for index in indexes:
            if index is None:
                print("\n安装 PyTorch（PyPI，自带 MPS）...", flush=True)
                if pip("torch", "torchvision"):
                    break
            else:
                url = f"{TORCH_INDEX}/{index}"
                print(f"\n安装 PyTorch（{index}）...", flush=True)
                other = other_torch(index)
                if other:
                    # pip removes the old build only after the new wheels are downloaded, so a failed download
                    # leaves the working torch in place. The plain install below adds what the new build depends on.
                    print(f"已装的 torch {other} 不是 {index} 版本，下载好新版本后替换。", flush=True)
                    if not pip("--force-reinstall", "--no-deps", "torch", "torchvision", "--index-url", url):
                        continue
                if pip("torch", "torchvision", "--index-url", url):
                    break
        else:
            raise SystemExit("PyTorch 安装失败，见上面的信息。")
    print("\n安装其余依赖 ...", flush=True)
    if not pip("-r", ROOT / "requirements.txt"):
        raise SystemExit("依赖安装失败，见上面的信息。")
    return index


def main():
    ap = argparse.ArgumentParser(description="Install the dependencies into .venv for this computer's graphics card.")
    ap.add_argument("--kind", choices=["auto", "cuda", "rocm", "mps", "cpu"], default="auto")
    ap.add_argument("--yes", action="store_true", help="install without asking")
    a = ap.parse_args()

    info = detect()
    kind = info["kind"] if a.kind == "auto" else a.kind
    print("gpt-anime-denoise 一键安装依赖\n")
    print("显卡：" + ("、".join(info["names"]) or "没有识别到"))
    print(f"方案：{NAME[kind]}" + (f"（{info['note']}）" if a.kind == "auto" else "（手动指定）"))
    print(f"下载：{SIZE[kind]}")
    print(f"位置：{VENV}（只装在这里，不改系统；删掉这个文件夹即可卸载）\n")
    if not a.yes:
        try:
            if input("按回车开始安装，输入 n 取消：").strip().lower() in ("n", "no"):
                raise SystemExit("已取消。")
        except EOFError:
            raise SystemExit("没有收到确认。加 --yes 可以不询问直接安装。")

    index = install(kind, info)

    out = capture([str(PYTHON), "-c", CHECK], timeout=300)
    line = next((x for x in out.splitlines() if x.startswith("CHECK")), None)
    if not line:
        raise SystemExit("装完了，但依赖无法导入，见上面的信息。")
    r = json.loads(line[5:])
    backend = (f"ROCm {r['hip']}" if r["hip"] else f"CUDA {r['cuda']}" if r["cuda"]
               else "Metal / MPS" if r.get("mps") else "CPU")
    print(f"\n完成：torch {r['torch']} · {backend} · " + (f"显卡 {r['gpu']}" if r["gpu"] else "使用 CPU"))
    if kind == "cuda" and index == "cpu":  # no CUDA build could be installed, so the driver is not the cause
        print("没能装上 CUDA 版 PyTorch，原因见上面 pip 的信息。现在用的是 CPU 版，解决后再运行一次 setup_env.py。")
    elif kind in ("cuda", "rocm") and not r["gpu"]:
        print(f"环境可以用，但现在只能跑 CPU。请更新显卡驱动（{DRIVER_PAGE[kind]}），重启后再打开。")
    if kind == "mps" and not r.get("mps"):
        print("环境可以用，但没拿到 MPS 设备（是 arm64 的 Python 吗？Rosetta 下的 Python 没有 Metal）。现在跑 CPU。")


if __name__ == "__main__":
    main()
