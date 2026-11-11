"""Remove the fish-scale texture that GPT image models (gpt-image-2 / 2.5 / 2.6, ChatGPT, Codex image_gen) leave in
anime and TV-anime style images.

Default method (recommended): downscale to 60% -> Real-ESRGAN x4plus anime 6B -> resize back -> L0 smoothing 0.001.
With an NVIDIA GPU both steps run on it (about 2 s for a 1.6 MP image); without one everything runs on the CPU.

  python denoise.py image.png [more.png ...]            # default method only
  python denoise.py image.png --all                      # all 7 candidates + a 4x2 comparison sheet
  python denoise.py ./folder --out ./cleaned             # every image in a folder
  python denoise.py image.png --l0 0.002                 # stronger L0 (0.001 is only the recommended value)

Outputs keep the input size. Models are downloaded on first use into --models (default ~/.cache/gpt-anime-denoise).
"""
import argparse
import ctypes
import hashlib
import io
import math
import os
import re
import subprocess
import tempfile
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path

# AMD ROCm: by default MIOpen benchmarks every convolution algorithm the first time it meets a new input size,
# single-threaded on the CPU while the GPU sits idle. On an RX 7700 XT that is ~90 s per new image size for the
# ESRGAN layers. FAST picks a kernel by heuristics instead (~0.3 s); each pass is a few ms slower, which is noise
# next to that. Must be set before torch loads MIOpen; has no effect on NVIDIA or CPU. An explicit value wins.
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
# Bound the cuDNN execution-plan cache across image and tile shapes.
if os.name == "nt":
    os.environ.setdefault("TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT", "128")
    if __name__ == "__main__":
        # The CPU denoiser uses pocketfft and elementwise operations, not BLAS.
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from PIL import Image, ImageCms, ImageDraw, ImageFont, ImageOps  # noqa: E402

MODELS = {
    "esrgan": ("RealESRGAN_x4plus_anime_6B.pth",
               "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"),
    "apisr": ("2x_APISR_RRDB_GAN_generator.pth",
              "https://huggingface.co/camenduru/APISR/resolve/main/2x_APISR_RRDB_GAN_generator.pth"),
    "cugan": ("up2x-latest-denoise3x.pth",
              "https://github.com/bilibili/ailab/releases/download/Real-CUGAN/updated_weights.zip"),
}
SHA256 = {
    "RealESRGAN_x4plus_anime_6B.pth": "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da",
    "2x_APISR_RRDB_GAN_generator.pth": "3b0d2b3a3c0461ac17d00f4f32240666fb832b738ea5a48449b1acf07fbb07e5",
    "up2x-latest-denoise3x.pth": "0a14739f3f5fcbd74ec3ce2806d13a47916c916b20afe4a39d95f6df4ca6abd8",
}
DOWN = 0.6
L0 = 0.001  # recommended default; tune with --l0 (larger = flatter, too large wipes soft blush/gradients)
METHODS = ["esrgan60_l0", "esrgan60", "cugan60_l0", "cugan60", "apisr", "l0_0.001", "l0_0.0005"]
LABELS = {"original": "original", "l0_0.0005": "L0 0.0005", "l0_0.001": "L0 0.001", "apisr": "APISR",
          "cugan60": "CUGAN @60%", "cugan60_l0": "CUGAN @60% + L0 {lam:g}",
          "esrgan60": "ESRGAN @60%", "esrgan60_l0": "ESRGAN @60% + L0 {lam:g} (default)"}
SHEET_ORDER = ["original", "l0_0.0005", "l0_0.001", "apisr", "cugan60", "cugan60_l0", "esrgan60", "esrgan60_l0"]


def fetch(key, folder):
    name, url = MODELS[key]
    path = folder / name
    if path.exists():
        return path
    folder.mkdir(parents=True, exist_ok=True)
    print(f"downloading {name} ...", flush=True)
    data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "gpt-anime-denoise"})).read()
    if url.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            data = z.read(next(n for n in z.namelist() if n.endswith(name)))
    if hashlib.sha256(data).hexdigest() != SHA256[name]:
        raise SystemExit(f"checksum mismatch for {name} from {url}; not saved")
    path.write_bytes(data)
    return path


@lru_cache(maxsize=1)
def load_torch():
    import torch
    # TF32 rounding depends on tile shape and can introduce visible seams.
    torch.backends.cudnn.allow_tf32 = False
    return torch


def load_model(key, folder, device):
    import spandrel
    return spandrel.ModelLoader().load_from_file(str(fetch(key, folder))).model.to(device).eval()


def default_device():
    """CUDA, else Apple Metal (MPS), else the CPU; --device still overrides."""
    torch = load_torch()
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _is_oom(err):
    """CUDA raises its own OutOfMemoryError; MPS raises a RuntimeError that says the same thing."""
    return isinstance(err, (MemoryError, load_torch().cuda.OutOfMemoryError)) or "out of memory" in str(err).lower()


def _clear(device):
    """Release the backend's cached blocks after a big step (a no-op on the CPU)."""
    if str(device).startswith("cuda"):
        load_torch().cuda.empty_cache()
    elif str(device).startswith("mps"):
        load_torch().mps.empty_cache()


class Cancelled(Exception):
    """Raised when the optional stop() callback asks a long step to give up early."""


def _check(stop):
    if stop is not None and stop():
        raise Cancelled()


def _forward_differences(plane, horizontal, vertical):
    np.subtract(plane[:, 1:], plane[:, :-1], out=horizontal[:, :-1])
    np.subtract(plane[:, 0], plane[:, -1], out=horizontal[:, -1])
    np.subtract(plane[1:], plane[:-1], out=vertical[:-1])
    np.subtract(plane[0], plane[-1], out=vertical[-1])


FFT_OUTPUT_BUFFERS = np.lib.NumpyVersion(np.__version__) >= "2.0.0"


def _fft_into(transform, source, target, axis, pool, **options):
    if pool is None:
        transform(source, axis=axis, out=target, **options)
        return
    # Split only along the other axis, so each worker owns disjoint input and output lines.
    length = source.shape[1 - axis]
    step = (length + 2) // 3

    def transform_lines(start):
        lines = slice(start, start + step)
        index = (slice(None), lines) if axis == 0 else (lines, slice(None))
        transform(source[index], axis=axis, out=target[index], **options)

    pending = [pool.submit(transform_lines, start) for start in range(step, length, step)]
    try:
        transform_lines(0)
    finally:
        for future in pending:
            future.result()


def _rfft_into(plane, spectrum, pool):
    _fft_into(np.fft.rfft, plane, spectrum, 1, pool)
    _fft_into(np.fft.fft, spectrum, spectrum, 0, pool)


def l0_smooth(img, lam, kappa=2.0, beta_max=1e5, device="cpu", stop=None, *, threaded_cpu=False):
    """L0 gradient minimization (Xu et al. 2011) on an HxWx3 float image in [0, 1].

    Flattens small ripples inside flat colour areas while keeping lines and colour edges; invents no new detail.
    CUDA and threaded CPU mode use float64 with torch.fft. MPS uses float32.
    uint8 input is normalized during conversion to avoid an intermediate RGB float array.
    stop: optional callable checked every iteration; when it returns True, Cancelled is raised.
    """
    if str(device).startswith(("cuda", "mps")) or threaded_cpu:
        try:
            return _l0_torch(img, lam, kappa, beta_max, device, stop)
        except Exception as err:
            if not _is_oom(err):
                raise
        _clear(device)  # very large image: fall back to the NumPy path
    S = np.array(img.transpose(2, 0, 1), dtype=np.float64, order="C", copy=True)
    if img.dtype == np.uint8:
        S /= 255
    _, height, width = S.shape
    parallel = FFT_OUTPUT_BUFFERS and height * width >= 2**20 and min(height, width) >= 64 and (os.cpu_count() or 1) >= 3
    with (ThreadPoolExecutor(max_workers=2, thread_name_prefix="l0-fft") if parallel else nullcontext()) as pool:
        return _l0_numpy(S, lam, kappa, beta_max, stop, pool)


def _l0_numpy(S, lam, kappa, beta_max, stop, pool):
    _, H, W = S.shape
    # A forward difference has squared frequency response 4 sin²(pi f).
    fx = 4 * np.sin(np.pi * np.fft.rfftfreq(W)) ** 2
    fy = 4 * np.sin(np.pi * np.fft.fftfreq(H)) ** 2
    otf = fy[:, None] + fx
    fft_buffer = None
    if FFT_OUTPUT_BUFFERS:
        # Both axes share a complex output buffer. The inverse writes directly into the image plane.
        N1 = np.empty((S.shape[0], H, W // 2 + 1), dtype=np.complex128)
        for plane, frequency in zip(S, N1):
            _rfft_into(plane, frequency, pool)
        fft_buffer = np.empty_like(N1[0])
    else:
        N1 = np.fft.rfft2(S)
    h, v, energy = (np.empty((H, W), dtype=np.float64) for _ in range(3))
    quiet = np.empty((H, W), dtype=bool)
    denominator = np.empty_like(otf)
    beta = 2 * lam
    while beta < beta_max:
        _check(stop)
        # The gradient mask depends on all colors, but each Fourier solve is independent.
        # Reuse one channel's scratch buffers and update each plane in place.
        _forward_differences(S[0], energy, v)
        np.square(energy, out=energy)
        np.square(v, out=v)
        energy += v
        for plane in S[1:]:
            _forward_differences(plane, h, v)
            np.square(h, out=h)
            np.square(v, out=v)
            h += v
            energy += h
        np.less(energy, lam / beta, out=quiet)
        np.multiply(otf, beta, out=denominator)
        denominator += 1
        np.reciprocal(denominator, out=denominator)
        for index, plane in enumerate(S):
            _check(stop)
            _forward_differences(plane, h, v)
            np.copyto(h, 0, where=quiet)
            np.copyto(v, 0, where=quiet)
            # Backward differences form the divergence. Reuse energy and h as scratch space.
            np.subtract(h[:, :-1], h[:, 1:], out=energy[:, 1:])
            np.subtract(h[:, -1], h[:, 0], out=energy[:, 0])
            np.subtract(v[:-1], v[1:], out=h[1:])
            np.subtract(v[-1], v[0], out=h[0])
            energy += h
            if fft_buffer is None:
                spectrum = np.fft.rfft2(energy)
            else:
                _rfft_into(energy, fft_buffer, pool)
                spectrum = fft_buffer
            spectrum *= beta
            spectrum += N1[index]
            spectrum *= denominator
            if fft_buffer is None:
                plane[:] = np.fft.irfft2(spectrum, s=(H, W), axes=(-2, -1))
            else:
                _fft_into(np.fft.ifft, spectrum, spectrum, 0, pool)
                _fft_into(np.fft.irfft, spectrum, plane, 1, pool, n=W)
            del spectrum
        beta *= kappa
    np.clip(S, 0, 1, out=S)
    return S.transpose(1, 2, 0)


def _l0_torch(img, lam, kappa, beta_max, device, stop=None):
    torch = load_torch()
    dt = torch.float32 if str(device).startswith("mps") else torch.float64
    # Pillow's NumPy view can be read-only. Copy it before passing to PyTorch.
    source = np.array(img, order="C") if not img.flags.writeable else np.ascontiguousarray(img)
    S = torch.from_numpy(source).permute(2, 0, 1)
    S = S.to(device=device, dtype=dt, memory_format=torch.contiguous_format)
    if img.dtype == np.uint8:
        S.div_(255)
    _, H, W = S.shape
    fx = 4 * (torch.pi * torch.fft.rfftfreq(W, dtype=dt, device=device)).sin().square()
    fy = 4 * (torch.pi * torch.fft.fftfreq(H, dtype=dt, device=device)).sin().square()
    otf = fy[:, None] + fx
    N1 = torch.fft.rfft2(S)
    beta = 2 * lam
    while beta < beta_max:
        _check(stop)
        h = torch.roll(S, -1, 2) - S
        v = torch.roll(S, -1, 1) - S
        del S
        t = h.square().add_(v.square()).sum(0, keepdim=True) < lam / beta
        h.masked_fill_(t, 0)
        v.masked_fill_(t, 0)
        div = torch.roll(h, 1, 2).sub_(h).add_(torch.roll(v, 1, 1).sub_(v))
        del h, v, t
        spectrum = torch.fft.rfft2(div)
        del div
        spectrum.mul_(beta).add_(N1).div_(1 + beta * otf)
        S = torch.fft.irfft2(spectrum, s=(H, W))
        del spectrum
        beta *= kappa
    out = S.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    del S, N1, otf
    return out


def to_arr(im, *, normalized=True):
    rgb = im if im.mode == "RGB" else im.convert("RGB")
    if not normalized:
        return np.asarray(rgb)
    pixels = np.asarray(rgb, dtype=np.float64)
    pixels /= 255
    return pixels


def to_img(a):
    return Image.fromarray((a * 255 + 0.5).astype(np.uint8))


def _upscale_whole(model, im, device):
    torch = load_torch()
    t = torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1)[None].to(device)
    with torch.no_grad():
        y = model(t).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((y * 255 + 0.5).astype(np.uint8))


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".jfif", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".avif", ".tga")


def open_image(src):
    """Open an image the way a viewer shows it; returns (rgb, alpha or None, icc profile bytes or None).

    EXIF rotation is applied, 16-bit / 32-bit greyscale is scaled to 8-bit (a plain convert would clip it to white),
    CMYK is converted through its embedded profile when it has one, and transparency is kept separately.
    The ICC profile is returned only for RGB sources, where it still describes the pixel values.
    """
    im = Image.open(src)
    im.load()
    ImageOps.exif_transpose(im, in_place=True)
    icc = im.info.get("icc_profile")
    mode = im.mode
    if mode in ("I;16", "I;16B", "I;16L", "I;16N", "I", "F"):
        a = np.asarray(im, np.float64)
        scale = 257.0 if a.max() > 255 else 1.0
        im = Image.fromarray(np.clip(a / scale + 0.5, 0, 255).astype(np.uint8), "L")
    alpha = None
    if im.mode in ("RGBA", "LA", "PA", "La", "RGBa") or "transparency" in im.info:
        a = im.convert("RGBA").getchannel("A")
        if a.getextrema() != (255, 255):  # only keep alpha that is actually used
            alpha = a
    if im.mode == "CMYK" and icc:
        try:
            im = ImageCms.profileToProfile(im, ImageCms.ImageCmsProfile(io.BytesIO(icc)),
                                           ImageCms.createProfile("sRGB"), outputMode="RGB")
        except Exception:
            pass
    if mode not in ("RGB", "RGBA", "RGBX", "P"):
        icc = None  # a grey or CMYK profile does not describe the RGB we produce
    return (im if im.mode == "RGB" else im.convert("RGB")), alpha, icc


def save_png(im, path, alpha=None, icc=None, **kw):
    """Save as PNG with the source's alpha and colour profile put back."""
    if alpha is not None:
        im = im.copy(); im.putalpha(alpha)
    if icc:
        kw["icc_profile"] = icc
    im.save(path, "PNG", **kw)


# Exact memory-bounded super-resolution. Not tiling with guessed borders: every band carries enough real
# neighbour rows for the layers it runs, so each output pixel sees exactly the inputs a whole-image pass would.
# Measured on a 614x922 input: the trunk peaks at 1.8 GB, the two 3x3 convs at 4x size peak at 8.7 GB (2.2 GB
# input + 2.2 GB output + cuDNN workspace). So the trunk runs whole whenever it fits, the 4x tail runs in bands.
HEAD_HALO = 96      # input rows: the trunk stacks ~92 3x3 convs (6 RRDB x 3 RDB x 5 + first/last conv)
TAIL_HALO = 3       # input rows: the tail is 4 3x3 convs around two nearest 2x upsamples (needs 2)
HEAD_BYTES = 3500   # per input pixel for the trunk (1.8 GB / 0.57 MP, plus headroom)
TAIL_BYTES = 16000  # per input pixel of a tail band (64 channels at 4x: input + output + conv workspace)


def _free_bytes(device):
    if str(device).startswith("cuda"):
        return load_torch().cuda.mem_get_info(device)[0]
    if str(device).startswith("mps"):
        try:
            torch = load_torch()
            return torch.mps.recommended_max_memory() - torch.mps.driver_allocated_memory()
        except Exception:  # older PyTorch lacks recommended_max_memory: Apple GPUs share system RAM, so use that
            pass
    try:
        import psutil
        return psutil.virtual_memory().available
    except Exception:
        return 4 << 30


def _rrdb_split(model):
    """(trunk, tail) of a plain RRDB ESRGAN, split at its first upsample; None for other architectures."""
    torch = load_torch()
    seq = getattr(model, "model", None)
    if getattr(model, "shuffle_factor", None) or not isinstance(seq, torch.nn.Sequential):
        return None
    ups = [i for i, m in enumerate(seq) if isinstance(m, torch.nn.Upsample)]
    return (seq[:ups[0]], seq[ups[0]:]) if ups else None


def _bands(n, step, halo):
    for r0 in range(0, n, step):
        r1 = min(n, r0 + step)
        yield r0, r1, max(0, r0 - halo), min(n, r1 + halo)


def _run_tiles(fn, width, height, budget, bytes_per_pixel, halo, stop, device):
    """Use strips across the shorter side, or 2D tiles when a strip would have too much overlap."""
    pixels = max(1, int(budget / bytes_per_pixel))
    short_side = min(width, height)
    step = pixels // short_side - 2 * halo
    if step >= max(8, halo // 2):
        cols, rows = (width, min(height, step)) if width <= height else (min(width, step), height)
    else:
        side = max(8, math.isqrt(pixels) - 2 * halo)
        cols, rows = min(width, side), min(height, side)
    while True:
        try:
            for y0, y1, top, bottom in _bands(height, rows, halo):
                for x0, x1, left, right in _bands(width, cols, halo):
                    _check(stop)
                    fn((x0, y0, x1, y1), (left, top, right, bottom))
            return
        except Exception as err:
            if not _is_oom(err) or max(rows, cols) <= 8:
                raise
        # Tracebacks can hold references to tile tensors until the except block ends.
        _clear(device)
        if rows >= cols:
            rows = max(8, rows // 2)
        else:
            cols = max(8, cols // 2)


def upscale(model, im, device, stop=None, *, memory_budget=None):
    """Super-resolve with the result of a single whole-image pass but bounded GPU memory / RAM (see above)."""
    torch = load_torch()
    split = _rrdb_split(model)
    if split is None:  # other models (APISR, CUGAN in --all): whole image
        return _upscale_whole(model, im, device)
    trunk, tail = split
    W, H = im.size
    x = torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1)[None]
    with torch.no_grad():
        budget = min(_free_bytes(device) * 0.6, memory_budget or float("inf"))
        f = None
        if H * W * HEAD_BYTES <= budget:
            try:
                f = trunk(x.to(device))
            except Exception as err:
                if not _is_oom(err):
                    raise
        if f is None:  # large image or failed whole pass: gather exact bands on the CPU
            _clear(device)
            f = torch.empty((1, 64, H, W), dtype=torch.float32)
            def head_tile(box, padded):
                x0, y0, x1, y1 = box
                left, top, right, bottom = padded
                tile = trunk(x[:, :, top:bottom, left:right].to(device))
                f[:, :, y0:y1, x0:x1] = tile[:, :, y0 - top:y1 - top, x0 - left:x1 - left].cpu()
            _run_tiles(head_tile, W, H, budget, HEAD_BYTES, HEAD_HALO, stop, device)
        out = {}
        def tail_tile(box, padded):
            x0, y0, x1, y1 = box
            left, top, right, bottom = padded
            y = tail(f[:, :, top:bottom, left:right].to(device)).clamp_(0, 1)
            s = y.shape[2] // (bottom - top)
            if "img" not in out:
                out["img"] = np.empty((H * s, W * s, 3), np.uint8)
            y = y[0, :, (y0 - top) * s:(y1 - top) * s, (x0 - left) * s:(x1 - left) * s]
            rgb = y.permute(1, 2, 0).cpu().numpy()
            out["img"][y0 * s:y1 * s, x0 * s:x1 * s] = (rgb * 255 + 0.5).astype(np.uint8)
        budget = min(_free_bytes(device) * 0.6, memory_budget or float("inf"))
        _run_tiles(tail_tile, W, H, budget, TAIL_BYTES, TAIL_HALO, stop, device)
    del f
    _clear(device)
    return Image.fromarray(out["img"])


def sr_at(model, src, device, stop=None, *, down=None, memory_budget=None):
    """Downscale (box filter averages the fine texture away), upscale with the model, resize back to the input size."""
    down = DOWN if down is None else down
    W, H = src.size
    small = src.resize((round(W * down), round(H * down)), Image.BOX)
    return upscale(model, small, device, stop=stop, memory_budget=memory_budget).resize((W, H), Image.LANCZOS)


VULKAN_LOW_TILE = 64


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_uint32),
        ("memory_load", ctypes.c_uint32),
        ("total_phys", ctypes.c_uint64),
        ("avail_phys", ctypes.c_uint64),
        ("total_page_file", ctypes.c_uint64),
        ("avail_page_file", ctypes.c_uint64),
        ("total_virtual", ctypes.c_uint64),
        ("avail_virtual", ctypes.c_uint64),
        ("avail_extended_virtual", ctypes.c_uint64),
    ]


def _vulkan_tile_size(size):
    width, height = size
    # Keep larger images on the low-memory tile. The native worker also allocates
    # a full 4x output and staging buffers proportional to the image width.
    if os.name != "nt" or max(size) > 1024 or width * height > 640_000:
        return VULKAN_LOW_TILE
    status = _MemoryStatusEx()
    status.length = ctypes.sizeof(status)
    # Check available Windows commit since it can run low even with free RAM.
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) and status.avail_page_file >= 2 * 1024**3:
        return 112
    return VULKAN_LOW_TILE


def _run_vulkan_worker(command, deadline, stop):
    _check(stop)
    if time.monotonic() >= deadline:
        raise RuntimeError("Vulkan 超分超过 5 分钟")
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          creationflags=subprocess.CREATE_NO_WINDOW) as worker:
        try:
            while time.monotonic() < deadline:
                _check(stop)
                try:
                    log, _ = worker.communicate(timeout=0.15)
                    return worker.returncode, log
                except subprocess.TimeoutExpired:
                    continue
            raise RuntimeError("Vulkan 超分超过 5 分钟")
        finally:
            if worker.poll() is None:
                worker.kill()
                worker.communicate()


def sr_vulkan(executable, src, down, stop=None):
    """Run the upstream Real-ESRGAN Vulkan build without loading the CUDA runtime."""
    _check(stop)
    small = src.resize(tuple(round(n * down) for n in src.size), Image.BOX)
    with tempfile.TemporaryDirectory(prefix="anime-denoise-") as tmp:
        source = Path(tmp) / "input.png"
        output = Path(tmp) / "output.png"
        small.save(source, compress_level=1)
        tile = _vulkan_tile_size(small.size)
        command = [str(executable), "-i", str(source), "-o", str(output),
                   "-m", str(executable.parent / "models"), "-n", "realesrgan-x4plus-anime",
                   "-s", "4", "-j", "1:1:1"]
        deadline = time.monotonic() + 300
        tiles = (tile,) if tile == VULKAN_LOW_TILE else (tile, VULKAN_LOW_TILE)
        for attempt_tile in tiles:
            code, log = _run_vulkan_worker([*command, "-t", str(attempt_tile)], deadline, stop)
            # Available system commit does not account for the Vulkan heap budget.
            allocation_error = re.search(rb"vk(?:AllocateMemory|CreateBuffer|CreateImage) failed -[12]\b", log)
            if allocation_error:
                if attempt_tile == VULKAN_LOW_TILE:
                    raise RuntimeError(f"Vulkan 超分失败：{allocation_error[0].decode()}")
                output.unlink(missing_ok=True)
                continue
            if code != 0 or not output.is_file():
                lines = log.decode(errors="replace").strip().splitlines()
                raise RuntimeError(f"Vulkan 超分失败：{lines[-1] if lines else code}")
            break
        with Image.open(output) as enlarged:
            if enlarged.size != tuple(n * 4 for n in small.size):
                raise RuntimeError("Vulkan 超分输出尺寸异常")
            rgb = enlarged if enlarged.mode == "RGB" else enlarged.convert("RGB")
            if rgb.getbbox() is None and any(hi > 16 for _, hi in small.getextrema()):
                raise RuntimeError("Vulkan 超分输出了异常的全黑图像")
            return rgb.resize(src.size, Image.LANCZOS)


def run(src, methods, models, device, lam=L0):
    W, H = src.size
    out, cache = {}, {}
    for m in methods:
        if m.startswith("l0_"):
            out[m] = to_img(l0_smooth(to_arr(src), float(m[3:]), device=device))
        elif m == "apisr":
            out[m] = upscale(models["apisr"], src, device).resize((W, H), Image.LANCZOS)
        else:
            key = m.split("60")[0]
            if key not in cache:
                cache[key] = sr_at(models[key], src, device)
            use_l0 = m.endswith("_l0") and lam > 0  # --l0 0 skips L0, as the web app does
            out[m] = to_img(l0_smooth(to_arr(cache[key]), lam, device=device)) if use_l0 else cache[key]
    return out


def sheet(src, res, path, lam=L0):
    items = [("original", src)] + [(k, res[k]) for k in SHEET_ORDER[1:]]
    w, h = src.size
    g, top, cols = 20, max(40, w // 18), 4
    s = Image.new("RGB", (cols * (w + g) - g, 2 * (h + top + g)), (60, 60, 60))
    d = ImageDraw.Draw(s)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", top // 2)
    except OSError:
        try:
            font = ImageFont.truetype("arial.ttf", top // 2)
        except OSError:
            font = ImageFont.load_default()
    for i, (k, im) in enumerate(items):
        x, y = (i % cols) * (w + g), (i // cols) * (h + top + g)
        s.paste(im, (x, y + top))
        d.text((x + 12, y + top // 5), LABELS[k].format(lam=lam), fill="white", font=font)
    s.save(path)


def l0_strength(text):
    """--l0 value. 0 skips L0; below 0 the L0 loop would never end (beta keeps doubling away from beta_max)."""
    lam = float(text)
    if not 0 <= lam < float("inf"):  # also rejects nan and inf
        raise argparse.ArgumentTypeError(f"must be a finite number, 0 or more (0 skips L0), got {text}")
    return lam


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="images or folders")
    ap.add_argument("--out", help="output folder (default: <image folder>/<stem>_denoise)")
    ap.add_argument("--all", action="store_true", help="write all 7 candidates and a comparison sheet")
    ap.add_argument("--method", default="esrgan60_l0", choices=METHODS)
    ap.add_argument("--l0", type=l0_strength, default=L0,
                    help="L0 strength for the *_l0 methods (default 0.001; try 0.0005-0.002; 0 skips L0)")
    ap.add_argument("--models", default=str(Path.home() / ".cache" / "gpt-anime-denoise"))
    ap.add_argument("--device", default=default_device())
    a = ap.parse_args()

    files = []
    for p in map(Path, a.inputs):
        files += sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS) if p.is_dir() else [p]
    methods = METHODS if a.all else [a.method]
    need = {m.split("60")[0] if "60" in m else m for m in methods if not m.startswith("l0_")}
    models = {k: load_model(k, Path(a.models), a.device) for k in need}

    for p in files:
        src, alpha, icc = open_image(p)
        out = Path(a.out) if a.out else p.parent / f"{p.stem}_denoise"
        out.mkdir(parents=True, exist_ok=True)
        res = run(src, methods, models, a.device, a.l0)
        for k, im in res.items():
            save_png(im, out / f"{p.stem}_{k}.png", alpha, icc)
        if a.all:
            sheet(src, res, out / f"{p.stem}_sheet.png", a.l0)
        print(f"{p} -> {out}", flush=True)


if __name__ == "__main__":
    main()
