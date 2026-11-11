"""Start the installed Windows edition with its own port and output folder."""

import ctypes
import os
import runpy
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    output = Path(os.environ["LOCALAPPDATA"]) / "AnimeDenoiseVulkan" / "outputs"
    os.environ.setdefault("ANIME_DENOISE_OUTPUT", str(output))
    os.environ.setdefault("ANIME_DENOISE_PORT", "7866")
    sys.dont_write_bytecode = True
    try:
        runpy.run_path(str(root / "denoise_app.py"), run_name="__main__")
    except Exception as error:
        if sys.stderr is not None:
            raise
        ctypes.windll.user32.MessageBoxW(None, str(error), "Anime Denoise Vulkan", 0x10)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
