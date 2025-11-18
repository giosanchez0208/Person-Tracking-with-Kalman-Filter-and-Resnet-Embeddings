from __future__ import annotations
import os
import sys
import subprocess
import shutil
from pathlib import Path

"""
    Creates a virtual environment in ./venv
    Installs packages from requirements.txt into the venv
    Downloads yolov11n.pt and yolo11n-seg.pt into ./model/
"""

import urllib.request

VENV_DIR = Path("venv")
MODEL_DIR = Path("model")
REQ_FILE = Path("requirements.txt")

DEFAULT_YOLO11N_URL = os.environ.get(
    "YOLOV8_URL",
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
)

DEFAULT_YOLO11N_SEG_URL = os.environ.get(
    "YOLOV8_SEG_URL",
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt",
)



def run(cmd, check=True, **kwargs):
        print("Running:", " ".join(cmd))
        return subprocess.run(cmd, check=check, **kwargs)


def create_venv(venv_path: Path):
        if venv_path.exists():
                print(f"{venv_path} already exists — skipping venv creation.")
                return
        print("Creating virtual environment:", venv_path)
        run([sys.executable, "-m", "venv", str(venv_path)])


def get_venv_python(venv_path: Path) -> Path:
        if os.name == "nt":
                return venv_path / "Scripts" / "python.exe"
        else:
                return venv_path / "bin" / "python"


def pip_install_requirements(python_exe: Path, requirements: Path):
        if not requirements.exists():
                print(f"No {requirements} found — skipping pip install.")
                return
        print("Upgrading pip/setuptools in venv...")
        run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip", "setuptools"])
        print(f"Installing packages from {requirements} ...")
        run([str(python_exe), "-m", "pip", "install", "-r", str(requirements)])


def download_file(url: str, dest: Path):
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        print(f"Downloading {url} -> {dest}")
        try:
                with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
                        total = resp.getheader("Content-Length")
                        total = int(total) if total and total.isdigit() else None
                        chunk_size = 8192
                        downloaded = 0
                        while True:
                                chunk = resp.read(chunk_size)
                                if not chunk:
                                        break
                                out.write(chunk)
                                downloaded += len(chunk)
                                if total:
                                        pct = downloaded * 100 // total
                                        print(f"\r{downloaded}/{total} bytes ({pct}%)", end="")
                if total:
                        print()
                tmp.replace(dest)
                print(f"Saved: {dest}")
        except Exception as e:
                if tmp.exists():
                        tmp.unlink()
                print(f"Failed to download {url}: {e}")
                raise


def main():
        # 1) Create venv
        create_venv(VENV_DIR)
        python_exe = get_venv_python(VENV_DIR)
        if not python_exe.exists():
                print(f"Error: cannot find python in venv at {python_exe}")
                sys.exit(1)

        # 2) Install requirements
        pip_install_requirements(python_exe, REQ_FILE)

        # 3) Create model dir and download models
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        yolo1_url = DEFAULT_YOLO11N_URL
        yolo2_url = DEFAULT_YOLO11N_SEG_URL

        yolo1_dest = MODEL_DIR / "yolov11n.pt"
        yolo2_dest = MODEL_DIR / "yolo11n-seg.pt"

        try:
                download_file(yolo1_url, yolo1_dest)
        except Exception:
                print(f"Could not download {yolo1_url}. You can set YOLO11N_URL to a valid URL and re-run.")

        try:
                download_file(yolo2_url, yolo2_dest)
        except Exception:
                print(f"Could not download {yolo2_url}. You can set YOLO11N_SEG_URL to a valid URL and re-run.")

        print("Setup complete.")


if __name__ == "__main__":
        main()