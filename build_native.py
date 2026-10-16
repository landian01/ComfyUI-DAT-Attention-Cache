# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 landian01
"""Build the optional Windows CUDA operator without Python headers or Ninja."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))  # Portable Python's ._pth may omit the script directory.
from native_backend import build_identity


def find_vcvars(explicit: Path | None) -> Path | None:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(explicit)
        return explicit
    if shutil.which("cl.exe"):
        return None
    vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    if vswhere.is_file():
        result = subprocess.run([str(vswhere), "-latest", "-products", "*", "-requires",
                                 "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                                 "-property", "installationPath"], check=True, capture_output=True, text=True)
        vcvars = Path(result.stdout.strip()) / "VC/Auxiliary/Build/vcvars64.bat"
        if vcvars.is_file():
            return vcvars
    raise RuntimeError("Install MSVC x64 C++ Build Tools, or pass --vcvars path/to/vcvars64.bat")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", help="Compile for this GPU's compute capability")
    parser.add_argument("--cuda-root", type=Path, help="CUDA toolkit directory (otherwise CUDA_PATH or nvcc on PATH)")
    parser.add_argument("--vcvars", type=Path, help="Optional MSVC vcvars64.bat location")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("The native build helper currently supports Windows x64 only; the PyTorch backend works elsewhere")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("A CUDA-enabled PyTorch environment and an NVIDIA GPU are required")
    cuda_root = args.cuda_root or (Path(os.environ["CUDA_PATH"]) if os.environ.get("CUDA_PATH") else None)
    nvcc = str(cuda_root / "bin/nvcc.exe") if cuda_root else shutil.which("nvcc.exe")
    if not nvcc or not Path(nvcc).is_file():
        parser.error("Cannot find nvcc; pass --cuda-root or install a matching CUDA toolkit")
    cuda_root = Path(nvcc).resolve().parents[1]
    version = subprocess.run([nvcc, "--version"], check=True, capture_output=True, text=True).stdout
    match = re.search(r"release (\d+\.\d+)", version)
    if match is None or match.group(1) != torch.version.cuda:
        parser.error(f"CUDA toolkit must match PyTorch CUDA {torch.version.cuda}; found {version.strip()}")
    capability = torch.cuda.get_device_capability(device)
    identity = build_identity(ROOT)
    key = hashlib.sha256(json.dumps({"identity": identity, "capability": capability,
                                     "nvcc": version}, sort_keys=True).encode()).hexdigest()[:16]
    output = ROOT / "_native"
    output.mkdir(exist_ok=True)
    library = output / f"dat_mask_{key}.dll"
    torch_root = Path(torch.__file__).resolve().parent
    args_nvcc = [nvcc, str(ROOT / "csrc/dat_mask_cuda.cu"), "-o", str(library),
                 "-shared", "-std=c++20", "-O3", "-lineinfo", f"-arch=sm_{capability[0]}{capability[1]}",
                 "--cudart", "shared", "-diag-suppress=1388,1394", "-Xcompiler", "/MD,/EHsc,/wd4819",
                 "-I" + str(torch_root / "include"), "-I" + str(torch_root / "include/torch/csrc/api/include"),
                 "-L" + str(torch_root / "lib"), "-lc10", "-lc10_cuda", "-ltorch_cpu", "-ltorch_cuda",
                 "-L" + str(cuda_root / "lib/x64"), "-lcudart"]
    vcvars = find_vcvars(args.vcvars)
    batch = output / "compile.cmd"
    command = "@echo off\r\nchcp 65001 > nul\r\n"
    if vcvars:
        command += 'call "' + str(vcvars) + '"\r\nif errorlevel 1 exit /b %errorlevel%\r\n'
    command += subprocess.list2cmdline(args_nvcc) + "\r\nexit /b %errorlevel%\r\n"
    batch.write_bytes(command.encode("utf-8"))
    print(f"Building for {torch.cuda.get_device_name(device)} (SM {capability[0]}{capability[1]})", flush=True)
    result = subprocess.run(["cmd.exe", "/d", "/c", str(batch)], cwd=output,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = result.stdout.decode("utf-8", errors="replace")
    (output / "build.log").write_text(log, encoding="utf-8")
    print("\n".join(log.splitlines()[-25:]), flush=True)
    result.check_returncode()
    torch.ops.load_library(str(library))
    first, second = torch.ops.dat_attention_cache_native.make_masks(
        16, 16, 8, 16, 4, 8, device, torch.float32)
    torch.cuda.synchronize(device)
    assert first.shape == second.shape == (2, 128, 128)
    record = {"identity": identity, "compute_capability": list(capability),
              "library": library.name, "nvcc": version.strip()}
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "manifest.json")
    print(f"Built and smoke-tested {library}. Restart ComfyUI to enable it.", flush=True)


if __name__ == "__main__":
    main()
