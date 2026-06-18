"""
rnnoise — pure-ctypes loader for the RNNoise shared library (Xiph RNNoise).

Drop the CI-built binaries into python/lib/:
    python/lib/librnnoise-windows-x64.dll
    python/lib/librnnoise-linux-x86_64.so
    python/lib/librnnoise-linux-aarch64.so

Then:
    from rnnoise import Denoiser
    dn = Denoiser()                       # 48 kHz, 480-sample frames
    clean_i16, vad = dn.process(frame_i16)

`frame_i16` is a 10 ms mono int16 numpy array of exactly dn.frame samples
(480 @ 48 kHz). RNNoise works on float samples in int16 *range* (not normalised
to ±1), which this wrapper handles. Returns (denoised int16 frame, VAD prob 0..1).
"""

import ctypes
import os
import platform

import numpy as np

_LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")


def _lib_name() -> str:
    sysname = platform.system()
    machine = platform.machine().lower()
    if sysname == "Windows":
        return "librnnoise-windows-x64.dll"
    if sysname == "Linux":
        if machine in ("aarch64", "arm64"):
            return "librnnoise-linux-aarch64.so"
        return "librnnoise-linux-x86_64.so"
    raise RuntimeError(f"rnnoise: unsupported platform {sysname}/{machine}")


def _load():
    path = os.path.join(_LIB_DIR, _lib_name())
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"rnnoise binary not found: {path}\n"
            "Download it from the GitHub Actions 'build-rnnoise' run (Release tag "
            "'latest') and place it in python/lib/."
        )
    lib = ctypes.CDLL(path)
    # DenoiseState *rnnoise_create(RNNModel *model);  NULL = built-in model
    lib.rnnoise_create.argtypes = [ctypes.c_void_p]
    lib.rnnoise_create.restype = ctypes.c_void_p
    # float rnnoise_process_frame(DenoiseState*, float *out, const float *in);
    lib.rnnoise_process_frame.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.rnnoise_process_frame.restype = ctypes.c_float
    lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
    # int rnnoise_get_frame_size(void);  (newer API; fall back to 480 if absent)
    if hasattr(lib, "rnnoise_get_frame_size"):
        lib.rnnoise_get_frame_size.restype = ctypes.c_int
    return lib


class Denoiser:
    """RNNoise neural noise suppressor. Process 10 ms mono int16 frames @ 48 kHz."""

    SAMPLE_RATE = 48000

    def __init__(self):
        self._lib = _load()
        self._st = self._lib.rnnoise_create(None)
        if not self._st:
            raise RuntimeError("rnnoise: failed to create denoise state")
        if hasattr(self._lib, "rnnoise_get_frame_size"):
            self.frame = int(self._lib.rnnoise_get_frame_size())
        else:
            self.frame = 480

    def process(self, frame_i16: np.ndarray):
        """Denoise one frame. Returns (denoised int16 array, VAD probability 0..1)."""
        x = np.ascontiguousarray(frame_i16, dtype=np.int16)
        if len(x) != self.frame:
            raise ValueError(f"frame must be {self.frame} samples (10 ms @ 48 kHz)")
        fin = x.astype(np.float32)                  # int16 range, NOT normalised
        fout = np.empty(self.frame, dtype=np.float32)
        vad = self._lib.rnnoise_process_frame(
            self._st,
            fout.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            fin.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        out = np.clip(np.rint(fout), -32768, 32767).astype(np.int16)
        return out, float(vad)

    def close(self):
        if getattr(self, "_st", None):
            self._lib.rnnoise_destroy(self._st)
            self._st = None

    def __del__(self):
        self.close()
