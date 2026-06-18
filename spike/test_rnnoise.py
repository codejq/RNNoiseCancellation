r"""
test_rnnoise.py — test RNNoise neural noise suppression on this PC.

RNNoise runs at 48 kHz, 480-sample (10 ms) frames, so this test captures the mic
at 48 kHz natively (no resampling needed). Mirrors C:\qb-robot\aec\noise_test.py so
you can A/B RNNoise against the WebRTC suppressor.

Usage (from C:\qb-robot\RNNoise):
  py spike\test_rnnoise.py --list
      list audio devices

  py spike\test_rnnoise.py --record 8
      record 8 s @ 48 kHz, write raw48.wav + denoised48.wav, print RMS reduction
      and mean voice-activity probability. Stay SILENT to measure pure noise
      removal, or talk to judge how clean your voice sounds. Compare the WAVs.

  py spike\test_rnnoise.py --live
      realtime monitor: mic -> RNNoise -> output. USE HEADPHONES (speaker output
      feeds back into the mic). Ctrl-C to stop.

  Options:
    --device N       input (mic) device index   (default: STT mic config or 1)
    --out-device N   output device index for --live
"""

import argparse
import os
import sys
import wave

import numpy as np
import sounddevice as sd

# Load the ctypes Denoiser from ../python.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
try:
    from rnnoise import Denoiser
except Exception as e:  # pragma: no cover
    print(f"ERROR: could not import rnnoise: {e}")
    sys.exit(1)

SR = Denoiser.SAMPLE_RATE          # 48000
_STT_MIC_CFG = r"C:\qb-robot\config\stt_mic.txt"


def default_mic() -> int:
    try:
        with open(_STT_MIC_CFG) as f:
            return int(f.read().strip())
    except Exception:
        return 1


def list_devices():
    print(sd.query_devices())
    print(f"\nDefault input : {sd.default.device[0]}")
    print(f"Default output: {sd.default.device[1]}")
    print(f"STT mic config: {default_mic()}")


def rms(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    return float(np.sqrt(np.mean((x.astype(np.float64) / 32768.0) ** 2)))


def write_wav(path: str, samples: np.ndarray):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(samples.astype(np.int16).tobytes())


def record(seconds: float, device: int):
    dn = Denoiser()
    fr = dn.frame
    print(f"Recording {seconds:.0f}s from device {device} "
          f"({sd.query_devices(device)['name']}) @ {SR} Hz…")
    print("  (stay silent to measure noise removal, or talk to judge voice clarity)")

    raw = sd.rec(int(seconds * SR), samplerate=SR, channels=1,
                 dtype="int16", device=device)
    sd.wait()
    raw = raw.reshape(-1)

    clean = np.empty_like(raw)
    vads = []
    n = len(raw) // fr
    for i in range(n):
        a, b = i * fr, (i + 1) * fr
        out, vad = dn.process(raw[a:b])
        clean[a:b] = out
        vads.append(vad)
    tail = n * fr
    if tail < len(raw):
        clean[tail:] = raw[tail:]
    dn.close()

    here = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(here, "raw48.wav")
    clean_path = os.path.join(here, "denoised48.wav")
    write_wav(raw_path, raw)
    write_wav(clean_path, clean)

    r_raw, r_clean = rms(raw), rms(clean)
    red_db = 20 * np.log10(r_raw / r_clean) if r_clean > 0 and r_raw > 0 else 0.0
    print("\n── results ─────────────────────────────")
    print(f"  raw   RMS  : {r_raw:.5f}")
    print(f"  clean RMS  : {r_clean:.5f}")
    print(f"  reduction  : {red_db:+.1f} dB")
    print(f"  mean VAD   : {np.mean(vads):.2f}  (RNNoise's own voice-activity prob)")
    print(f"  raw      → {raw_path}")
    print(f"  denoised → {clean_path}")
    print("  Listen to both and compare. Speech should stay clean; noise should be")
    print("  far lower than the WebRTC result (aec\\denoised.wav).")


def live(device: int, out_device):
    dn = Denoiser()
    fr = dn.frame
    print(f"Live monitor: device {device} -> RNNoise -> "
          f"{'default' if out_device is None else out_device}")
    print("  ⚠  USE HEADPHONES — speaker output will echo back into the mic.")
    print("  Ctrl-C to stop.\n")

    def callback(indata, outdata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        mic = indata.reshape(-1).astype(np.int16)
        out, _ = dn.process(mic)
        outdata[:] = out.reshape(-1, 1)

    try:
        with sd.Stream(samplerate=SR, blocksize=fr, dtype="int16",
                       channels=1, device=(device, out_device),
                       callback=callback):
            while True:
                sd.sleep(200)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        dn.close()


def main():
    ap = argparse.ArgumentParser(description="RNNoise neural noise-suppression tester")
    ap.add_argument("--list", action="store_true", help="list audio devices")
    ap.add_argument("--record", nargs="?", const=8.0, type=float,
                    metavar="SECONDS", help="record + denoise (default 8s)")
    ap.add_argument("--live", action="store_true", help="realtime mic monitor")
    ap.add_argument("--device", type=int, default=None, help="input device index")
    ap.add_argument("--out-device", type=int, default=None, help="output device for --live")
    args = ap.parse_args()

    if args.list:
        list_devices()
        return

    dev = args.device if args.device is not None else default_mic()
    if args.live:
        live(dev, args.out_device)
    else:
        record(args.record if args.record is not None else 8.0, dev)


if __name__ == "__main__":
    main()
