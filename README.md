# rnnoise-maya — neural noise suppression for MAYA

[RNNoise](https://github.com/xiph/rnnoise) (Xiph) is a small recurrent-neural-net
noise suppressor. Unlike WebRTC's gentle noise suppression (which tops out ~70–80 %
and only on *steady* noise), RNNoise removes far more — including non-stationary
noise (keyboard, clicks, traffic) — while keeping speech natural. Single mic stream,
runs realtime on CPU, no GPU.

This repo builds RNNoise as a **ctypes-loadable shared library** for every platform
MAYA runs on, exactly like `webrtc-aec`:

| platform        | artifact                          | where it runs        |
|-----------------|-----------------------------------|----------------------|
| Windows x64     | `librnnoise-windows-x64.dll`      | the dev PC           |
| Linux x86_64    | `librnnoise-linux-x86_64.so`      | x86 Linux / WSL      |
| Linux aarch64   | `librnnoise-linux-aarch64.so`     | the Pi 5 / Jetson    |

## How it's built

GitHub Actions (`.github/workflows/build.yml`) builds RNNoise with **its own
autotools build** on each platform (MSYS2/MinGW on Windows, gcc on Linux). That build
already downloads the model, generates `config.h`, and exports the public C ABI:

```c
DenoiseState *rnnoise_create(RNNModel *model);   // NULL = built-in model
int   rnnoise_get_frame_size(void);              // 480 samples (10 ms @ 48 kHz)
float rnnoise_process_frame(DenoiseState*, float *out, const float *in);  // returns VAD prob
void  rnnoise_destroy(DenoiseState*);
```

So there is **no C wrapper** — Python loads the library and calls those directly.

## Using it

1. Push this repo to GitHub. The Actions run builds all three binaries and publishes
   them to a Release tagged `latest`.
2. Download the three files into `python/lib/`.
3. Test on the dev PC:

```powershell
cd C:\qb-robot\RNNoise
py spike\test_rnnoise.py --list                 # audio devices
py spike\test_rnnoise.py --record 8             # raw48.wav + denoised48.wav + dB + mean VAD
py spike\test_rnnoise.py --live                 # realtime monitor (HEADPHONES ONLY)
```

`raw48.wav` vs `denoised48.wav` is the A/B test. Compare against the WebRTC result
from `C:\qb-robot\aec\noise_test.py` to hear the difference.

## Notes

- **Sample rate is 48 kHz, frame is 480 samples.** The test captures at 48 kHz so no
  resampling is needed. To wire RNNoise into the 16 kHz STT later, resample
  16 k→48 k→denoise→48 k→16 k (or run capture at 48 k and downsample after).
- `rnnoise_process_frame` also returns a **voice-activity probability** (0..1) — handy
  as a built-in VAD for the STT gate.
- `--live` will howl on speakers (no echo cancellation here — that's a separate
  problem). Use headphones, or just use `--record`.
