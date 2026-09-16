# Finds your microphone and prints the values to put in config.py.
# Lists input devices and prints the config values for the one you pick, so you know the mic works before writing code against it.

import sys
import numpy as np

try:
    import sounddevice as sd
except Exception as e:
    sys.exit(f"sounddevice is not working: {e}\n"
             "On Linux you may need: sudo apt install libportaudio2")

TARGET_RATE = 16000
TEST_RATES = [16000, 32000, 44100, 48000]
RECORD_SECONDS = 3


def list_inputs():
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    inputs = [(i, d) for i, d in enumerate(devices) if d["max_input_channels"] > 0]

    if not inputs:
        sys.exit("No input devices found. Is the microphone plugged in?")

    try:
        default_in = sd.default.device[0]
    except Exception:
        default_in = None

    print(f"{'idx':>4}  {'ch':>3}  {'rate':>9}  {'backend':<12} name")
    print("-" * 74)
    for i, d in inputs:
        api = hostapis[d["hostapi"]]["name"].replace("Windows ", "")
        mark = "  <- system default" if i == default_in else ""
        print(f"{i:>4}  {d['max_input_channels']:>3}  "
              f"{int(d['default_samplerate']):>7} Hz  {api:<12} {d['name']}{mark}")

    if any(hostapis[d["hostapi"]]["name"].startswith("Windows WASAPI") for _, d in inputs):
        print("\nOne mic appears under several backends. Prefer WASAPI:")
        print("MME caps at 2 channels and shortens names.")
    return inputs, default_in


def supported_rates(index, channels):
    ok = []
    for rate in TEST_RATES:
        try:
            sd.check_input_settings(device=index, channels=channels,
                                    samplerate=rate, dtype="float32")
            ok.append(rate)
        except Exception:
            pass
    return ok


def record(index, channels, rate):
    print(f"\nRecording {RECORD_SECONDS}s at {rate} Hz, {channels} channel(s). "
          "Say something.")
    audio = sd.rec(int(RECORD_SECONDS * rate), samplerate=rate,
                   channels=channels, dtype="float32", device=index)
    sd.wait()
    return audio


def describe(audio):
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    print(f"\n  peak {peak:.3f}   level {20 * np.log10(rms + 1e-12):.1f} dBFS")

    if peak < 1e-4:
        print("  SILENT. Usually the wrong device index, a muted input, "
              "or missing OS permission.")
    elif peak > 0.99:
        print("  CLIPPING. Turn the input gain down.")
    else:
        print("  Looks healthy.")

    if audio.shape[1] > 1:
        print("\n  per channel:")
        for c in range(audio.shape[1]):
            c_rms = float(np.sqrt(np.mean(audio[:, c] ** 2)))
            print(f"    channel {c}: {20 * np.log10(c_rms + 1e-12):6.1f} dBFS")
        print("  Channels at the same level means the mic is mono, "
              "duplicated. Pick one with use_channel.")


def main():
    print("INPUT DEVICES\n")
    _, default_in = list_inputs()

    raw = input(f"\nWhich device index? [{default_in}] ").strip()
    index = int(raw) if raw else default_in
    info = sd.query_devices(index)
    channels = int(info["max_input_channels"])

    print(f"\nTesting rates on '{info['name']}'...")
    rates = supported_rates(index, channels)
    if not rates:
        sys.exit("This device would not open at any rate tested. Try another "
                 "index, or close whatever else is using the microphone.")
    print(f"  opens at: {', '.join(str(r) for r in rates)} Hz")

    # Prefer a clean multiple of the target, so resampling stays exact.
    if TARGET_RATE in rates:
        capture_rate = TARGET_RATE
    else:
        clean = [r for r in rates if r % TARGET_RATE == 0]
        capture_rate = min(clean) if clean else min(rates)

    describe(record(index, channels, capture_rate))

    api = sd.query_hostapis()[info["hostapi"]]["name"]
    print("\n" + "=" * 64)
    print("PASTE INTO audio_context/config.py\n")
    print(f'    "input_device":        {info["name"].split("(")[-1].rstrip(")")!r},')
    print(f'    "input_backend":       {api.replace("Windows ", "")!r},')
    print(f'    "input_channels":      {channels},')
    if capture_rate == TARGET_RATE:
        print(f'    "capture_sample_rate": None,')
    else:
        print(f'    "capture_sample_rate": {capture_rate},')
    print("=" * 64)


if __name__ == "__main__":
    main()