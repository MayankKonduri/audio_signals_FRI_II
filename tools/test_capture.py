# Checks that capture.py works on this machine and microphone.
#
#   python tools/test_capture.py                  live mic, 10 seconds
#   python tools/test_capture.py --seconds 5
#   python tools/test_capture.py --wav clips/hallway.wav

# Runs the real capture code for ten seconds and checks it behaves: steady block timing, no dropped blocks, no silence or clipping, and saves the audio so you can listen back.

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_context.capture import AudioCapture
from audio_context.config import CONFIG, check

OUT = Path(__file__).resolve().parents[1] / "clips" / "capture_test.wav"


def meter(db, width=34, lo=-60.0, hi=0.0):
    filled = int(np.clip((db - lo) / (hi - lo), 0, 1) * width)
    return "#" * filled + "-" * (width - filled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--wav", default=None, help="read a file instead of the mic")
    args = ap.parse_args()

    problems = check()
    if problems:
        print("Fix config.py first:")
        for p in problems:
            print("  -", p)
        return 1

    cap = AudioCapture(CONFIG, wav=args.wav, realtime=True)
    try:
        cap.start()
    except Exception as e:
        print(f"\nCould not open audio: {e}\n")
        print("  - wrong input_device in config.py; run tools/check_devices.py")
        print("  - another program is holding the microphone")
        print("  - Windows: Settings > Privacy > Microphone, allow desktop apps")
        return 1

    if not args.wav:
        print(f"\nTalk into the microphone for {args.seconds:.0f} seconds.\n")

    stamps, levels, chunks = [], [], []
    deadline = time.time() + args.seconds
    try:
        for block in cap.blocks():
            db = 20 * np.log10(np.sqrt(np.mean(block.samples ** 2)) + 1e-12)
            stamps.append(block.t)
            levels.append(db)
            chunks.append(block.samples)
            print(f"\r  {db:6.1f} dBFS  [{meter(db)}]", end="", flush=True)
            if time.time() >= deadline:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.stop()
    print()

    if not chunks:
        print("\nFAIL: no audio blocks arrived. The stream opened but the callback "
              "never fired. Check the device and the OS microphone permission.")
        return 1

    audio = np.concatenate(chunks)
    sizes = {len(c) for c in chunks}
    gaps = np.diff(stamps) * 1000
    expected_gap = CONFIG["block_samples"] / CONFIG["sample_rate"] * 1000
    peak = float(np.max(np.abs(audio)))
    overall = 20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-12)

    print("\n" + "=" * 52)
    print(f"  blocks            {len(chunks)}")
    print(f"  block size        {sizes.pop() if len(sizes) == 1 else sizes} samples")
    print(f"  audio captured    {len(audio) / CONFIG['sample_rate']:.2f} s")
    print(f"  block spacing     {gaps.mean():.1f} ms (expected {expected_gap:.1f})")
    print(f"  worst spacing     {gaps.max():.1f} ms")
    print(f"  dropped blocks    {cap.dropped}")
    print(f"  level             {overall:.1f} dBFS, peak {peak:.3f}")
    print(f"  quietest / loudest{min(levels):7.1f} / {max(levels):.1f} dBFS")
    print("=" * 52)

    ok = True

    if peak < 1e-4:
        print("FAIL  Silence. The device opened but captured nothing.")
        ok = False
    elif peak > 0.99:
        print("WARN  Clipping. Turn the input gain down.")
    elif max(levels) - min(levels) < 6 and not args.wav:
        print("WARN  The level barely moved. Either nobody spoke, or the mic")
        print("      applies automatic gain. noise_level depends on that range.")

    if cap.dropped:
        print(f"WARN  {cap.dropped} blocks dropped. Something downstream is too slow.")

    if gaps.max() > expected_gap * 2.5:
        print(f"WARN  A {gaps.max():.0f} ms gap between blocks. Occasional hiccups")
        print("      are normal; frequent ones mean the machine is struggling.")

    OUT.parent.mkdir(exist_ok=True)
    sf.write(OUT, audio, CONFIG["sample_rate"])
    print(f"\nSaved to {OUT}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())