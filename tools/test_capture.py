# Goal: check that capture.py delivers blocks steadily and the audio saves.
# Level analysis belongs in levels.py, not here.
#
#   python tools/test_capture.py                  live mic, 5 seconds
#   python tools/test_capture.py --seconds 20
#   python tools/test_capture.py --wav clips/hallway.wav

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=5.0)
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
        print("  - run tools/check_devices.py to see if the mic is found")
        print("  - another program may be holding the microphone")
        print("  - Windows: Settings > Privacy > Microphone, allow desktop apps")
        return 1

    if not args.wav:
        print(f"\nCapturing for {args.seconds:.0f} seconds...")

    stamps, chunks = [], []
    deadline = time.time() + args.seconds
    try:
        for block in cap.blocks():
            stamps.append(block.t)
            chunks.append(block.samples)
            print(f"\r  {len(chunks)} blocks", end="", flush=True)
            if time.time() >= deadline:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.stop()
    print()

    if not chunks:
        print("\nFAIL: no blocks arrived. The stream opened but the callback "
              "never fired.")
        return 1

    audio = np.concatenate(chunks)
    sizes = {len(c) for c in chunks}
    gaps = np.diff(stamps) * 1000
    expected = CONFIG["block_samples"] / CONFIG["sample_rate"] * 1000
    duration = len(audio) / CONFIG["sample_rate"]

    print("\n" + "=" * 46)
    print(f"  blocks          {len(chunks)}")
    print(f"  block size      {sizes.pop() if len(sizes) == 1 else sizes} samples")
    print(f"  audio           {duration:.2f} s")
    print(f"  block spacing   {gaps.mean():.1f} ms (expected {expected:.1f})")
    print(f"  worst spacing   {gaps.max():.1f} ms")
    print(f"  dropped blocks  {cap.dropped}")
    print("=" * 46)

    ok = True

    if len(sizes) > 1:
        print(f"WARN  Block size varied: {sorted(sizes)}. Can happen when the")
        print("      capture rate is not a whole multiple of 16000. Harmless.")

    if cap.dropped:
        print(f"FAIL  {cap.dropped} blocks dropped, so audio was lost.")
        ok = False

    if gaps.max() > expected * 2.5:
        print(f"WARN  A {gaps.max():.0f} ms gap between blocks. Occasional hiccups")
        print("      are normal; frequent ones mean the machine is struggling.")

    # Blocks should account for the elapsed time, or something stalled.
    covered = duration / (stamps[-1] - stamps[0] + expected / 1000)
    if covered < 0.95:
        print(f"WARN  Blocks cover only {covered:.0%} of the elapsed time.")

    OUT.parent.mkdir(exist_ok=True)
    sf.write(OUT, audio, CONFIG["sample_rate"])
    saved = sf.info(OUT)
    print(f"\n  saved           {OUT.name}")
    print(f"  reads back      {saved.duration:.2f} s, {saved.samplerate} Hz, "
          f"{saved.channels} ch")

    if abs(saved.duration - duration) > 0.05 or saved.samplerate != CONFIG["sample_rate"]:
        print("FAIL  The saved file does not match what was captured.")
        ok = False

    print("\nPASS" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())