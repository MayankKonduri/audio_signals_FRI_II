# Goal: check loudness.py on real audio. Reads from capture.py and hands
# each block to loudness.py, the way processor.py will.
#
#   python tools/test_loudness.py                 live mic, 5 seconds
#   python tools/test_loudness.py --seconds 20
#   python tools/test_loudness.py --wav clips/hallway.wav

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_context import loudness
from audio_context.capture import AudioCapture
from audio_context.config import CONFIG, check


def bar(db, width=26, lo=-80.0, hi=-10.0):
    filled = int(np.clip((db - lo) / (hi - lo), 0, 1) * width)
    return "#" * filled + "-" * (width - filled)


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

    cap = AudioCapture(CONFIG, wav=args.wav)
    try:
        cap.start()
    except Exception as e:
        print(f"\nFAIL  could not open audio: {e}")
        return 1

    if not args.wav:
        print(f"\nTalk for {args.seconds:.0f} seconds, then go quiet.\n")

    stamps, levels = [], []
    n_frames = 0
    deadline = time.time() + args.seconds
    try:
        for block in cap.blocks():
            db = loudness.from_block(block)
            if len(db) == 0:
                continue
            stamps.append(block.t)
            levels.append(db)
            n_frames += len(db)
            real = db[loudness.is_real(db)]
            shown = float(np.median(real)) if len(real) else -120.0
            print(f"\r  {n_frames:5d} frames   {shown:6.1f} dBFS  [{bar(shown)}]",
                  end="", flush=True)
            if time.time() >= deadline:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.stop()
    print()

    if not levels:
        print("\nFAIL  no blocks arrived")
        return 1

    db = np.concatenate(levels)
    sizes = {len(d) for d in levels}
    gaps = np.diff(stamps) * 1000
    expected = CONFIG["block_samples"] / CONFIG["sample_rate"] * 1000
    real = db[loudness.is_real(db)]
    silent = len(db) - len(real)

    print("\n" + "=" * 52)
    print(f"  blocks          {len(levels)}")
    print(f"  frames          {n_frames}   "
          f"({sizes.pop() if len(sizes) == 1 else sizes} per block)")
    print(f"  block spacing   {gaps.mean():.1f} ms (expected {expected:.1f})")
    print(f"  worst spacing   {gaps.max():.1f} ms")
    print(f"  dropped blocks  {cap.dropped}")
    print(f"  digital silence {silent}  ({silent / len(db):.1%})")
    if len(real):
        print(f"  quietest frame  {real.min():7.1f} dBFS")
        print(f"  median frame    {np.median(real):7.1f} dBFS")
        print(f"  loudest frame   {real.max():7.1f} dBFS")
        print(f"  range           {real.max() - real.min():7.1f} dB")
    print("=" * 52)

    if len(real) == 0:
        print("\nFAIL  every frame was digital silence")
        return 1

    # Nothing real sits above 0 dBFS. If one does, the conversion is broken.
    if real.max() > 0.0:
        print(f"\nFAIL  a frame read {real.max():.1f} dBFS; above 0 is impossible")
        return 1

    if cap.dropped:
        print(f"\nFAIL  {cap.dropped} blocks dropped, so audio was lost")
        return 1

    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())