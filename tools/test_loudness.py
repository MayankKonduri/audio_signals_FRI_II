# Goal: check loudness.py on real audio, and see how the frame levels spread.
#
#   python tools/test_loudness.py                 live mic, 10 seconds
#   python tools/test_loudness.py --seconds 30
#   python tools/test_loudness.py --wav clips/hallway.wav
#
# Talk, then go quiet, then talk again. The quiet stretches are what matter:
# they are what the noise floor will be built from.

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_context.capture import AudioCapture
from audio_context.config import CONFIG, check
from audio_context.loudness import SILENCE_DB, is_real, rms_db, to_frames


def bar(db, width=28, lo=-70.0, hi=-10.0):
    filled = int(np.clip((db - lo) / (hi - lo), 0, 1) * width)
    return "#" * filled + "-" * (width - filled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--wav", default=None)
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
        print(f"\nCould not open audio: {e}")
        print("Run tools/check_devices.py to see if the mic is found.")
        return 1

    if not args.wav:
        print(f"\n{args.seconds:.0f} seconds. Talk, go quiet, then talk again.\n")

    fs = CONFIG["frame_samples"]
    per_window = CONFIG["window_sec"]
    all_db = []
    window, window_start = [], None
    deadline = time.time() + args.seconds

    try:
        for block in cap.blocks():
            db = rms_db(to_frames(block.samples, fs))
            all_db.append(db)
            window.append(db)
            if window_start is None:
                window_start = block.t

            # Report once per config window, the same cadence the node will use.
            if block.t - window_start >= per_window:
                w = np.concatenate(window)
                real = w[is_real(w)]
                if len(real):
                    print(f"  {len(all_db) * len(db):5d} frames   "
                          f"quietest {real.min():6.1f}   median {np.median(real):6.1f}   "
                          f"loudest {real.max():6.1f}   [{bar(np.median(real))}]")
                else:
                    print(f"  {len(all_db) * len(db):5d} frames   all silent")
                window, window_start = [], block.t

            if time.time() >= deadline:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.stop()

    if not all_db:
        print("\nFAIL: no audio arrived.")
        return 1

    db = np.concatenate(all_db)
    real = db[is_real(db)]
    silent = len(db) - len(real)

    print("\n" + "=" * 54)
    print(f"  frames                {len(db)}")
    print(f"  digital silence       {silent}  ({silent / len(db):.1%})")
    if len(real) == 0:
        print("=" * 54)
        print("FAIL  Every frame was digital silence.")
        return 1
    for label, value in [("quietest", real.min()), ("5th percentile", np.percentile(real, 5)),
                         ("median", np.median(real)), ("95th percentile", np.percentile(real, 95)),
                         ("loudest", real.max())]:
        print(f"  {label:<21} {value:7.1f} dBFS")
    print(f"  range                 {real.max() - real.min():7.1f} dB")
    print("=" * 54)

    ok = True

    # The 5th percentile stands in for the quiet stretches, which is roughly
    # what the noise floor will settle on.
    floor_guess = np.percentile(real, 5)
    print(f"\nA noise floor here would land near {floor_guess:.0f} dBFS.")
    print(f"Current cutoffs: quiet below {CONFIG['quiet_max_db']:.0f}, "
          f"loud above {CONFIG['moderate_max_db']:.0f}.")

    if silent / len(db) > 0.02:
        print(f"\nWARN  {silent / len(db):.1%} of frames were digital silence, which is")
        print("      more than stream startup explains. The microphone may mute")
        print("      itself when the room is quiet, which would make quiet rooms")
        print("      impossible to tell apart. Check for a noise gate setting.")
        ok = False
    elif silent:
        print(f"\n{silent} silent frame(s), rare enough to be stream edges. Excluded "
              "from the median, so harmless.")

    if real.max() - real.min() < 15 and not args.wav:
        print("\nWARN  Only a small spread between quiet and loud. Either nobody")
        print("      spoke, or the mic is applying automatic gain.")
        ok = False

    print("\nPASS" if ok else "\nNeeds a look")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())