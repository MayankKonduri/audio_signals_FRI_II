# Goal: check vad.py on real audio. Reads from capture.py and hands each
# block to vad.py, the way processor.py will.
#
#   python tools/test_vad.py                            live mic, 5 seconds
#   python tools/test_vad.py --seconds 20
#   python tools/test_vad.py --wav clips/x.wav
#   python tools/test_vad.py --wav clips/x.wav --play    hear it as you watch

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_context import vad
from audio_context.capture import AudioCapture
from audio_context.config import CONFIG, check


def strip_of(voice):
    return "".join("#" if v else "." for v in voice)


def playback(path):
    # Work out every verdict up front, then play the clip and print them in
    # time with the audio, so you can check by ear what it decided.
    import sounddevice as sd
    import soundfile as sf

    fs = CONFIG["frame_samples"]
    frame_sec = fs / CONFIG["sample_rate"]

    audio, sr = sf.read(path, dtype="float32", always_2d=True)

    print("loading the voice detector...")
    vad.warm_up()

    # Same path as live: capture.py decodes the file, vad.py judges it.
    verdicts = []
    with AudioCapture(CONFIG, wav=path, realtime=False) as cap:
        for block in cap.blocks():
            verdicts.append(vad.from_block(block))
    if not verdicts:
        print("Clip is too short.")
        return 1
    voice = np.concatenate(verdicts)

    print(f"\n{path}   {len(voice) * frame_sec:.1f}s, {len(voice)} frames")
    print("# = voice, . = background\n")

    sd.play(audio, sr)
    start = time.perf_counter()
    for i in range(len(voice)):
        wait = start + i * frame_sec - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        if i % 4 == 0 or i == len(voice) - 1:
            mark = "VOICE" if voice[i] else "     "
            print(f"\r  {i * frame_sec:5.1f}s  "
                  f"{strip_of(voice[max(0, i - 45):i + 1]):<46}  {mark}",
                  end="", flush=True)
    sd.wait()
    print()

    print("\n" + "=" * 52)
    print(f"  frames          {len(voice)}")
    print(f"  judged voice    {voice.sum()}  ({voice.mean():.0%})")
    print(f"  background      {(~voice).sum()}")
    print("=" * 52)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--wav", default=None, help="read a file instead of the mic")
    ap.add_argument("--play", action="store_true",
                    help="play the wav and show the verdict in time with it")
    args = ap.parse_args()

    problems = check()
    if problems:
        print("Fix config.py first:")
        for p in problems:
            print("  -", p)
        return 1

    if args.play:
        if not args.wav:
            print("--play needs --wav. Record one with tools/test_capture.py")
            return 1
        return playback(args.wav)

    print("loading the voice detector...")
    vad.warm_up()

    cap = AudioCapture(CONFIG, wav=args.wav)
    try:
        cap.start()
    except Exception as e:
        print(f"\nFAIL  could not open audio: {e}")
        return 1

    if not args.wav:
        print(f"\nTalk for {args.seconds:.0f} seconds, then tap the desk.\n")

    verdicts = []
    n_frames = 0
    deadline = time.time() + args.seconds
    try:
        for block in cap.blocks():
            voice = vad.from_block(block)
            if len(voice) == 0:
                continue
            verdicts.append(voice)
            n_frames += len(voice)
            mark = "VOICE" if voice.sum() > len(voice) / 2 else "     "
            print(f"\r  {n_frames:5d} frames   {strip_of(voice)}   "
                  f"{voice.sum()}/{len(voice)} voice  {mark}", end="", flush=True)
            if time.time() >= deadline:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.stop()
    print()

    if not verdicts:
        print("\nFAIL  no blocks arrived")
        return 1

    voice = np.concatenate(verdicts)
    sizes = {len(v) for v in verdicts}

    print("\n" + "=" * 52)
    print(f"  blocks          {len(verdicts)}")
    print(f"  frames          {len(voice)}   "
          f"({sizes.pop() if len(sizes) == 1 else sizes} per block)")
    print(f"  judged voice    {voice.sum()}  ({voice.mean():.0%})")
    print(f"  background      {(~voice).sum()}")
    print(f"  dropped blocks  {cap.dropped}")
    print("=" * 52)

    if voice.dtype != bool:
        print(f"\nFAIL  expected True/False per frame, got {voice.dtype}")
        return 1

    if cap.dropped:
        print(f"\nFAIL  {cap.dropped} blocks dropped, so audio was lost")
        return 1

    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())