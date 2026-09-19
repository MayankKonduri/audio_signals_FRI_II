# Goal: run the whole audio node. Two loops on their own schedules:
#
#   processor.py   runs as fast as blocks arrive, about 8 times a second,
#                  measuring each one and filling the stores
#   audio_builder  wakes once a second, reads those stores, builds the JSON
#
# Both run until Ctrl+C.
#
#   python main.py                        both loops print
#   python main.py --show builder          only the once-a-second builds
#   python main.py --show processor        only the per-block lines
#   python main.py --show none             silent until the summary
#
#   python main.py --wav clips/capture_test.wav    stops when the file ends
#   python main.py --seconds 10                   for a quick check

import argparse
import threading

from audio_context.audio_builder import Builder
from audio_context.config import CONFIG, check
from audio_context.processor import Processor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=None,
                    help="stop after this long; runs until Ctrl+C otherwise")
    ap.add_argument("--wav", default=None, help="read a file instead of the mic")
    ap.add_argument("--show", default="both",
                    choices=["both", "processor", "builder", "none"],
                    help="which loop prints its output")
    args = ap.parse_args()

    problems = check()
    if problems:
        print("Fix config.py first:")
        for p in problems:
            print("  -", p)
        return 1

    show_proc = args.show in ("both", "processor")
    show_build = args.show in ("both", "builder")

    proc = Processor(CONFIG, verbose=show_proc)
    builder = Builder(CONFIG, proc.buffer, proc.utterance, proc.counters,
                      verbose=show_build)

    stopping = threading.Event()   # set by Ctrl+C, watched by both loops
    finished = threading.Event()   # set when the audio source runs out

    def capture():
        # Drain the stream, measuring each block into the stores. Checking
        # `stopping` each block lets this return normally on Ctrl+C, so the
        # `with AudioCapture(...)` inside closes the device properly. Killing
        # the thread instead would leave the microphone open.
        try:
            for _ in proc.split(wav=args.wav, seconds=args.seconds):
                if stopping.is_set():
                    return
        except Exception as e:
            print(f"\ncapture stopped: {e}")
        finally:
            finished.set()

    thread = threading.Thread(target=capture, daemon=True)
    thread.start()

    interval = CONFIG["window_sec"]
    print(f"capture running, building every {interval:.0f}s, showing "
          f"{args.show}. Ctrl+C to stop.\n")

    # The builder's own loop. finished.wait() doubles as the sleep, so this
    # wakes early if the audio source ends rather than sitting out the second.
    try:
        while not finished.wait(interval):
            builder.build()
    except KeyboardInterrupt:
        print("\nstopping")

    stopping.set()
    thread.join(timeout=2.0)       # let capture close the device
    if thread.is_alive():
        print("warning: capture thread did not stop cleanly")

    print(f"\n{proc.blocks} blocks, {proc.n_frames} frames, "
          f"{builder.builds} builds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())