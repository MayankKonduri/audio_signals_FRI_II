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
#   python main.py --engaged                      start already engaged
#
# While running, press Enter to toggle robot_speaking, standing in for the
# ROS subscriber that will set it once node.py exists. Audio is discarded
# while it is on, and for half a second after, so the robot does not hear
# its own voice as a person talking.
#
# engaged is set with --engaged at startup for now.

import argparse
import json
import threading
from collections import deque

from audio_context import state
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
    ap.add_argument("--engaged", action="store_true",
                    help="start engaged, so transcription runs from the off")
    args = ap.parse_args()

    problems = check()
    if problems:
        print("Fix config.py first:")
        for p in problems:
            print("  -", p)
        return 1

    state.set_engaged(args.engaged)

    show_proc = args.show in ("both", "processor")
    show_build = args.show in ("both", "builder")

    proc = Processor(CONFIG, verbose=show_proc)
    builder = Builder(CONFIG, proc.buffer, proc.utterance, proc.counters,
                      proc.stt, verbose=show_build)

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

    def keys():
        # Stands in for the ROS subscribers until node.py exists.
        while not stopping.is_set() and not finished.is_set():
            try:
                input()
            except (EOFError, OSError):
                return          # no terminal, so nothing to read

            state.set_robot_speaking(not state.robot_speaking())
            print(f"  >> robot_speaking = {state.robot_speaking()}")

    threading.Thread(target=keys, daemon=True).start()

    interval = CONFIG["window_sec"]
    print(f"capture running, building every {interval:.0f}s, showing "
          f"{args.show}.")
    print(f"engaged = {state.engaged()}.  Enter = toggle robot_speaking, "
          f"Ctrl+C to stop.\n")

    # The builder's own loop. finished.wait() doubles as the sleep, so this
    # wakes early if the audio source ends rather than sitting out the second.
    recent = deque(maxlen=5)
    try:
        while not finished.wait(interval):
            recent.append(builder.build())
    except KeyboardInterrupt:
        print("\nstopping")

    stopping.set()
    thread.join(timeout=2.0)       # let capture close the device
    if thread.is_alive():
        print("warning: capture thread did not stop cleanly")

    proc.stt.stop()
    print(f"\n{proc.blocks} blocks, {proc.n_frames} frames, "
          f"{builder.builds} builds, {proc.stt.done} transcripts")

    if recent:
        print(f"\nlast {len(recent)} audio objects:")
        for i, obj in enumerate(recent, 1):
            print(f"\n[{i}]")
            print(json.dumps(obj, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())