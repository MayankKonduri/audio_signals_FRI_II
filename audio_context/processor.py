# Goal: own the one audio stream and feed every module from it.
#
# The microphone can only be held by one thing at a time, so this is the only
# place AudioCapture is opened. Each block goes to loudness.py and vad.py,
# which frame it identically, so their results line up index by index:
#
#   db     [-65.2, -64.8, -41.3, -39.7]
#   voice  [False, False,  True,  True]
#
# Frame 2 was -41.3 dBFS and a voice, so that is a person at -41.3. Neither
# array says that on its own.
#
# run() yields one Result per block, carrying the raw arrays. split()
# yields the same audio annotated per 32 ms frame and divided into the two
# piles the next modules read:
#
#   for background, speech in Processor(CONFIG).split(seconds=5):
#       print(len(background), len(speech))
#
# Pass verbose=True to print a line per block, which is what main.py does
# and what running this module directly does:
#   python -m audio_context.processor
#
# More modules join _analyse as they are written: background.py and speech.py
# next, then the utterance tracker and the counters, then the JSON.

import time
from collections import namedtuple

import numpy as np

from audio_context import frames as fr
from audio_context import loudness, vad
from audio_context.capture import AudioCapture
from audio_context.frames import Buffer

# One per block. db and voice are the same length, one entry per 32 ms frame.
Result = namedtuple("Result", "t db voice")


class Processor:

    def __init__(self, cfg, verbose=False):
        self.cfg = cfg
        self.verbose = verbose     # print a line per block
        self.frame_samples = cfg["frame_samples"]
        self.vad_threshold = cfg["vad_threshold"]
        self.frame_sec = cfg["frame_samples"] / cfg["sample_rate"]
        self.blocks = 0
        self.n_frames = 0

        # The recent past of both piles, one SQLite store each, oldest
        # frames dropped once full.
        self.buffer = Buffer(cfg["buffer_dir"], cfg["buffer_seconds"],
                             self.frame_sec)

    def _report(self, background, speech):
        # One line per block: the two piles, then the state of the stores.
        stamp = min(f.t for f in background + speech)
        bg = f"{np.median([f.db for f in background]):7.1f}" if background else "      -"
        sp = f"{np.median([f.db for f in speech]):7.1f}" if speech else "      -"
        n_bg, n_sp = self.buffer.counts()
        print(f"  {stamp:.3f}  |  background {len(background)} @{bg} dBFS  |  "
              f"speech {len(speech)} @{sp} dBFS  |  bg {n_bg:3d} / voice {n_sp:3d} "
              f"/ oldest {self.buffer.span():4.1f}s")

    def _analyse(self, block):
        # Everything that happens to one block. Both modules receive the same
        # block object, which is what keeps the two arrays aligned.
        db = loudness.from_block(block, self.frame_samples)
        voice = vad.from_block(block, self.frame_samples, self.vad_threshold)

        self.blocks += 1
        self.n_frames += len(db)
        return Result(block.t, db, voice)

    def run(self, wav=None, seconds=None):
        # Open the stream and yield one Result per block. Stops at the end of
        # a wav file, after `seconds`, or when interrupted.
        vad.warm_up()   # load the detector before the first block arrives
        deadline = None if seconds is None else time.time() + seconds

        with AudioCapture(self.cfg, wav=wav) as cap:
            for block in cap.blocks():
                result = self._analyse(block)
                if len(result.db) == 0:
                    continue
                yield result
                if deadline is not None and time.time() >= deadline:
                    return

    def frames(self, wav=None, seconds=None):
        # The same stream, one annotated Frame at a time rather than a block
        # of arrays. Each Frame is a timestamp, a level and a verdict.
        for result in self.run(wav=wav, seconds=seconds):
            yield from fr.annotate(result, self.frame_sec)

    def split(self, wav=None, seconds=None):
        # Per block, the frames divided in two: those with nobody talking,
        # and those with a voice. background.py reads the first pile,
        # speech.py the second.
        for result in self.run(wav=wav, seconds=seconds):
            background, speech = fr.from_result(result, self.frame_sec)
            self.buffer.add(background, speech)
            if self.verbose:
                self._report(background, speech)
            yield background, speech


if __name__ == "__main__":
    import argparse

    from audio_context.config import CONFIG

    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--wav", default=None)
    args = ap.parse_args()

    proc = Processor(CONFIG, verbose=True)
    n_bg = n_sp = 0
    for background, speech in proc.split(wav=args.wav, seconds=args.seconds):
        n_bg += len(background)
        n_sp += len(speech)
    print(f"\n{proc.blocks} blocks, {proc.n_frames} frames: "
          f"{n_bg} background, {n_sp} speech")