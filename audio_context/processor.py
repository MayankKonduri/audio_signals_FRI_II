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
from audio_context import state
from audio_context.counters import Counters
from audio_context.frames import Buffer
from audio_context.ring import RingBuffer
from audio_context.stt import Transcriber
from audio_context.utterance import Utterance

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

        # Every frame goes through the tracker in order, so it can smooth the
        # verdicts into utterances. The builder reads speech_now from it.
        self.utterance = Utterance(cfg)
        self.spans = []            # utterances that have finished

        # Rolling counts over the last few seconds, fed the tracker's
        # smoothed verdict rather than the raw per-frame one.
        self.counters = Counters(cfg)

        # Raw audio, always kept, so a finished utterance can be pulled back
        # out. Filling even when not engaged is the point: the words spoken
        # just before the decision to engage are still here.
        self.ring = RingBuffer(cfg)
        self.stt = Transcriber(cfg, verbose=verbose)
        self.ignored = 0           # blocks dropped while the robot spoke

    def _report(self, background, speech):
        # One line per block: the two piles, then the state of the stores.
        if not background and not speech:
            return              # suppressed block, nothing was measured
        stamp = min(f.t for f in background + speech)
        bg = f"{np.median([f.db for f in background]):7.1f}" if background else "      -"
        sp = f"{np.median([f.db for f in speech]):7.1f}" if speech else "      -"
        n_bg, n_sp = self.buffer.counts()
        print(f"  {stamp:.3f}  |  background {len(background)} @{bg} dBFS  |  "
              f"speech {len(speech)} @{sp} dBFS  |  bg {n_bg:3d} / voice {n_sp:3d} "
              f"/ oldest {self.buffer.span():4.1f}s")

    def _finish(self, span):
        # An utterance just ended. Transcribe it, but only when engaged:
        # this is the most expensive thing in the pipeline, and transcribing
        # bystanders who never agreed to anything is worth avoiding.
        self.spans.append(span)
        if not state.engaged():
            return
        samples = self.ring.slice(span.start, span.end)
        if samples is not None:
            self.stt.request(samples, when=span.end)

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

        self.stt.warm_up()
        self.stt.start()

        with AudioCapture(self.cfg, wav=wav) as cap:
            for block in cap.blocks():
                # The robot hears itself. While its own voice is playing, and
                # briefly after while the room still rings with it, none of
                # this audio describes the room or a person.
                if state.ignore_audio(block.t):
                    # Still yield, so a caller watching for a stop signal
                    # sees it. Both piles empty says nothing was measured.
                    self.ignored += 1
                    yield Result(block.t, np.empty(0), np.empty(0, dtype=bool))
                    continue

                self.ring.add(block)
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

            # In time order, which matters: the tracker is a state machine.
            for f in sorted(background + speech, key=lambda f: f.t):
                span = self.utterance.push(f)
                if span is not None:
                    self._finish(span)
                self.counters.push(f, self.utterance.speech_now)
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