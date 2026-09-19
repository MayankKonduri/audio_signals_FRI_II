# Goal: turn the flickering per-frame voice verdicts into utterances, and
# answer whether someone is talking right now. Produces speech_now.
#
#   utt = Utterance(CONFIG)
#   for f in frames:
#       finished = utt.push(f)      # a Span when one just ended, else None
#   utt.speech_now                  # True while one is open
#
# The raw verdicts cannot be used directly. A single sentence flips between
# voice and silence a dozen times, because the gaps between words and the
# silent parts of consonants all read as non-voice. Published raw,
# speech_now would flicker several times a second and seconds_since_speech
# would reset constantly.
#
# So an utterance opens only after sustained voice and closes only after
# sustained silence. Opening is harder than continuing, which stops a cough
# from opening one while keeping the gaps inside a sentence from ending one.
#
# Opening counts voice frames within a sliding window rather than demanding
# an unbroken run. At low SNR the detector flickers, so a run of eight in a
# row almost never happens and distant speech would never register at all.

from collections import deque, namedtuple

# A finished utterance. counters.py, stt.py and tone.py all work on these.
Span = namedtuple("Span", "start end")


class Utterance:

    def __init__(self, cfg):
        self.frame_sec = cfg["frame_samples"] / cfg["sample_rate"]
        self.open_frames = round(cfg["utterance_open_sec"] / self.frame_sec)
        self.open_needed = max(1, round(self.open_frames * cfg["utterance_open_ratio"]))
        self.close_frames = round(cfg["utterance_close_sec"] / self.frame_sec)
        self.max_frames = round(cfg["utterance_max_sec"] / self.frame_sec)

        self.speech_now = False    # is an utterance open right now?
        self._window = deque(maxlen=self.open_frames)   # recent verdicts
        self._run = 0              # silent frames in a row, for closing
        self._start = None         # timestamp the current utterance began
        self._held = 0             # frames since it opened, for the cap

    def push(self, frame):
        # One frame in. Returns a Span when an utterance just ended, else None.
        if not self.speech_now:
            # Waiting to open: enough voice frames within the window, not
            # necessarily consecutive.
            self._window.append(frame)
            voiced = sum(1 for f in self._window if f.voice)
            if voiced >= self.open_needed:
                # Backdate the start to the first voice frame in the window.
                # By the time the count is reached the person has been
                # talking for a moment already, so marking the start here
                # would clip the opening of every utterance.
                first = next(f for f in self._window if f.voice)
                self._start = first.t
                self.speech_now = True
                self._run = 0
                self._held = len(self._window)
                self._window.clear()
            return None

        # Open: count silence frames, reset on any voice.
        self._held += 1
        self._run = 0 if frame.voice else self._run + 1

        if self._run >= self.close_frames:
            # Backdate the end too: the silence began close_frames ago, so
            # the utterance really finished then rather than now.
            return self._close(frame.t - (self.close_frames - 1) * self.frame_sec)

        if self._held >= self.max_frames:
            # A cap, so someone talking without pause cannot run past the end
            # of the audio buffer and leave nothing to transcribe.
            return self._close(frame.t)
        return None

    def _close(self, end):
        span = Span(self._start, end)
        self.speech_now = False
        self._start = None
        self._run = 0
        self._held = 0
        self._window.clear()
        return span