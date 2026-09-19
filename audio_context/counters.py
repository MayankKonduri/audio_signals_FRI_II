# Goal: answer the two questions about time rather than the current moment.
# Produces speech_ratio_10s and seconds_since_speech.
#
#   c = Counters(CONFIG)
#   c.push(frame, speech_now)       # every frame, in order
#   c.ratio(now)                    # 0.62
#   c.since_speech(now)             # 3.4
#
# Both read the utterance tracker's smoothed verdict rather than the raw
# per-frame one, so a gap between two words does not count as speech having
# stopped.

from collections import deque


class Counters:

    def __init__(self, cfg):
        self.window = cfg["ratio_window_sec"]
        self._recent = deque()     # (timestamp, was speech) over the window
        self._last_speech = None   # when speech was last heard

    def push(self, frame, speech_now):
        # One frame in, with the tracker's verdict for it.
        self._recent.append((frame.t, speech_now))
        if speech_now:
            self._last_speech = frame.t

        # Drop anything past the window. Pruned by time rather than count so
        # the span is right even if frames arrive unevenly.
        cutoff = frame.t - self.window
        while self._recent and self._recent[0][0] < cutoff:
            self._recent.popleft()

    def ratio(self):
        # Share of the window that had someone talking.
        #
        # High means a conversation is already under way, which is the case
        # where the robot should stay out of it. A single remark barely
        # moves this, which is the distinction that matters.
        if not self._recent:
            return None
        return round(sum(1 for _, s in self._recent if s) / len(self._recent), 2)

    def since_speech(self, now=None):
        # Seconds since anyone last spoke, so the robot can pick a pause
        # rather than cut in. 0.0 while someone is still talking.
        #
        # None when nobody has spoken at all yet: that is different from
        # having just stopped, and the LLM should be able to tell them apart.
        if self._last_speech is None:
            return None
        if now is None:
            now = self._recent[-1][0] if self._recent else self._last_speech
        return round(max(0.0, now - self._last_speech), 1)