# Goal: build the audio JSON once per second, by reading the stores that
# processor.py fills and asking each module for its numbers.
#
# Runs on its own schedule. processor.py loops about 8 times a second as
# blocks arrive; this runs once a second, which is the rate the JSON is
# published at. Keeping them apart means a slow build never stalls capture.
#
# robot_speaking and engaged are not computed here. They arrive from other
# nodes via state.py and are echoed back, so the LLM can tell why transcript
# and tone are empty: nobody spoke, or this node was not listening yet.
#
# Fields still to come: tone.

import time

from audio_context import state
from audio_context.background import Background
from audio_context.speech import Speech


class Builder:

    def __init__(self, cfg, buffer, utterance, counters, stt=None,
                 verbose=False):
        self.cfg = cfg
        self.buffer = buffer       # the stores processor.py is filling
        self.utterance = utterance # the tracker processor.py is feeding
        self.counters = counters   # the rolling counts it also feeds
        self.stt = stt             # the transcriber, when there is one
        self.verbose = verbose
        self.background = Background(cfg)
        self.speech = Speech(cfg)
        self.builds = 0

    def build(self):
        # One audio object. Called once per second by main.py.
        self.builds += 1

        # While the robot is talking, its own voice is all the microphone
        # can hear, so nothing is being measured. Report null rather than
        # the last numbers taken before it started: frozen values look like
        # live ones, and the LLM would read a stale pause as a fresh one.
        if state.ignore_audio():
            return self._suppressed()

        floor, level = self.background.read(self.buffer.background_levels())
        snr = self.speech.snr(self.buffer.speech_levels(), floor)

        audio = {
            # When the sound happened, not when this ran. If transcription
            # later takes a moment, the stamp must still point at the audio,
            # because the LLM node uses it to judge how stale the context is.
            "stamp": self.buffer.newest(),
            "noise_floor_db": floor,
            "noise_level": level,
            "speech_snr_db": snr,
            "speech_now": self.utterance.speech_now,
            "speech_ratio_10s": self.counters.ratio(),
            "seconds_since_speech": self.counters.since_speech(),

            # Whatever text is ready. Empty when nobody has spoken, or when
            # not engaged, or while a transcription is still running.
            "transcript": self.stt.latest() if self.stt else "",

            # Echoed from state.py, not measured. Whichever value was in
            # force when these readings were taken.
            "engaged": state.engaged(),
            "robot_speaking": state.robot_speaking(),
        }

        if self.verbose:
            n_bg, n_sp = self.buffer.counts()
            f = "None" if floor is None else f"{floor:.1f}"
            s = "None" if snr is None else f"{snr:+.1f}"
            talking = "TALKING" if self.utterance.speech_now else "       "
            r = self.counters.ratio()
            since = self.counters.since_speech()
            r_s = " -  " if r is None else f"{r:.2f}"
            q_s = "  -  " if since is None else f"{since:4.1f}s"
            flags = ""
            if state.engaged():
                flags += " ENGAGED"
            if state.robot_speaking():
                flags += " ROBOT"
            print(f"  BUILD {self.builds:3d}  floor {f:>7} dBFS  {str(level):<9} "
                  f"snr {s:>6} dB   {talking}   ratio {r_s}  quiet {q_s}{flags}")
        return audio

    def _suppressed(self):
        # Nothing was measured this second. robot_speaking being true is
        # what tells the LLM why.
        audio = {
            "stamp": time.time(),      # the stores are not advancing
            "noise_floor_db": None,
            "noise_level": None,
            "speech_snr_db": None,
            "speech_now": None,
            "speech_ratio_10s": None,
            "seconds_since_speech": None,
            "transcript": "",
            "engaged": state.engaged(),
            "robot_speaking": state.robot_speaking(),
        }
        if self.verbose:
            print(f"  BUILD {self.builds:3d}  not measuring, robot is speaking")
        return audio