# Goal: build the audio JSON once per second, by reading the stores that
# processor.py fills and asking each module for its numbers.
#
# Runs on its own schedule. processor.py loops about 8 times a second as
# blocks arrive; this runs once a second, which is the rate the JSON is
# published at. Keeping them apart means a slow build never stalls capture.
#
# Fields still to come: the gated transcript and tone, then robot_speaking
# and engaged, which arrive over ROS.

from audio_context.background import Background
from audio_context.speech import Speech


class Builder:

    def __init__(self, cfg, buffer, utterance, counters, verbose=False):
        self.cfg = cfg
        self.buffer = buffer       # the stores processor.py is filling
        self.utterance = utterance # the tracker processor.py is feeding
        self.counters = counters   # the rolling counts it also feeds
        self.verbose = verbose
        self.background = Background(cfg)
        self.speech = Speech(cfg)
        self.builds = 0

    def build(self):
        # One audio object. Called once per second by main.py.
        self.builds += 1

        floor, level = self.background.read(self.buffer.background_levels())
        snr = self.speech.snr(self.buffer.speech_levels(), floor)

        audio = {
            "noise_floor_db": floor,
            "noise_level": level,
            "speech_snr_db": snr,
            "speech_now": self.utterance.speech_now,
            "speech_ratio_10s": self.counters.ratio(),
            "seconds_since_speech": self.counters.since_speech(),
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
            print(f"  BUILD {self.builds:3d}  floor {f:>7} dBFS  {str(level):<9} "
                  f"snr {s:>6} dB   {talking}   ratio {r_s}  quiet {q_s}")
        return audio