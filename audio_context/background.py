# Goal: answer one question from the background store: how loud is this room
# when nobody is talking? Produces noise_floor_db and noise_level.
#
#   bg = Background(CONFIG)
#   floor, level = bg.read(buffer.background_levels())
#   # -73.1, "quiet"
#
# Background does not mean quiet. A corridor of footsteps and a rolling cart
# is all background, and reporting that as loud is the point of the field.

import numpy as np


class Background:

    def __init__(self, cfg):
        self.quiet_max = cfg["quiet_max_db"]
        self.moderate_max = cfg["moderate_max_db"]
        self.hysteresis = cfg["level_hysteresis_db"]
        self.min_frames = cfg["floor_min_frames"]
        self._last = None          # previous label, for hysteresis

    def floor(self, levels):
        # The room's level: the median of every background frame in the store.
        #
        # A median rather than a mean, so one door slam among hundreds of
        # frames cannot move the room's level, while a cart rolling past for
        # ten seconds can.
        #
        # None rather than a guess below min_frames. A floor from thirty
        # frames is mostly chance, and telling the LLM that a library is
        # noisy is worse than telling it nothing.
        if len(levels) < self.min_frames:
            return None
        return round(float(np.median(levels)), 1)

    def label(self, floor):
        # One word for the LLM, from the two cutoffs in config.
        #
        # Each cutoff shifts by the hysteresis amount against whichever band
        # we are already in, so a floor sitting on a boundary does not flip
        # the word every second as it drifts a decibel back and forth.
        if floor is None:
            self._last = None
            return None

        quiet_max, moderate_max = self.quiet_max, self.moderate_max
        h = self.hysteresis
        if self._last == "quiet":
            quiet_max += h                    # harder to stop being quiet
        elif self._last == "loud":
            moderate_max -= h                 # harder to stop being loud
        elif self._last == "moderate":
            quiet_max -= h
            moderate_max += h

        if floor < quiet_max:
            self._last = "quiet"
        elif floor < moderate_max:
            self._last = "moderate"
        else:
            self._last = "loud"
        return self._last

    def read(self, levels):
        # Both fields at once, which is how audio_builder calls it.
        floor = self.floor(levels)
        return floor, self.label(floor)