# Goal: answer one question from the speech store: how far does a voice rise
# above the room? Produces speech_snr_db.
#
#   sp = Speech(CONFIG)
#   snr = sp.snr(buffer.speech_levels(), floor)
#   # 23.4
#
# The raw speech level is not the field, because on its own it means nothing.
# A voice at -36 dBFS in a silent library is close and clear; the same -36 in
# a hallway where the floor is -38 is someone barely audible. Only the gap
# between the two carries any meaning, which is why this needs the floor from
# background.py and returns the difference.

import numpy as np


class Speech:

    def __init__(self, cfg):
        self.min_frames = cfg["min_speech_frames"]

    def level(self, levels):
        # Median level of the speech frames in the store.
        #
        # None below min_frames: a level from one or two frames is mostly
        # chance, since the quiet tail of a word reads very differently from
        # the middle of a vowel.
        if len(levels) < self.min_frames:
            return None
        return round(float(np.median(levels)), 1)

    def snr(self, levels, floor):
        # How far the voice sits above the room, in dB.
        #
        # Decibels are logarithmic, so subtracting is already a ratio. A
        # large number means close and clear, a small one means distant or
        # drowned out.
        #
        # None when either side is missing: no speech to measure, or no floor
        # to measure it against.
        speech = self.level(levels)
        if speech is None or floor is None:
            return None
        return round(speech - floor, 1)