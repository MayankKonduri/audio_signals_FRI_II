# Goal: build the audio JSON once per second, by reading the stores that
# processor.py fills and asking each module for its numbers.
#
# Runs on its own schedule. processor.py loops about 8 times a second as
# blocks arrive; this runs once a second, which is the rate the JSON is
# published at. Keeping them apart means a slow build never stalls capture.
#
# Nothing is built yet. The modules get called from here as they are
# written: background.py for the noise floor, speech.py for the speech
# level, then the utterance tracker and the counters.


class Builder:

    def __init__(self, cfg, buffer, verbose=False):
        self.cfg = cfg
        self.buffer = buffer       # the stores processor.py is filling
        self.verbose = verbose
        self.builds = 0

    def build(self):
        # One audio object. Called once per second by main.py.
        self.builds += 1
        if self.verbose:
            print("Hello World")
        return {}