# Goal: the one place audio enters the program. Turns a microphone or a WAV
# file into identical blocks of mono 16 kHz audio, each stamped with when it
# was captured. Nothing is written to disk.
#
# The microphone can only be held by one thing at a time, so nothing else
# opens it. Every consumer reads from here instead.
#
# One consumer:
#   with AudioCapture(CONFIG) as cap:            # or wav="clips/x.wav"
#       for block in cap.blocks():
#           print(block.t, len(block.samples))
#
# Several at once, each receiving every block, so loudness.py and vad.py
# analyse exactly the same audio rather than splitting it between them:
#   with AudioCapture(CONFIG) as cap:
#       q1, q2 = cap.subscribe(), cap.subscribe()
#       cap.run_in_thread()          # None arrives on each queue at the end

import queue
import threading
import time
from collections import namedtuple
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import firwin, lfilter, resample_poly

Block = namedtuple("Block", "t samples")


def resolve_device(spec):
    # spec is None (system default), an int index, or a name fragment.
    # Matching ignores spaces and case, so "USB Audio" finds both
    # "Microphone (USBAudio1.0)" on Windows and "USB Audio Device" on Linux.
    # If nothing matches, fall back to the system default rather than failing.
    import sounddevice as sd

    if spec is None or isinstance(spec, int):
        return spec

    hostapis = sd.query_hostapis()
    wanted = spec.lower().replace(" ", "")

    def api_of(d):
        return hostapis[d["hostapi"]]["name"]

    matches = [
        (i, d) for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
        and wanted in d["name"].lower().replace(" ", "")
    ]

    if not matches:
        return None      # not found, so the system default is used

    # Windows lists one mic under several backends. WASAPI reaches every
    # channel at the device's real rate, where MME caps at 2 and reports 44100.
    wasapi = [m for m in matches if "wasapi" in api_of(m[1]).lower()]
    return (wasapi or matches)[0][0]


class Resampler:
    # Downsamples to the target rate. Keeps filter state across blocks, or
    # every block boundary would leave a click for the voice detector to find.

    def __init__(self, from_rate, to_rate):
        self.from_rate, self.to_rate = from_rate, to_rate
        self.passthrough = from_rate == to_rate
        self.integer = (not self.passthrough) and from_rate % to_rate == 0

        if self.passthrough:
            return

        if self.integer:
            self.factor = from_rate // to_rate
            self.taps = firwin(8 * self.factor + 1, 1.0 / self.factor).astype(np.float64)
            self.zi = np.zeros(len(self.taps) - 1)
            self.phase = 0
        else:
            # e.g. 44100 -> 16000. resample_poly has no state, so block edges
            # are slightly imperfect. Prefer an integer-multiple rate.
            g = gcd(from_rate, to_rate)
            self.up, self.down = to_rate // g, from_rate // g

    def __call__(self, x):
        if self.passthrough:
            return x.astype(np.float32, copy=False)

        if not self.integer:
            return resample_poly(x, self.up, self.down).astype(np.float32)

        y, self.zi = lfilter(self.taps, [1.0], x, zi=self.zi)
        idx = np.arange(self.phase, len(y), self.factor)
        self.phase = (self.phase - len(y)) % self.factor   # keep the grid even
        return y[idx].astype(np.float32)


def to_mono(x):
    if x.ndim == 1:
        return x
    if x.shape[1] == 1:
        return x[:, 0]
    return x.mean(axis=1)


class AudioCapture:

    def __init__(self, cfg, wav=None, realtime=True):
        self.cfg = cfg
        self.wav = wav
        self.realtime = realtime            # file mode: pace it like live audio
        self.target_rate = cfg["sample_rate"]
        self.block_samples = cfg["block_samples"]

        self._raw = queue.Queue(maxsize=64)  # undecoded, straight off the device
        self._subs = []                      # one delivery queue per consumer
        self._stream = None
        self._thread = None
        self._pump = None
        self._stop = threading.Event()
        self._resampler = None
        self._seen = 0
        self._warmup = 0 if wav else cfg.get("warmup_blocks", 2)
        self.dropped = 0
        self.source_rate = None     # what the device or file gave us
        self.channels = None
        self.device_name = None     # nothing is printed from here; read these
        self.backend = None         # if a caller wants to report the device

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        if self.wav:
            self._start_file()
        else:
            self._start_mic()
        return self

    def stop(self):
        self._stop.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        for t in (self._thread, self._pump):
            if t is not None:
                t.join(timeout=2.0)
        self._thread = self._pump = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # -- sources -----------------------------------------------------------

    def _start_mic(self):
        import sounddevice as sd

        device = resolve_device(self.cfg["input_device"])
        info = sd.query_devices(device if device is not None else sd.default.device[0])
        self.device_name = info["name"]
        self.backend = sd.query_hostapis()[info["hostapi"]]["name"]

        self.channels = self.cfg["input_channels"] or int(info["max_input_channels"])
        self.source_rate = int(self.cfg["capture_sample_rate"]
                               or info["default_samplerate"])
        self._resampler = Resampler(self.source_rate, self.target_rate)

        # Size the device blocks so they resample to a whole number of samples.
        in_block = round(self.block_samples * self.source_rate / self.target_rate)

        def callback(indata, frames, _time, status):
            if status:
                print(f"audio status: {status}")
            t = time.time() - frames / self.source_rate
            try:
                self._raw.put_nowait(Block(t, indata.copy()))
            except queue.Full:
                self.dropped += 1

        self._stream = sd.InputStream(
            device=device, channels=self.channels, samplerate=self.source_rate,
            blocksize=in_block, dtype="float32", callback=callback,
        )
        self._stream.start()

    def _start_file(self):
        data, rate = sf.read(self.wav, dtype="float32", always_2d=True)
        self.device_name = self.wav
        self.source_rate = rate
        self.channels = data.shape[1]
        self._resampler = Resampler(rate, self.target_rate)
        in_block = round(self.block_samples * rate / self.target_rate)

        def feed():
            base = time.time()
            for i in range(0, len(data) - in_block + 1, in_block):
                if self._stop.is_set():
                    return
                t = base + i / rate
                if self.realtime:
                    delay = t - time.time()
                    if delay > 0:
                        time.sleep(delay)
                self._raw.put(Block(t, data[i:i + in_block]))
            self._raw.put(None)             # end of file

        self._thread = threading.Thread(target=feed, daemon=True)
        self._thread.start()

    # -- decoding ----------------------------------------------------------

    def _next(self, timeout):
        # One finished block, or None when the source runs out.
        while not self._stop.is_set():
            try:
                item = self._raw.get(timeout=timeout)
            except queue.Empty:
                continue
            if item is None:
                return None

            self._seen += 1
            mono = to_mono(item.samples)
            resampled = self._resampler(mono)   # always run, to keep state warm

            # Some backends send digital silence while the device spins up,
            # which is not a reading of the room.
            if self._seen <= self._warmup:
                continue
            return Block(item.t, resampled)
        return None

    # -- one consumer ------------------------------------------------------

    def blocks(self, timeout=1.0):
        # Iterate the stream directly. Use this when one thing needs the
        # audio; use subscribe() plus run() when several do.
        while True:
            block = self._next(timeout)
            if block is None:
                return
            yield block

    # -- several consumers -------------------------------------------------

    def subscribe(self, maxsize=64):
        # Register a consumer and hand back its own queue. Every subscriber
        # receives every block, so two modules reading the same stream see
        # identical audio instead of taking alternate blocks.
        # Call before run(); a queue added later misses what has gone past.
        q = queue.Queue(maxsize=maxsize)
        self._subs.append(q)
        return q

    def run(self, timeout=1.0):
        # Pump blocks to every subscriber until stopped or the source ends,
        # then put None on each queue so consumers know it is over.
        while True:
            block = self._next(timeout)
            for q in self._subs:
                try:
                    q.put_nowait(block)
                except queue.Full:
                    self.dropped += 1
            if block is None:
                return

    def run_in_thread(self, timeout=1.0):
        # run() blocks, so this is the usual way: start pumping and carry on.
        self._pump = threading.Thread(target=self.run, args=(timeout,), daemon=True)
        self._pump.start()
        return self._pump