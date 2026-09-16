# Goal: turn a microphone or a WAV file into identical blocks of mono 16 kHz
# audio, each stamped with when it was captured. Nothing is written to disk.
#
#   cap = AudioCapture(CONFIG)                       # microphone
#   cap = AudioCapture(CONFIG, wav="clips/x.wav")    # file, same output
#   cap.start()
#   for block in cap.blocks():
#       print(block.t, len(block.samples))
#   cap.stop()

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
        print(f"note: no input device matching {spec!r}, using the system default")
        return None

    # Windows lists one mic under several backends. WASAPI reaches every
    # channel at the device's real rate, where MME caps at 2 and reports 44100.
    wasapi = [m for m in matches if "wasapi" in api_of(m[1]).lower()]
    index, dev = (wasapi or matches)[0]
    print(f"device [{index}] {dev['name']!r} via {api_of(dev)}")
    return index


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

        self._q = queue.Queue(maxsize=64)
        self._stream = None
        self._thread = None
        self._stop = threading.Event()
        self._resampler = None
        self.dropped = 0
        self.source_rate = None
        self.channels = None

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
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def _start_mic(self):
        import sounddevice as sd

        device = resolve_device(self.cfg["input_device"])
        info = sd.query_devices(device if device is not None else sd.default.device[0])

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
                self._q.put_nowait(Block(t, indata.copy()))
            except queue.Full:
                self.dropped += 1

        self._stream = sd.InputStream(
            device=device, channels=self.channels, samplerate=self.source_rate,
            blocksize=in_block, dtype="float32", callback=callback,
        )
        self._stream.start()
        print(f"microphone: {info['name']!r}, {self.channels} ch @ {self.source_rate} Hz "
              f"-> mono @ {self.target_rate} Hz")

    def _start_file(self):
        data, rate = sf.read(self.wav, dtype="float32", always_2d=True)
        self.source_rate = rate
        self.channels = data.shape[1]
        self._resampler = Resampler(rate, self.target_rate)
        in_block = round(self.block_samples * rate / self.target_rate)

        def pump():
            base = time.time()
            for i in range(0, len(data) - in_block + 1, in_block):
                if self._stop.is_set():
                    return
                t = base + i / rate
                if self.realtime:
                    delay = t - time.time()
                    if delay > 0:
                        time.sleep(delay)
                self._q.put(Block(t, data[i:i + in_block]))
            self._q.put(None)

        self._thread = threading.Thread(target=pump, daemon=True)
        self._thread.start()
        print(f"file: {self.wav}, {self.channels} ch @ {rate} Hz, "
              f"{len(data) / rate:.1f}s -> mono @ {self.target_rate} Hz")

    def blocks(self, timeout=1.0):
        # Skips the first blocks: some backends send digital silence while the
        # device spins up, which is not a reading of the room.
        warmup = 0 if self.wav else self.cfg.get("warmup_blocks", 2)
        seen = 0

        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=timeout)
            except queue.Empty:
                continue
            if item is None:
                return

            seen += 1
            mono = to_mono(item.samples)
            resampled = self._resampler(mono)      # always run, to keep state warm

            if seen <= warmup:
                continue
            yield Block(item.t, resampled)