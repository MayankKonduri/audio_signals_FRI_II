# Turns a microphone or a WAV file into identical blocks of mono 16 kHz audio,
# each stamped with when it was captured.
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


def resolve_device(spec, backend=None, channels=None, samplerate=None):
    # spec is None (system default), an int index, or a name fragment.
    # Indices move between machines, so config holds a name.
    import sounddevice as sd

    if spec is None or isinstance(spec, int):
        return spec

    hostapis = sd.query_hostapis()
    wanted = spec.lower()
    want_api = backend.lower() if backend else None

    def api_of(d):
        return hostapis[d["hostapi"]]["name"]

    matches = [
        (i, d) for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
        and wanted in d["name"].lower()
        and (want_api is None or want_api in api_of(d).lower())
    ]

    if not matches:
        listed = "\n  ".join(
            f"[{i}] {api_of(d):<20} {d['name']}"
            for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0
        )
        raise RuntimeError(
            f"No input device matching name {spec!r}"
            + (f" and backend {backend!r}" if backend else "")
            + f". Available:\n  {listed}"
        )

    # One mic can appear under several backends with the same name. Keeping
    # only those that accept the requested channels and rate separates them.
    if channels or samplerate:
        usable = []
        for i, d in matches:
            try:
                sd.check_input_settings(device=i, channels=channels,
                                        samplerate=samplerate, dtype="float32")
                usable.append((i, d))
            except Exception:
                pass
        if not usable:
            listed = "\n  ".join(
                f"[{i}] {api_of(d):<20} {d['name']}  "
                f"({d['max_input_channels']} ch, {int(d['default_samplerate'])} Hz)"
                for i, d in matches
            )
            raise RuntimeError(
                f"{len(matches)} device(s) match {spec!r}, but none accept "
                f"channels={channels}, samplerate={samplerate}:\n  {listed}\n"
                "Adjust input_channels or capture_sample_rate in config.py."
            )
        matches = usable

    if len(matches) > 1:
        listed = "\n  ".join(f"[{i}] {api_of(d):<20} {d['name']}" for i, d in matches)
        print(f"note: {len(matches)} devices still match, using the first:\n  {listed}")

    index, dev = matches[0]
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


def to_mono(x, use_channel=None):
    if x.ndim == 1:
        return x
    if x.shape[1] == 1:
        return x[:, 0]
    if use_channel is None:
        return x.mean(axis=1)
    return x[:, use_channel]


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

        device = resolve_device(
            self.cfg["input_device"],
            backend=self.cfg.get("input_backend"),
            channels=self.cfg["input_channels"],
            samplerate=self.cfg["capture_sample_rate"],
        )
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
        use_channel = self.cfg["use_channel"]
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
            mono = to_mono(item.samples, use_channel)
            resampled = self._resampler(mono)      # always run, to keep state warm

            if seen <= warmup:
                continue
            yield Block(item.t, resampled)