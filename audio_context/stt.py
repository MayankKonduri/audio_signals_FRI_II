# Goal: turn a finished utterance into text, without ever holding up the
# rest of the pipeline. Produces the transcript field.
#
#   stt = Transcriber(CONFIG)
#   stt.request(samples)      # hand it an utterance, returns immediately
#   stt.latest()              # the most recent text, or ""
#
# On its own thread for a reason. Transcription takes a moment, and the
# per-second numbers have to keep flowing while it runs. The builder reads
# whatever text is ready and does not wait.
#
# Only runs when engaged. Most of the time nobody is talking to the robot,
# so this costs nothing, and bystanders who never agreed to anything do not
# get transcribed.

import queue
import threading
import time

import numpy as np

_model = None


def _get_model(cfg):
    # Loaded once. Keep it on the CPU even on a machine with a GPU, or
    # timings here will not match the robot.
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(
            cfg["stt_model"],
            device=cfg["stt_device"],
            compute_type=cfg["stt_compute_type"],
            cpu_threads=cfg["stt_threads"],
        )
    return _model


class Transcriber:

    def __init__(self, cfg, verbose=False):
        self.cfg = cfg
        self.rate = cfg["sample_rate"]
        self.verbose = verbose

        # Depth one: if a request is already waiting, a newer utterance
        # replaces it. Falling behind is better than queueing up stale audio
        # and reporting a transcript from a minute ago.
        self._queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._text = ""
        self._text_t = None        # when the transcribed audio happened
        self._stop = threading.Event()
        self._thread = None
        self.done = 0              # utterances transcribed
        self.skipped = 0           # requests dropped because one was in flight
        self.available = None      # set by warm_up: False if the model failed

    def start(self):
        self._thread = threading.Thread(target=self._work, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def warm_up(self):
        # Load the model before the first real utterance, so the first
        # transcript is not delayed by a one-off setup cost.
        #
        # Never fatal. Transcription is one optional field; if the model
        # cannot be fetched, the node should carry on measuring the room
        # rather than refusing to start. Returns whether it is usable.
        try:
            _get_model(self.cfg)
            self.available = True
        except Exception as e:
            self.available = False
            print(f"transcription unavailable, carrying on without it: "
                  f"{type(e).__name__}")
        return self.available

    def request(self, samples, when=None):
        # Hand over an utterance. Returns immediately.
        if self.available is False:
            return False          # no model, so nothing to hand it to
        if samples is None or len(samples) < self.rate * 0.2:
            return False          # under 0.2 s, nothing worth transcribing
        try:
            self._queue.put_nowait((np.asarray(samples, dtype=np.float32),
                                    when if when is not None else time.time()))
            return True
        except queue.Full:
            self.skipped += 1
            return False

    def latest(self):
        with self._lock:
            return self._text

    def latest_time(self):
        with self._lock:
            return self._text_t

    def clear(self):
        # Called when the robot disengages, so an old transcript does not
        # linger into the next interaction.
        with self._lock:
            self._text = ""
            self._text_t = None

    def _work(self):
        while not self._stop.is_set():
            try:
                samples, when = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            try:
                model = _get_model(self.cfg)
                segments, _ = model.transcribe(samples, language="en",
                                               beam_size=1)
                text = " ".join(s.text.strip() for s in segments).strip()
            except Exception as e:
                # One failure should not kill the thread, or every later
                # utterance would be lost too.
                print(f"transcription failed: {type(e).__name__}")
                continue

            took = time.perf_counter() - t0
            audio_sec = len(samples) / self.rate
            self.done += 1

            with self._lock:
                self._text = text
                self._text_t = when

            if self.verbose:
                print(f"  TRANSCRIPT ({audio_sec:.1f}s audio in {took:.1f}s): "
                      f"{text!r}")