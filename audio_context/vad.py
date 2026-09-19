# Goal: decide, for each 32 ms frame, whether it is a voice or background.
# Receives blocks, never opens the microphone, and reads the raw waveform
# rather than the dB values from loudness.py.
#
#   voice = vad.from_block(block)    # True per frame = a person talking
#
# The caller owns the stream. processor.py opens one AudioCapture and hands
# the same block to this module and to loudness.py, so both read identical
# frames and their results line up index by index.
#
# Uses Silero, which ships inside faster-whisper as a small ONNX file, so
# there is no PyTorch and no separate download.
#
# The block is scaled to a fixed level before being scored. Silero is
# level-sensitive, and distant speech arriving at -60 dBFS is much easier to
# spot once brought up: in testing, detection at 2 dB SNR went from 69% to
# 93%, with no increase in false positives on footsteps, carts or hum. Only
# the detector sees the scaled copy; loudness.py measures the real level.

import numpy as np

_model = None


def _get_model():
    # Loaded once and reused. Loading per call would make this hundreds of
    # times slower and is the usual reason a VAD "runs too slowly".
    global _model
    if _model is None:
        from faster_whisper.vad import get_vad_model
        _model = get_vad_model()
    return _model


def normalise(samples, target_db):
    # Scale a block to a fixed RMS level. Whole block rather than per frame,
    # so the relative levels within it are preserved.
    if target_db is None:
        return samples
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    if rms < 1e-9:
        return samples          # digital silence, nothing to scale
    gain = 10.0 ** (target_db / 20.0) / rms
    return np.clip(samples * gain, -1.0, 1.0).astype(np.float32)


def warm_up():
    # Load the model and run one frame through it, so the first real block
    # is not delayed by a one-off setup cost.
    _get_model()(np.zeros(512, dtype=np.float32))


def is_voice(frames, threshold=0.5):
    # True where the frame is a person talking, False where it is the room.
    # Silero requires exactly 512-sample frames at 16 kHz, which is why
    # config fixes frame_samples at 512.
    if frames.size == 0:
        return np.empty(0, dtype=bool)
    if frames.shape[1] != 512:
        raise ValueError(
            f"Silero needs 512-sample frames, got {frames.shape[1]}. "
            "Check frame_samples in config.py."
        )
    flat = np.ascontiguousarray(frames.reshape(-1), dtype=np.float32)
    scores = np.asarray(_get_model()(flat)).reshape(-1)
    return scores > threshold


def from_block(block, frame_samples=None, threshold=None, target_db=...):
    # True/False per frame of a block from capture.py.
    from audio_context.config import CONFIG
    from audio_context.loudness import to_frames
    if frame_samples is None:
        frame_samples = CONFIG["frame_samples"]
    if threshold is None:
        threshold = CONFIG["vad_threshold"]
    if target_db is ...:
        target_db = CONFIG["vad_normalise_db"]

    samples = normalise(block.samples, target_db)
    return is_voice(to_frames(samples, frame_samples), threshold)