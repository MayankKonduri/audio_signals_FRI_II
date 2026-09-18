# Goal: measure how loud each 32 ms slice of audio is, in dBFS.
# Receives blocks, never opens the microphone, and makes no judgement about
# what the sound is - that belongs to vad.py.
#
#   db = loudness.from_block(block)   # one dB value per 32 ms frame
#
# The caller owns the stream. processor.py opens one AudioCapture and hands
# the same block to this module and to vad.py, so both read identical frames
# and their results line up index by index.

import numpy as np

# Below this a frame is digital silence, not a reading of the room. Some
# audio backends emit exact zeros, which would otherwise report around
# -240 dB and drag any average down to a level no real room reaches.
SILENCE_DB = -100.0


def to_frames(samples, frame_samples):
    # Reshape into (n, frame_samples). Any leftover tail is dropped, since a
    # partial frame would read quieter than it really is.
    n = len(samples) // frame_samples
    if n == 0:
        return np.empty((0, frame_samples), dtype=np.float32)
    return samples[:n * frame_samples].reshape(n, frame_samples)


def rms_db(frames):
    # Root mean square per frame, converted to decibels relative to full
    # scale. 0 dBFS is the loudest a sample can be, so values are negative.
    # float64 because a quiet room squares down to around 1e-7.
    if frames.size == 0:
        return np.empty(0, dtype=np.float64)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    return 20.0 * np.log10(rms + 1e-12)


def is_real(db):
    # True for frames loud enough to be a measurement of something.
    return db > SILENCE_DB


def from_block(block, frame_samples=None):
    # One dB value per frame of a block from capture.py.
    if frame_samples is None:
        from audio_context.config import CONFIG
        frame_samples = CONFIG["frame_samples"]
    return rms_db(to_frames(block.samples, frame_samples))