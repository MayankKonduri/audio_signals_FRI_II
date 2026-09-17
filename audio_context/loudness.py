# Goal: measure how loud each 32 ms slice of audio is, in dBFS.
# Takes arrays, returns arrays. Never touches the microphone.
#
#   frames = to_frames(block.samples, 512)
#   db = rms_db(frames)          # one value per frame, always negative

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
    if frames.size == 0:
        return np.empty(0, dtype=np.float64)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    return 20.0 * np.log10(rms + 1e-12)


def is_real(db):
    # True for frames loud enough to be a measurement of something.
    return db > SILENCE_DB


def summarize(db, mask=None):
    # Median dB of the frames selected by mask, ignoring digital silence.
    # Median rather than mean so one loud clatter cannot move the answer.
    # Returns None when nothing usable is left, which the caller must handle.
    sel = is_real(db) if mask is None else (is_real(db) & mask)
    if not np.any(sel):
        return None
    return float(np.median(db[sel]))