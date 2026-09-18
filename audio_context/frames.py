# Goal: turn the two parallel arrays from processor.py into one annotated
# record per frame, then split those into the two piles the next modules read.
#
#   Frame(t=1789084800.5, db=-65.2, voice=False)
#
# Each frame carries when it happened, how loud it was, and whether it was a
# person talking. That is the complete description of 32 ms of audio.
#
#   background, speech = frames.from_result(result, frame_sec)
#
# background -> background.py, for the noise floor
# speech     -> speech.py, for how loud the talking is
#
# A Buffer keeps the recent past of both piles in two SQLite files, so a
# module can look across seconds rather than one block at a time, and so you
# can open the stores and inspect them while it runs:
#
#   buf = Buffer("data", seconds=15.0, frame_sec=0.032)
#   buf.add(background, speech)
#   buf.background()   # every background frame in the buffer
#
# Frames of digital silence are dropped on the way in. A frame of exact
# zeros reads around -240 dB and is not a measurement of the room, so
# letting it into the store would drag the noise floor below anything a
# real room reaches.

import sqlite3
import threading
from collections import namedtuple
from pathlib import Path

import numpy as np

from audio_context.loudness import is_real

Frame = namedtuple("Frame", "t db voice")


def annotate(result, frame_sec):
    # One Frame per 32 ms slice of a block.
    #
    # result.t is the start of the whole block, so each frame steps forward
    # from there. Without this, four frames 128 ms apart would all claim the
    # same instant and seconds_since_speech would be wrong by up to 0.1 s.
    return [
        Frame(result.t + i * frame_sec, float(db), bool(voice))
        for i, (db, voice) in enumerate(zip(result.db, result.voice))
    ]


def split(frames):
    # The two piles. Every frame lands in exactly one, so nothing is lost
    # and nothing is counted twice.
    background = [f for f in frames if not f.voice]
    speech = [f for f in frames if f.voice]
    return background, speech


def from_result(result, frame_sec):
    # Annotate a block and split it in one step. This is what processor.py
    # calls per block.
    return split(annotate(result, frame_sec))


def levels(frames):
    # Just the decibel values, for when a module wants array maths.
    return np.array([f.db for f in frames], dtype=np.float64)


class FrameStore:
    """One pile of frames in a SQLite file, oldest dropped once full.

    A plain list would do the same job, but a file can be opened and queried
    while the node runs, which is worth the small cost when you are trying to
    work out why a reading looks wrong.
    """

    def __init__(self, path, max_frames, reset=True):
        self.path = Path(path)
        self.max_frames = max_frames
        self._lock = threading.Lock()

        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False because capture runs its own thread.
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS frames (t REAL NOT NULL, db REAL NOT NULL)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS frames_t ON frames (t)")

        # This is a rolling window, not a log. Rows left from a previous run
        # would sit in the buffer and report a room that is no longer there.
        if reset:
            self._conn.execute("DELETE FROM frames")
        self._conn.commit()

    def add(self, frames, cutoff=None):
        # Two limits, and both are needed.
        #
        # The frame cap is the FIFO: past max_frames, the oldest rows go.
        # On its own that is not enough for the speech store, because speech
        # frames arrive in bursts. 470 of them gathered from scattered
        # bursts can reach half a minute back, so the store would report a
        # window far longer than intended. The time cutoff fixes that.
        if not frames:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT INTO frames (t, db) VALUES (?, ?)",
                [(f.t, f.db) for f in frames])

            if cutoff is not None:
                self._conn.execute("DELETE FROM frames WHERE t < ?", (cutoff,))

            self._conn.execute(
                "DELETE FROM frames WHERE rowid IN "
                "(SELECT rowid FROM frames ORDER BY t LIMIT MAX(0, "
                " (SELECT COUNT(*) FROM frames) - ?))",
                (self.max_frames,))
            self._conn.commit()

    def all(self, voice):
        # Every frame in the store, oldest first, as Frame objects.
        with self._lock:
            rows = self._conn.execute(
                "SELECT t, db FROM frames ORDER BY t").fetchall()
        return [Frame(t, db, voice) for t, db in rows]

    def levels(self):
        # Just the dB values, which is what a median needs.
        with self._lock:
            rows = self._conn.execute("SELECT db FROM frames ORDER BY t").fetchall()
        return np.array([r[0] for r in rows], dtype=np.float64)

    def span(self):
        with self._lock:
            lo, hi = self._conn.execute(
                "SELECT MIN(t), MAX(t) FROM frames").fetchone()
        return 0.0 if lo is None else hi - lo

    def newest(self):
        with self._lock:
            hi = self._conn.execute("SELECT MAX(t) FROM frames").fetchone()[0]
        return hi

    def __len__(self):
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0]

    def close(self):
        self._conn.close()


class Buffer:
    """Both piles, each in its own store, holding the last `seconds`."""

    def __init__(self, directory, seconds, frame_sec, reset=True):
        self.seconds = seconds
        self.max_frames = round(seconds / frame_sec)
        self.dropped = 0                 # frames of digital silence refused
        d = Path(directory)
        self._bg = FrameStore(d / "background.db", self.max_frames, reset)
        self._sp = FrameStore(d / "voice.db", self.max_frames, reset)

    def add(self, background, speech):
        # Newest stamp in this block sets the window for both stores, so the
        # two always cover the same stretch of time. Taken before filtering,
        # so a block of nothing but silence still advances the window.
        stamps = [f.t for f in background + speech]
        cutoff = max(stamps) - self.seconds if stamps else None

        # Only real readings go in. Dropped frames are counted, because a
        # high count means the microphone is muting itself rather than
        # recording a quiet room.
        keep_bg = [f for f in background if is_real(f.db)]
        keep_sp = [f for f in speech if is_real(f.db)]
        self.dropped += (len(background) - len(keep_bg)) + (len(speech) - len(keep_sp))

        self._bg.add(keep_bg, cutoff)
        self._sp.add(keep_sp, cutoff)

    def background(self):
        return self._bg.all(voice=False)

    def speech(self):
        return self._sp.all(voice=True)

    def background_levels(self):
        return self._bg.levels()

    def speech_levels(self):
        return self._sp.levels()

    def span(self):
        return max(self._bg.span(), self._sp.span())

    def counts(self):
        # (background, speech), so the two stores can be read apart.
        return len(self._bg), len(self._sp)

    def newest(self):
        # Timestamp of the most recent frame in either store, or None when
        # both are empty. This is what stamps the JSON: the time the sound
        # happened, not the time the message was built.
        stamps = [s for s in (self._bg.newest(), self._sp.newest()) if s is not None]
        return max(stamps) if stamps else None

    def __len__(self):
        return len(self._bg) + len(self._sp)

    def close(self):
        self._bg.close()
        self._sp.close()