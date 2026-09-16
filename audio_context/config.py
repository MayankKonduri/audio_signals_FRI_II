# Goal: hold every number the audio node uses, so nothing else in the codebase
# contains a threshold, a duration or a sample count. Values marked TUNE have
# to be measured on the robot before they mean anything.

CONFIG = {

    # --- capture ---------------------------------------------------
    "sample_rate":         16000,  # Silero and the STT models expect this
    "frame_samples":       512,    # 32 ms, also fixed by Silero
    "block_samples":       2048,   # 128 ms per callback

    # Measured on Windows with tools/check_devices.py. Re-run that on the lab
    # machines: the name may differ and input_backend should become None.
    "input_device":        "USBAudio1.0",
    "input_backend":       "WASAPI",
    "input_channels":      2,
    "use_channel":         0,      # the mic is mono, duplicated to stereo
    "capture_sample_rate": 48000,
    "warmup_blocks":       2,      # some backends open with digital silence

    # --- voice detection -------------------------------------------
    "vad_threshold":       0.5,    # TUNE against the Hallway clip

    # --- noise floor -----------------------------------------------
    "quiet_max_db":        -50.0,  # TUNE. Placeholders, not measurements.
    "moderate_max_db":     -40.0,  # TUNE. Read the floor in all three
                                   # locations with the robot running, then
                                   # put the cutoffs in the gaps.
    "level_hysteresis_db": 2.0,    # stops the label flickering at a cutoff
    "floor_window_sec":    30.0,
    "floor_min_frames":    100,    # report no floor until this much quiet
    "min_speech_frames":   3,      # fewer than this and SNR is mostly chance

    # --- utterances ------------------------------------------------
    "utterance_open_sec":  0.25,
    "utterance_close_sec": 0.50,
    "utterance_max_sec":   15.0,

    # --- output timing ---------------------------------------------
    "window_sec":          1.0,
    "ratio_window_sec":    10.0,   # rename speech_ratio_10s if you change this

    # --- ring buffer, in memory only -------------------------------
    "ring_buffer_sec":     20.0,   # long enough for the longest utterance
    "stt_lookback_sec":    10.0,   # how far back to transcribe when engaged

    # --- gated path, only runs when engaged ------------------------
    "stt_model":           "tiny.en",
    "stt_device":          "cpu",  # keep on cpu even with a GPU, or timings
                                   # here will not match the robot
    "stt_compute_type":    "int8",
    "stt_threads":         1,
    "tone_enabled":        True,
    "pitch_floor_hz":      75.0,   # parselmouth search range for adult speech
    "pitch_ceiling_hz":    500.0,

    # --- robot hearing itself --------------------------------------
    "robot_speaking_tail_sec": 0.5,
}


def derive(cfg=CONFIG):
    # Anything that follows from the values above, so it is never written twice.
    sr, fs = cfg["sample_rate"], cfg["frame_samples"]
    frame_sec = fs / sr
    return {
        "frame_sec":         frame_sec,
        "frames_per_block":  cfg["block_samples"] // fs,
        "frames_per_window": round(cfg["window_sec"] / frame_sec),
        "open_frames":       round(cfg["utterance_open_sec"] / frame_sec),
        "close_frames":      round(cfg["utterance_close_sec"] / frame_sec),
        "max_utt_frames":    round(cfg["utterance_max_sec"] / frame_sec),
        "floor_frames":      round(cfg["floor_window_sec"] / frame_sec),
        "ratio_frames":      round(cfg["ratio_window_sec"] / frame_sec),
        "ring_samples":      round(cfg["ring_buffer_sec"] * sr),
        "lookback_samples":  round(cfg["stt_lookback_sec"] * sr),
    }


def check(cfg=CONFIG):
    # Catches settings that are wrong but would not crash.
    problems = []
    if cfg["block_samples"] % cfg["frame_samples"]:
        problems.append("block_samples must be a multiple of frame_samples")
    if cfg["frame_samples"] != 512:
        problems.append("Silero expects 512-sample frames")
    if cfg["sample_rate"] != 16000:
        problems.append("Silero and the STT models expect 16 kHz")
    if cfg["quiet_max_db"] >= cfg["moderate_max_db"]:
        problems.append("quiet_max_db must be below moderate_max_db")
    if cfg["ring_buffer_sec"] < cfg["utterance_max_sec"]:
        problems.append("ring_buffer_sec is shorter than utterance_max_sec, so long "
                        "utterances cannot be recovered for transcription")
    if cfg["ring_buffer_sec"] < cfg["stt_lookback_sec"]:
        problems.append("ring_buffer_sec is shorter than stt_lookback_sec, so the "
                        "retroactive window reaches past the start of the buffer")
    if cfg["utterance_close_sec"] <= cfg["utterance_open_sec"]:
        problems.append("utterance_close_sec should exceed utterance_open_sec, or "
                        "utterances will split on the gaps between words")
    csr = cfg["capture_sample_rate"]
    if csr is not None and csr % cfg["sample_rate"]:
        problems.append(f"capture_sample_rate {csr} is not a whole multiple of "
                        f"{cfg['sample_rate']}, so resampling is less exact")
    frames_per_window = round(cfg["window_sec"] / (cfg["frame_samples"] / cfg["sample_rate"]))
    if cfg["min_speech_frames"] > frames_per_window:
        problems.append("min_speech_frames exceeds the frames in one window, so "
                        "speech_snr_db can never be reported")
    return problems


if __name__ == "__main__":
    for p in check():
        print("CONFIG PROBLEM:", p)
    d = derive()
    print(f"{'frame length':22s}{d['frame_sec']*1000:8.1f} ms")
    print(f"{'frames per callback':22s}{d['frames_per_block']:8d}")
    print(f"{'frames per window':22s}{d['frames_per_window']:8d}")
    print(f"{'to open an utterance':22s}{d['open_frames']:8d} frames")
    print(f"{'to close one':22s}{d['close_frames']:8d} frames")
    print(f"{'max utterance':22s}{d['max_utt_frames']:8d} frames")
    print(f"{'noise floor history':22s}{d['floor_frames']:8d} frames")
    print(f"{'speech ratio history':22s}{d['ratio_frames']:8d} frames")
    print(f"{'ring buffer':22s}{d['ring_samples']:8d} samples")