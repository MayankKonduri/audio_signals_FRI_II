# Goal: say whether the microphone named in config.py is plugged in and found.
# No recording happens here - tools/test_capture.py does that.
#
#   python tools/check_devices.py           is the configured mic there?
#   python tools/check_devices.py --list    show every input device

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_context.config import CONFIG

try:
    import sounddevice as sd
except Exception as e:
    sys.exit(f"sounddevice is not working: {e}\n"
             "On Linux you may need: sudo apt install libportaudio2")


def inputs():
    apis = sd.query_hostapis()
    return [(i, d, apis[d["hostapi"]]["name"].replace("Windows ", ""))
            for i, d in enumerate(sd.query_devices())
            if d["max_input_channels"] > 0]


def show(rows):
    print(f"{'idx':>4}  {'ch':>3}  {'rate':>9}  {'backend':<12} name")
    print("-" * 74)
    for i, d, api in rows:
        print(f"{i:>4}  {d['max_input_channels']:>3}  "
              f"{int(d['default_samplerate']):>7} Hz  {api:<12} {d['name']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    rows = inputs()
    if not rows:
        print("No input devices at all. Is anything plugged in?")
        return 1

    if args.list:
        show(rows)
        return 0

    wanted = CONFIG["input_device"]
    if wanted is None:
        print("config uses the system default microphone, so there is nothing "
              "to look up. Run tools/test_capture.py.")
        return 0

    # Same rule capture.py uses: substring, ignoring spaces and case.
    key = str(wanted).lower().replace(" ", "")
    matches = [r for r in rows if key in r[1]["name"].lower().replace(" ", "")]

    print(f"config asks for: {wanted!r}\n")

    if not matches:
        print("NOT FOUND\n")
        print("Nothing matches that name, so the system default would be used")
        print("instead. Either the mic is unplugged, or its name here differs.\n")
        show(rows)
        print("\nPick a fragment of the right name and put it in input_device "
              "in config.py.")
        return 1

    # Windows lists one mic under several backends; WASAPI is the one to use.
    wasapi = [r for r in matches if "wasapi" in r[2].lower()]
    chosen = (wasapi or matches)[0]

    print(f"FOUND, {len(matches)} matching entr{'y' if len(matches)==1 else 'ies'}:\n")
    show(matches)
    i, d, api = chosen
    print(f"\ncapture.py will use [{i}] {d['name']!r} via {api}, "
          f"{d['max_input_channels']} ch @ {int(d['default_samplerate'])} Hz")

    rate = int(CONFIG["capture_sample_rate"] or d["default_samplerate"])
    if rate % CONFIG["sample_rate"]:
        print(f"Note: {rate} Hz is not a whole multiple of "
              f"{CONFIG['sample_rate']}, so resampling is less exact. Harmless.")

    print("\nNext: python tools/test_capture.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())