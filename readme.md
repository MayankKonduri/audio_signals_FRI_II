# audio_signals_FRI_II

Audio subsystem for Team Read the Room (FRI 2, Autonomous Intelligent Robotics).

Produces the `"audio"` block of the shared context object, once per second:
room noise level, whether anyone is speaking, how far their voice rises above
the room, and - only once the robot has engaged someone - a transcript and
basic vocal tone.

## Setup

Python 3.10, to match the robot's ROS 2 Humble.

    python3.10 -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

## First run

    python tools/check_devices.py

This finds your USB microphone and prints the three values to paste into
`audio_context/config.py`.

## Layout

    audio_context/config.py   every tunable number lives here
    tools/                    one-off diagnostics
    clips/                    recorded scenario clips for offline testing