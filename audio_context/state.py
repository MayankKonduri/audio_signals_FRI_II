# Goal: hold the two flags that arrive from other nodes, so any module can
# read them without having them threaded through every call.
#
#   node.py, from its ROS subscriber callbacks:
#       state.set_engaged(msg.data)
#       state.set_robot_speaking(msg.data)
#
#   anywhere else:
#       if state.engaged(): ...
#
# Separate from config.py on purpose. Config holds values you tune and then
# leave alone; these change second to second. Keeping them apart means
# config.check() is only ever validating settings.
#
# Without ROS, set them by hand to test the gated paths:
#       state.set_engaged(True)

import time

_engaged = False
_robot_speaking = False
_robot_stopped_at = None       # when it last finished speaking

# Keep ignoring audio for this long after the robot stops, to cover the
# room's echo of its own voice.
TAIL_SEC = 0.5


def set_engaged(value):
    global _engaged
    _engaged = bool(value)


def engaged():
    # True once the robot is in a conversation. Gates transcription and tone,
    # which are the two expensive things in the pipeline.
    return _engaged


def set_robot_speaking(value):
    global _robot_speaking, _robot_stopped_at
    value = bool(value)
    if _robot_speaking and not value:
        _robot_stopped_at = time.time()
    _robot_speaking = value


def robot_speaking():
    # The raw flag, for echoing into the JSON.
    return _robot_speaking


def ignore_audio(now=None):
    # Whether audio should be discarded right now: the robot is talking, or
    # it just stopped and the room is still ringing with it. This is what the
    # processor checks, rather than the raw flag, or the tail end of the
    # robot's own greeting would register as a person speaking.
    if _robot_speaking:
        return True
    if _robot_stopped_at is None:
        return False
    return (now or time.time()) - _robot_stopped_at < TAIL_SEC


def reset():
    # For tests, so one case cannot leak into the next.
    global _engaged, _robot_speaking, _robot_stopped_at
    _engaged = False
    _robot_speaking = False
    _robot_stopped_at = None