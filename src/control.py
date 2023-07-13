# import board
import digitalio
from enum import Enum
import utility

logger = utility.get_logger(__name__)

# last_position = None
# def update():
#     global last_position
#     position = -encoder.position
#     if position != last_position:
#         last_position = position
#         logger.info(f"encoder position {position}")


encoder = None


def init():
    global encoder
    # encoderA = digitalio.DigitalInOut(board.D24)
    # encoderB = digitalio.DigitalInOut(board.D25)
    # encoder = RotaryEncoder(encoderA, encoderB)


def update():
    if encoder is None:
        return
    encoder.poll()
    # match encoder.poll():
    #     case RotaryEncoder.Direction.CLOCK:
    #         logger.info(f"turned clockwise {encoder.value()}")
    #     case RotaryEncoder.Direction.COUNTERCLOCK:
    #         logger.info(f"turned counterclockwise {encoder.value()}")
    #     case None:
    #         pass
    #     case _:
    #         logger.error("shouldn't happen")


class RotaryEncoder:
    Direction = Enum('Direction', ['CLOCK', 'COUNTERCLOCK'])
    POLL_TIME = 0.010

    def __init__(self, pinA, pinB):
        self.pinA = pinA
        self.pinB = pinB
        self.pinA.direction = digitalio.Direction.INPUT
        self.pinB.direction = digitalio.Direction.INPUT
        self.pinA.pull = digitalio.Pull.UP
        self.pinB.pull = digitalio.Pull.UP
        self.prev_pos = 0
        self.flags = 0
        self.counter = 0
        self.last = self.pinA.value
        self.last_poll = None

        if not self.pinA.value:
            self.prev_pos |= 1
        if not self.pinB.value:
            self.prev_pos |= 2

    def value(self):
        return self.counter // 2

    def poll(self):
        pos = 0
        if not self.pinA.value:
            pos |= 1
        if not self.pinB.value:
            pos |= 2

        clkstate = self.pinA.value
        dtstate = self.pinB.value
        if clkstate != self.last:
            if dtstate != clkstate:
                self.counter -= 1
            else:
                self.counter += 1
            logger.info(f"encoder: {self.counter}")
        self.last = clkstate
        return self.value()


def bit_is_set(n, i):
    return n & (1 << i) != 0
