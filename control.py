from adafruit_blinka.board.raspberrypi.raspi_4b import D24
import board
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
    encoderA = digitalio.DigitalInOut(board.D24)
    encoderB = digitalio.DigitalInOut(board.D25)
    encoder = RotaryEncoder(encoderA, encoderB)

def update():
    if encoder is None:
        return
    match encoder.poll():
        case RotaryEncoder.Direction.CLOCK:
            logger.info(f"turned clockwise")
        case RotaryEncoder.Direction.COUNTERCLOCK:
            logger.info(f"turned counterclockwise")
        case None:
            pass
        case _:
            logger.error("shouldn't happen")

    

class RotaryEncoder:

    Direction = Enum('Direction', ['CLOCK', 'COUNTERCLOCK'])

    
    def __init__(self, pinA, pinB):
        self.pinA = pinA
        self.pinB = pinB
        self.pinA.direction = digitalio.Direction.INPUT
        self.pinB.direction = digitalio.Direction.INPUT
        self.pinA.pull = digitalio.Pull.UP
        self.pinB.pull = digitalio.Pull.UP
        self.prev_pos = 0
        self.flags = 0

        if not self.pinA.value:
            self.prev_pos |= 1
        if not self.pinB.value:
            self.prev_pos |= 2

    def poll(self):
        action = None
        pos = 0
        if not self.pinA.value:
            pos |= 1
        if not self.pinB.value:
            pos |= 2

        logger.info(f'encoder position: {pos:x}')
        if pos != self.prev_pos:
            if self.prev_pos == 0x00:
                if pos == 0x01:
                    self.flags |= 1
                if pos == 0x02:
                    self.flags |= 2
            if pos == 0x03:
                self.flags |= (1 << 4)
            elif pos == 0x00:
                if self.prev_pos == 0x02:
                    self.flags |= (1 << 2)
                elif self.prev_pos == 0x01:
                    if bit_is_set(self.flags, 0) and (bit_is_set(self.flags, 2) or bit_is_set(self.flags, 4)):
                        action = self.Direction.CLOCK
                    elif (bit_is_set(self.flags, 2) and (bit_is_set(self.flags, 0) or bit_is_set(self.flags, 4))):
                        action = self.Direction.CLOCK
                    elif (bit_is_set(self.flags, 1) and (bit_is_set(self.flags, 3) or bit_is_set(self.flags, 4))):
                        action = self.Direction.COUNTERCLOCK
                    elif (bit_is_set(self.flags, 3) and (bit_is_set(self.flags, 1) or bit_is_set(self.flags, 4))):
                        action = self.Direction.COUNTERCLOCK

                    self.flags = 0
        self.prev_pos = pos
        return action


def bit_is_set(n, i):
    return n & (1 << i) != 0
