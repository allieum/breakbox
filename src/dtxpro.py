# Yamaha DTX-pro drum module
import time
from dataclasses import dataclass, field
from enum import Enum
from sample import Sample

from utility import get_logger

logger = get_logger(__name__)


class DtxPad(Enum):
    SNARE = 0
    TOM1 = 1
    TOM2 = 2
    TOM3 = 3
    CRASH1 = 4
    CRASH2 = 5
    RIDE = 6
    HAT = 7
    CLOSED_HAT = 8


HIT_TIMEOUT = 0.25
ROLL_THRESHOLD = 4
ROLL_DEBOUNCE = 2
PRO_HIT_COUNT = 100


@dataclass
class DrumPad:
    pad: DtxPad
    note_numbers: list[int]
    last_hit: float = field(default=0)
    last_roll: float = field(default=0)
    hit_count: int = field(default=0)

    def register_hit(self):
        self.last_hit = time.time()
        self.hit_count += 1

    def roll_detected(self):
        roll = self.hit_count >= ROLL_THRESHOLD
        if roll and time.time() - self.last_roll > ROLL_DEBOUNCE:
            self.last_roll = time.time()
            return True
        return False

    def update(self):
        if time.time() - self.last_hit > HIT_TIMEOUT:
            self.hit_count = 0

selected_sample: Sample | None = None

pads = [
    DrumPad(DtxPad.SNARE, [38, 37, 40]),
    DrumPad(DtxPad.TOM1, [48]),
    DrumPad(DtxPad.TOM2, [47]),
    DrumPad(DtxPad.TOM3, [43]),
    DrumPad(DtxPad.CRASH1, [49, 55, 59]),
    DrumPad(DtxPad.CRASH2, [16, 17, 20, 57]),
    DrumPad(DtxPad.RIDE, [51, 52, 53]),
    DrumPad(DtxPad.HAT, [46, 77, 78, 79]),
    DrumPad(DtxPad.CLOSED_HAT, [42]),
]


def update():
    for pad in pads:
        pad.update()


def total_hit_count():
    return sum(pad.hit_count for pad in pads)


def hit_count(dtx_pad: DtxPad):
    for pad in pads:
        if pad.pad == dtx_pad:
            return pad.hit_count
    return None


def struck_pad(note_number) -> DrumPad | None:
    if any(note_number in (drum_pad := p).note_numbers for p in pads):
        drum_pad.register_hit()
        logger.info(f"got dtx note {note_number}")
        return drum_pad
    return None


def kit_index(prog_num):
    match bank_lsb:
        case 0:
            offset = 0
        case 1:
            offset = 40
        case 2:
            offset = 40 + 100
        case _:
            offset = 0
    return prog_num + offset


bank_lsb = 0


def update_bank(cc_num, cc_val):
    global bank_lsb
    if cc_num == 32:
        bank_lsb = cc_val
