# Yamaha DTX-pro drum module
from dataclasses import dataclass
from enum import Enum

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

@dataclass
class DrumPad:
    pad: DtxPad
    note_numbers: list[int]

pads = [
    DrumPad(DtxPad.SNARE, [38, 37, 40]),
    DrumPad(DtxPad.TOM1, [48]),
    DrumPad(DtxPad.TOM2, [47]),
    DrumPad(DtxPad.TOM3, [43]),
    DrumPad(DtxPad.CRASH1, [49, 55, 59]),
    DrumPad(DtxPad.CRASH2, [16, 17, 20, 57]),
    DrumPad(DtxPad.RIDE, [49, 55, 59]),
    DrumPad(DtxPad.HAT, [46, 77, 78]),
]

def struck_pad(note_number) -> DtxPad | None:
    if any(note_number in (drum_pad := p).note_numbers for p in pads):
        return drum_pad.pad
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
