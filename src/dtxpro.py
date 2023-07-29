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
]

def struck_pad(note_number) -> DtxPad | None:
    if any(note_number in (drum_pad := p).note_numbers for p in pads):
        return drum_pad.pad
    return None
