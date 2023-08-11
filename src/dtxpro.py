# Yamaha DTX-pro drum module
import random
import time
from dataclasses import dataclass, field
from enum import Enum

import keys
import sample
from sample import Sample
from sequence import Sequence
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
        logger.debug(f"got dtx note {note_number}")
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

#
# nonsense pile
#

# spiciness = dtxpro.total_hit_count() / dtxpro.PRO_HIT_COUNT
# if spiciness > smpl.spice_level.get():
#     smpl.spice_level.set_gradient(spiciness, 0, 5)
# elif midi.is_control_change(midi_status):
#     logger.info(
#         f"received CC #{(cc_num := midi_data[0])}: {(cc_value := midi_data[1])}")
#     dtxpro.update_bank(cc_num, cc_value)
# elif midi.is_program_change(midi_status):
#     prog_num = midi_data[0]
#     kit = dtxpro.kit_index(prog_num)
#     logger.info(f"received program change {prog_num} -> {kit}")
#     keys.select_sample(kit)
#
#
# max_velocity = 127
# intensity = velocity / max_velocity
# gate = 0.1 + 1 - intensity
# smpl.gate.set_gradient(gate, 1, duration=hit_gate * 2)

# gate_period = 2 if dtxpad.hit_count < 5 else 1
# smpl.gate_period.set(gate_period)
# smpl.update_gates()

def hit_dtx_pad(sequence: Sequence, dtxpad: DrumPad, velocity: int):
    smpl = selected_sample if selected_sample else keys.selected_sample
    if smpl is None:
        keys.selected_sample = sample.loaded_samples[0]
        smpl = keys.selected_sample

    if not sequence.is_started:
        sequence.start_internal()

    step_time = sequence.step_duration()
    unmute_time = 2 * step_time

    # todo: move to dtxpro.py
    # logger.info(f"hit DTX pad {dtxpad}")
    match dtxpad.pad:
        case DtxPad.SNARE:
            smpl.drum_trigger(sequence.step)
            smpl.spice_level.set_gradient(1, 0, 3)
        case DtxPad.HAT:
            logger.info(f"hit hat")
            step = min(63, round(random.random() * 64))
            smpl.scatter_queue = step
        case DtxPad.CLOSED_HAT:
            logger.info(f"closed hat")
            smpl.dice()
            smpl.spice_level.set_gradient(1, 0, 5)
        case DtxPad.TOM1:
            velocity_threshold = 60
            if velocity < velocity_threshold:
                unmute_time *= 2
                smpl.start_halftime(duration=unmute_time)
            else:
                unmute_time *= 4
                smpl.start_quartertime(duration=unmute_time)
            smpl.spice_level.set_gradient(.6, 0, 15)
        case DtxPad.TOM2:
            smpl.start_latch_repeat(4, duration=unmute_time * 4)
            unmute_time *= 1.5
        case DtxPad.TOM3:
            min_gate = 0.5
            if smpl.gate.value <= min_gate:
                smpl.gate.set(1)
            else:
                smpl.gate_decrease()
            if dtxpad.roll_detected():
                smpl.looping = not smpl.looping
                smpl.gate.set(1)
        case DtxPad.CRASH1:
            smpl.pitch_mod(1, sequence.step, duration=unmute_time * 1.5)
        case DtxPad.CRASH2:
            smpl.pitch_mod(-1, sequence.step, duration=unmute_time * 1.5)
        case DtxPad.RIDE:
            unmute_time *= 2
            smpl.stop_latch_repeat()
            smpl.stop_stretch()
    smpl.unmute(duration=unmute_time)
    logger.debug(f"hit combo: {total_hit_count()}")
