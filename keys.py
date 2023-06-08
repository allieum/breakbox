from collections import defaultdict

import sample
from sequence import sequence
import utility

logger = utility.get_logger(__name__)

K_STOP = 'delete'
K_NEXT_BANK = '#'
K_RESET = 'tab'

# halftime (0.5 timestretch)
K_HT = 'enter'
K_HT_UP = 'ctrl'
K_HT_DOWN = 'space'
K_PITCH = 'alt'

# step repeat
K_SR4 = 's'
K_SR2 = 'd'
K_SR1 = 'f'
SR_KEYS = {
    K_SR4: 4,
    K_SR2: 2,
    K_SR1: 1
}

dactyl_keys =[
    ['esc',   '1', '2',   '3',   '4', '5'],
    ['`',     'q', 'w',   'e',   'r', 't'],
    ['tab',   'a', K_SR4, K_SR2, K_SR1, 'g'],
    ['shift', 'z', 'x',   'c',   'v', 'b'],
                  ['tab', K_NEXT_BANK],
                                 [K_STOP,     'shift'],
                                 [K_HT_DOWN,  K_HT_UP],
                                 [K_HT,    K_PITCH],
]

LOOP_KEYS = dactyl_keys[0]
TOGGLE_KEYS = dactyl_keys[1]
HOLD_KEYS = dactyl_keys[3]

key_held = defaultdict(bool)
key_frozen = defaultdict(bool)
def key_active(key):
    return key_held[key] or key_frozen[key]


def get_activated_samples():
    return [sample.current_samples()[i] for i, k in enumerate(TOGGLE_KEYS) if key_active(k)]

def pitch_press():
    for s in get_activated_samples():
        logger.info(f"activate pitch mod for {s.name}")
        s.pitch_mod(sequence)

def pitch_release():
    for s in sample.current_samples():
        if s.pitch.lfo is not None:
            logger.info(f"deactivate pitch mod for {s.name}")
            s.cancel_pitch_mod()

# todo dict of handlers, ie move everything into press and release
press = {
    K_PITCH: pitch_press
}

release = {
    K_PITCH: pitch_release
}

def key_pressed(e):
    logger.debug(f"start press handler for {e}")

    if key_active(e.name):
        logger.debug(f"{e} already active, doing nothing")
        return

    if e.name in press:
        press[e.name]()
        key_held[e.name] = True
        return

    for i, key in enumerate(LOOP_KEYS):
        if key == e.name:
            for j, s in enumerate(sample.current_samples()):
                s.looping = i == j
                # print(f"sample {j} queued: {sample.queued}")

    if e.name == K_HT:
        key_held[K_HT] = True
        for i, key in enumerate(TOGGLE_KEYS):
            if key_active(key):
                sample.current_samples()[i].halftime = True
        if any([key_held[k] for k in HOLD_KEYS]):
            # print(f"freezing {key}")
            key_frozen[K_HT] = True
        elif key_frozen[K_HT]:
            # print(f"unfreezing {key}")
            key_frozen[K_HT] = False
            process_release(K_HT)

    if e.name == K_HT_UP:
        sample.increase_ts_time()

    if e.name == K_HT_DOWN:
        sample.decrease_ts_time()

    for i, key in enumerate(TOGGLE_KEYS):
        if key != e.name:
            continue
        if not key_held[key] and not key_frozen[key]:
            sample.current_samples()[i].toggle_mute()
        if not key_frozen[key]:
            key_held[key] = True
        if any([key_held[k] for k in HOLD_KEYS]):
            # print(f"freezing {key}")
            key_frozen[key] = True
        elif key_frozen[key]:
            # print(f"unfreezing {key}")
            key_frozen[key] = False
            process_release(key)
        # print(f"holding {key}")
        for step_repeat_key, length in SR_KEYS.items():
            if key_active(step_repeat_key):
                sample.current_samples()[i].step_repeat_start(sequence.step, length)
        if key_active(K_HT):
            sample.current_samples()[i].halftime = True
    #
    # if loop or toggle and sequence not started, start it
    if not sequence.is_started:
        for key in LOOP_KEYS + TOGGLE_KEYS:
            if key == e.name:
                sequence.start_internal()

    for step_repeat_key, length in SR_KEYS.items():
        if step_repeat_key != e.name:
            continue
        # print(f"holding {step_repeat_key}")
        if not key_frozen[step_repeat_key]:
            key_held[step_repeat_key] = True
        if any([key_held[k] for k in HOLD_KEYS]): # make hold_active fn
            key_frozen[step_repeat_key] = True
            # print(f"freezing {step_repeat_key}")
        elif key_frozen[step_repeat_key]:
            # print(f"unfreezing {step_repeat_key}")
            key_frozen[step_repeat_key] = False
            process_release(step_repeat_key)
        for i, key in enumerate(TOGGLE_KEYS):
            if key_active(key):
                sample.current_samples()[i].step_repeat_start(sequence.step, length)

    if K_STOP == e.name:
        # cancel held keys
        for key in key_frozen:
            key_held[key] = False
            key_frozen[key] = False
        for s in sample.current_samples():
            s.mute()
            s.looping = False
        if sequence.is_internal():
            sequence.stop_internal()

    if K_NEXT_BANK == e.name:
        looping_index = None
        old_samples = sample.current_samples()
        for i, s in enumerate(old_samples):
            if not s.is_muted() and not key_active(TOGGLE_KEYS[i]):
                looping_index = i
                # print(f"looping index {i}")
        # cancel held keys
        for key in key_frozen:
            key_held[key] = False
            key_frozen[key] = False
        sample.bank = (sample.bank + 1) % sample.NUM_BANKS
        for new_sample, old_sample in zip(sample.current_samples(), old_samples):
            new_sample.swap_channel(old_sample)
        if looping_index is not None:
            sample.current_samples()[looping_index].looping = True

    # cases for hold button:
    # active means at least one key frozen
    #
    #   1) inactive, no other keys held -> do nothing
    #   2) inactive, keys held -> freeze those keys
    #   3) active, no other keys held -> unfreeze all
    #   4) active, nonfrozen keys held -> freeze them
    #    **active vs inactive irrelevant**
    #   5) frozen key pressed -> unfreeze
    #
    # freeze key by press down when hold pressed, or press hold when key pressed
    for key in HOLD_KEYS:
        if key != e.name:
            continue
        key_held[key] = True
        held_keys = [k for k,held in key_held.items() if held and not k in HOLD_KEYS]
        for k in held_keys:
            print(f"freezing {k}")
            key_frozen[k] = True
        if len(held_keys) == 0:
            frozen_keys = [k for k,frozen in key_frozen.items() if frozen]
            for k in frozen_keys:
                print(f"unfreezing {k}")
                key_frozen[k] = False
                process_release(k)

    if e.name == K_RESET:
        logger.warn(f"Reset key pressed, restarting program")
        utility.restart_program()

    logger.debug(f"finish press handler for {e}")


def key_released(e):
    if not key_held[e.name]:
        return
    key_held[e.name] = False
    if key_frozen[e.name]:
        return
    process_release(e.name)

def process_release(k):
    logger.debug(f"start release handler for {k}")

    if k in release:
        release[k]()
        return

    for i, key in enumerate(TOGGLE_KEYS):
        if key == k:
            sample.current_samples()[i].step_repeat_stop()
            sample.current_samples()[i].halftime = False
            sample.current_samples()[i].toggle_mute()
    for key, length in SR_KEYS.items():
        if key != k:
            continue
        sample.step_repeat_stop(length)
    if K_HT == k:
        sample.stop_halftime()
    logger.debug(f"finish release handler for {k}")
