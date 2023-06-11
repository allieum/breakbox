from codecs import lookup
from collections import defaultdict

import sample
from sequence import sequence
import utility

logger = utility.get_logger(__name__)

K_STOP = 'delete'
K_NEXT_BANK = '#'
# K_RESET = 'tab'
K_SHIFT = 'shift'


# step repeat
K_SR8 = 'x'
K_SR4 = 'c'
K_SR2 = 'v'
K_SR1 = 'b'
SR_KEYS = {
    K_SR8: 8,
    K_SR4: 4,
    K_SR2: 2,
    K_SR1: 1
}

K_GATE_UP = '5'
K_GATE_DOWN = 't'

K_GATE_PERIOD_UP = '4'
K_GATE_PERIOD_DOWN = 'r'

K_GATE_FOLLOW = '3'
K_GATE_INVERT = 'e'

# halftime (0.5 timestretch) / quartertime
K_QT = '2'
K_HT = 'w'
K_HT_UP = 'ctrl'
K_HT_DOWN = 'space'
K_PITCH = 'alt'

dactyl_keys =[
    ['esc',   '1', '2',   '3',   '4', '5'],
    ['`',     'q', 'w',   'e',   'r', 't'],
    ['tab',   'a', 's',   'd',   'f', 'g'],
    ['shift', 'z', 'x',   'c',   'v', 'b'],
                  ['tab', K_NEXT_BANK],
                                 [K_STOP,     'shift'],
                                 ['space',  'ctrl'],
                                 ['enter',    'alt'],
]

# LOOP_KEYS = dactyl_keys[0]
SAMPLE_KEYS = dactyl_keys[2]
# HOLD_KEYS = dactyl_keys[3]

selected_sample = None

key_held = defaultdict(bool)
# key_frozen = defaultdict(bool)

def get_activated_samples():
    return [sample.current_samples()[i] for i, k in enumerate(SAMPLE_KEYS) if key_held[(k)]]

def pitch_press(*_):
    for s in get_activated_samples():
        logger.info(f"activate pitch mod for {s.name}")
        s.pitch_mod(sequence)

def pitch_release(*_):
    for s in sample.current_samples():
        if s.pitch.lfo is not None:
            logger.info(f"deactivate pitch mod for {s.name}")
            s.cancel_pitch_mod()

def sample_press(i, is_repeat):
    global selected_sample
    prev_selected = selected_sample
    selected_sample = sample.current_samples()[i]

    if is_repeat:
        return

    if key_held[K_SHIFT]:
        selected_sample.looping = not selected_sample.looping
        logger.info(f"{selected_sample.name} looping set to {selected_sample.looping}")
    elif selected_sample.looping:
        selected_sample.looping = False
        logger.info(f"{selected_sample.name} looping set to {selected_sample.looping}")

    if key_held[K_GATE_FOLLOW] and prev_selected and prev_selected != selected_sample:
        logger.info(f"set {prev_selected.name} to invert gates of {selected_sample.name}")
        selected_sample.gate_follower = prev_selected
        prev_selected.gate_leader = selected_sample


    if not sequence.is_started:
        sequence.start_internal()

    selected_sample.unmute()

    for step_repeat_key, length in SR_KEYS.items():
        if key_held[(step_repeat_key)]:
            selected_sample.step_repeat_start(sequence.step, length)
    if key_held[(K_HT)]:
        selected_sample.halftime = True

def shift_press(*_):
    global shifted
    for s in [sample.current_samples()[i] for i,k in enumerate(SAMPLE_KEYS) if key_held[k]]:
        s.looping = not s.looping
        logger.info(f"{s.name} set looping to {s.looping}")

def shift_release(*_):
    global shifted
    shifted = False

def sample_release(i):
    s = sample.current_samples()[i]
    if not s.looping:
        s.mute()
        s.step_repeat_stop()
        s.halftime = False

def step_repeat_press(length, *_):
    if selected_sample is None:
        return
    selected_sample.step_repeat_start(sequence.step, length)

def step_repeat_release(length):
    sample.step_repeat_stop(length)

    # follower set logic
def gate_period_up_press(*_):
    if selected_sample is None:
        return
    selected_sample.gate_period_increase()
    logger.info(f"set gate period to {selected_sample.gate_period.value} for {selected_sample.name}")

def gate_period_down_press(*_):
    if selected_sample is None:
        return
    selected_sample.gate_period_decrease()
    logger.info(f"set gate period to {selected_sample.gate_period.value} for {selected_sample.name}")

def gate_up_press(*_):
    if selected_sample is None:
        return
    selected_sample.gate_increase()
    logger.info(f"set gate to {selected_sample.gate.value} for {selected_sample.name}")

def gate_down_press(*_):
    if selected_sample is None:
        return
    selected_sample.gate_decrease()
    logger.info(f"set gate to {selected_sample.gate.value} for {selected_sample.name}")

def gate_invert_press(*_):
    if selected_sample is None:
        return
    selected_sample.gates = selected_sample.invert_gates()
    logger.info(f"inverted gates for {selected_sample.name}")

def gate_follow_press(*_):
    if selected_sample is None:
        return
    if key_held[K_SHIFT]:
        if selected_sample.gate_leader:
            selected_sample.gate_leader.gate_follower = None
            selected_sample.gate_leader = None
        if selected_sample.gate_follower:
            selected_sample.gate_follower.gate_leader = None
            selected_sample.gate_follower = None

def ht_press(is_repeat):
    if selected_sample is None or is_repeat:
        return
    selected_sample.rate *= 0.5

def qt_press(is_repeat):
    if selected_sample is None or is_repeat:
        return
    selected_sample.rate *= 0.25

def ht_release(*_):
    if selected_sample is None:
        return
    selected_sample.rate *= 2

def qt_release(*_):
    if selected_sample is None:
        return
    selected_sample.rate *= 4

def make_handler(handler, x):
    def f(*args):
        handler(x, *args)
    return f

# todo dict of handlers, ie move everything into press and release
press = {
    K_HT: ht_press,
    K_QT: qt_press,
    K_PITCH: pitch_press,
    K_GATE_DOWN: gate_down_press,
    K_GATE_UP: gate_up_press,
    K_GATE_PERIOD_DOWN: gate_period_down_press,
    K_GATE_PERIOD_UP: gate_period_up_press,
    K_GATE_INVERT: gate_invert_press,
    K_GATE_FOLLOW: gate_follow_press,
    K_SHIFT: shift_press,
    **dict(zip(SAMPLE_KEYS, [make_handler(sample_press, i) for i in range(len(SAMPLE_KEYS))])),
    **dict([(sr_key, make_handler(step_repeat_press, length)) for sr_key, length in SR_KEYS.items()])
}

release = {
    K_HT: ht_release,
    K_QT: qt_release,
    K_PITCH: pitch_release,
    K_SHIFT: shift_release,
    **dict(zip(SAMPLE_KEYS, [make_handler(sample_release, i) for i in range(len(SAMPLE_KEYS))])),
    **dict([(sr_key, make_handler(step_repeat_release, length)) for sr_key, length in SR_KEYS.items()])
}

def key_pressed(e):
    logger.debug(f"start press handler for {e.name}")

    if e.name in press:
        press[e.name](key_held[e.name])

    if key_held[(e.name)]:
        logger.debug(f"{e} already active, doing nothing")
        return
    key_held[e.name] = True

    if e.name == K_HT_UP:
        sample.increase_ts_time()

    if e.name == K_HT_DOWN:
        sample.decrease_ts_time()

    if K_STOP == e.name:
        # cancel held keys
        for s in sample.current_samples():
            s.mute()
            s.looping = False
        if sequence.is_internal():
            sequence.stop_internal()

    if K_NEXT_BANK == e.name:
        looping_index = None
        old_samples = sample.current_samples()
        for i, s in enumerate(old_samples):
            if not s.is_muted() and not key_held[(SAMPLE_KEYS[i])]:
                looping_index = i
                # print(f"looping index {i}")
        # cancel held keys
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

    # if e.name == K_RESET:
    #     logger.warn(f"Reset key pressed, restarting program")
    #     utility.restart_program()

    logger.debug(f"finish press handler for {e}")


def key_released(e):
    if not key_held[e.name]:
        return
    key_held[e.name] = False
    process_release(e.name)

def process_release(k):
    logger.debug(f"start release handler for {k}")

    if k in release:
        release[k]()
        return

    if K_HT == k:
        sample.stop_halftime()
    logger.debug(f"finish release handler for {k}")
