from collections import defaultdict

import sample
from sequence import sequence
from modulation import Lfo
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

K_GATE_FOLLOW = 'esc'
K_GATE_INVERT = '`'

# halftime (0.5 timestretch) / quartertime
K_QT = '2'
K_HT = 'w'
K_TS_UP = 'ctrl'
K_TS_DOWN = 'space'
K_PITCH_UP = '3'
K_PITCH_DOWN = 'e'

FX_KEYS = set((K_QT, K_HT, K_PITCH_DOWN, K_PITCH_UP, *SR_KEYS.keys()))

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

class Effect:
    def __init__(self, cancel):
        self.cancel = cancel

# LOOP_KEYS = dactyl_keys[0]
SAMPLE_KEYS = dactyl_keys[2]
# HOLD_KEYS = dactyl_keys[3]

selected_sample = None
selected_effects = []
key_held = defaultdict(bool)

def get_activated_samples():
    return [sample.current_samples()[i] for i, k in enumerate(SAMPLE_KEYS) if key_held[(k)]]

def pitch_down_mod(s):
    logger.info(f"{s.name} pitch down activated")
    s.modulate(s.pitch, 1, Lfo.Shape.DEC, 1)

def pitch_up_mod(s):
    logger.info(f"{s.name} pitch up activated")
    s.modulate(s.pitch, 1, Lfo.Shape.INC, 1)

persist_fx_count = 0
def momentary_fx_press(handler, shift_persist=True):
    def fxpress(is_repeat, *args):
        global persist_fx_count
        if selected_sample is None or is_repeat:
            return
        selected_sample.unmute()
        if persist := shift_persist and key_held[K_SHIFT]:
            logger.info(f"persisting current effect")
            selected_sample.looping = True
            persist_fx_count += 1
        fx = handler(*args)
        if not persist:
            selected_effects.append(fx)
        else:
            selected_effects.clear()
    return fxpress

def momentary_fx_release(handler, shift_persist=True):
    def fxrelease(*args):
        global persist_fx_count
        if selected_sample is None:
            return
        if shift_persist and persist_fx_count > 0:
            selected_effects.clear()
            logger.info(f"skipping release so effect is persisted")
            persist_fx_count -= 1
            return
        if not selected_sample.looping and not is_pushed(selected_sample):
            selected_sample.mute()
        handler(*args)
    return fxrelease

def pitch_up_press(*_):
    pitch_up_mod(selected_sample)
    step_repeat_press(1)
    return (Effect(pitch_up_release))

def pitch_down_press(*_):
    pitch_down_mod(selected_sample)
    step_repeat_press(1)
    return (Effect(pitch_down_release))

def pitch_up_release(*_):
    if key_held[K_PITCH_DOWN]:
        pitch_down_mod(selected_sample)
    else:
        if selected_sample.pitch.lfo:
            selected_sample.pitch.lfo.enabled = False
        step_repeat_release(1)

def pitch_down_release(*_):
    if key_held[K_PITCH_UP]:
        pitch_up_mod(selected_sample)
    else:
        if selected_sample.pitch.lfo:
            selected_sample.pitch.lfo.enabled = False
        step_repeat_release(1)

def is_pushed(s: sample.Sample):
    try:
        i = sample.current_samples().index(s)
        return key_held[SAMPLE_KEYS[i]]
    except ValueError:
        return False

def sample_press(i, is_repeat):
    global selected_sample
    logger.debug(f"pressed sample key {i} {is_repeat} {selected_sample}")

    if is_repeat:
        return

    prev_selected = selected_sample
    if prev_selected and sample.current_samples()[i] != prev_selected:
        logger.info(f"{prev_selected.name} clearing active effects, switch to {sample.current_samples()[i].name}")
        logger.info(f"{prev_selected.name} looping = {prev_selected.looping}, {selected_effects}")
        for effect in selected_effects:
            if effect is not None:
                effect.cancel()
        selected_effects.clear()
        if not prev_selected.looping and not is_pushed(prev_selected):
            prev_selected.mute()
    selected_sample = sample.current_samples()[i]

    if key_held[K_SHIFT]:
        selected_sample.looping = True
        logger.info(f"{selected_sample.name} looping set to {selected_sample.looping}")
    elif selected_sample.looping:
        selected_sample.looping = False
        logger.info(f"{selected_sample.name} looping set to {selected_sample.looping}")

    if key_held[K_GATE_FOLLOW] and prev_selected and prev_selected != selected_sample:
        logger.info(f"set {prev_selected.name} to invert gates of {selected_sample.name}")
        selected_sample.gate_mirror = prev_selected
        prev_selected.gate_mirror = selected_sample

    if not sequence.is_started:
        sequence.start_internal()

    selected_sample.unmute()

    for step_repeat_key, length in SR_KEYS.items():
        if key_held[(step_repeat_key)]:
            selected_sample.step_repeat_start(sequence.step, length)
            # todo gotta append these direct like
            return (Effect(selected_sample.step_repeat_stop))

    if key_held[(K_HT)]:
        selected_sample.halftime = True
        return (Effect(selected_sample.stop_halftime))

    if key_held[(K_QT)]:
        selected_sample.quartertime = True
        return (Effect(selected_sample.stop_quartertime))

def sample_release(i):
    s = sample.current_samples()[i]
    if is_current := s == selected_sample:
        selected_effects.clear()
    if not s.looping and not (is_current and any([key_held[k] for k in FX_KEYS])):
        s.mute()

def shift_press(*_):
    global persist_fx_count
    fx_keys_pressed = sum([1 for k in FX_KEYS if key_held[k]])
    persist_fx_count += fx_keys_pressed
    if persist_fx_count > 0 and selected_sample is not None:
        selected_sample.looping = True
    for s in [sample.current_samples()[i] for i,k in enumerate(SAMPLE_KEYS) if key_held[k]]:
        s.looping = not s.looping
        logger.info(f"{s.name} set looping to {s.looping}")

def step_repeat_press(length, *_):
    selected_sample.step_repeat_start(sequence.step, length)
    return (Effect(selected_sample.step_repeat_stop))

def step_repeat_release(length):
    sample.step_repeat_stop(length)

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
        if mirror := selected_sample.gate_mirror:
            selected_sample.gate_mirror = None
            mirror.gate_mirror = None
            selected_sample.default_gates()
            mirror.default_gates()

def ht_press():
    selected_sample.halftime = True
    return (Effect(selected_sample.stop_halftime))

def qt_press():
    selected_sample.quartertime = True
    return (Effect(selected_sample.stop_quartertime))

def ht_release():
    selected_sample.halftime = False

def qt_release():
    selected_sample.quartertime = False

def make_handler(handler, x):
    def f(*args):
        handler(x, *args)
    return f

press = {
    K_TS_UP: sample.increase_ts_time,
    K_TS_DOWN: sample.decrease_ts_time,
    K_HT: momentary_fx_press(ht_press),
    K_QT: momentary_fx_press(qt_press),
    K_PITCH_UP: momentary_fx_press(pitch_up_press, shift_persist=False),
    K_PITCH_DOWN: momentary_fx_press(pitch_down_press, shift_persist=False),
    K_GATE_DOWN: gate_down_press,
    K_GATE_UP: gate_up_press,
    K_GATE_PERIOD_DOWN: gate_period_down_press,
    K_GATE_PERIOD_UP: gate_period_up_press,
    K_GATE_INVERT: gate_invert_press,
    K_GATE_FOLLOW: gate_follow_press,
    K_SHIFT: shift_press,
    **dict(zip(SAMPLE_KEYS, [make_handler(sample_press, i) for i in range(len(SAMPLE_KEYS))])),
    **{sr_key: momentary_fx_press(make_handler(step_repeat_press, length)) for sr_key, length in SR_KEYS.items()}
}

release = {
    K_HT: momentary_fx_release(ht_release),
    K_QT: momentary_fx_release(qt_release),
    K_PITCH_UP: momentary_fx_release(pitch_up_release, shift_persist=False),
    K_PITCH_DOWN: momentary_fx_release(pitch_down_release, shift_persist=False),
    **dict(zip(SAMPLE_KEYS, [make_handler(sample_release, i) for i in range(len(SAMPLE_KEYS))])),
    **{sr_key: momentary_fx_release(make_handler(step_repeat_release, length)) for sr_key, length in SR_KEYS.items()}
}

def key_pressed(e):
    logger.debug(f"start press handler for {e.name}")

    if e.name in press:
        press[e.name](key_held[e.name])

    if key_held[(e.name)]:
        logger.debug(f"{e} already active, doing nothing")
        return
    key_held[e.name] = True


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
