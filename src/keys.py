import functools
import time
from collections import defaultdict

import sample
import utility
from effects import decrease_ts_time, increase_ts_time
from modulation import Lfo
from sequence import sequence

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
    K_SR1: 1,
}

K_GATE_UP = '5'
K_GATE_DOWN = 't'

K_GATE_PERIOD_UP = '4'
K_GATE_PERIOD_DOWN = 'r'

K_DICE_UP = 'esc'
K_DICE_DOWN = '`'

# halftime (0.5 timestretch) / quartertime
K_QT = '2'
K_HT = 'w'
# K_TS_UP = 'ctrl'
K_TS_UP = 'ctrl.blah'
K_ONESHOT = 'ctrl'
K_TS_DOWN = 'space'
K_PITCH_UP = '3'
K_PITCH_DOWN = 'e'

K_SPICE_UP = '1'
K_SPICE_DOWN = 'q'

K_FX_CANCEL = 'z'

K_RECORD = 'enter'
K_ERASE = 'alt'

AUTOPLAY_KEYS = {
    K_SPICE_UP, K_SPICE_DOWN, K_QT, K_HT, K_PITCH_DOWN, K_PITCH_UP, *SR_KEYS.keys(),
}

dactyl_keys = [
    ['esc',   '1', '2',   '3',   '4', '5'],
    ['`',     'q', 'w',   'e',   'r', 't'],
    ['tab',   'a', 's',   'd',   'f', 'g'],
    ['shift', 'z', 'x',   'c',   'v', 'b'],
    ['tab', '#'],
    ['delete',     'shift'],
    ['space',  'ctrl'],
    ['enter',    'alt'],
]
RESET_KEYS = ['space', 'ctrl', 'enter', 'alt']


class Effect:
    def __init__(self, cancel):
        self.cancel = cancel


# LOOP_KEYS = dactyl_keys[0]
SAMPLE_KEYS = dactyl_keys[2]
# HOLD_KEYS = dactyl_keys[3]

selected_sample = None
selected_effects = []
key_held = defaultdict(bool)


def select_sample(i):
    global selected_sample
    bank = (i // sample.BANK_SIZE) % sample.NUM_BANKS
    if bank != sample.bank.get():
        sample.set_bank(bank)
    selected_sample = sample.all_samples()[i % len(sample.all_samples())]


def get_activated_samples():
    return [sample.current_samples()[i] for i, k in enumerate(SAMPLE_KEYS) if key_held[(k)]]


def pitch_down_mod(s):
    logger.info(f"{s.name} pitch down activated")
    s.modulate(s.pitch, 1, Lfo.Shape.DEC, 1)


def pitch_up_mod(s):
    logger.info(f"{s.name} pitch up activated")
    s.modulate(s.pitch, 1, Lfo.Shape.INC, 1)


persist_fx_count = 0


def momentary_fx_press(handler, shift_persist=True, autoplay_sample=True):
    def fxpress(is_repeat, *args):
        global persist_fx_count
        if selected_sample is None or is_repeat:
            return
        if autoplay_sample:
            selected_sample.mute_override = True
            selected_sample.unmute()
        if persist := shift_persist and key_held[K_SHIFT]:
            logger.info(f"persisting current effect")
            selected_sample.looping = True
            persist_fx_count += 1
        fx = handler(selected_sample, *args)
        if not persist:
            selected_effects.append(fx)
        else:
            selected_effects.clear()
    return fxpress


def momentary_fx_release(handler=None, shift_persist=True, autoplay_sample=True):
    def fxrelease(*args):
        global persist_fx_count
        if selected_sample is None:
            return
        if shift_persist and persist_fx_count > 0:
            selected_effects.clear()
            logger.info(f"skipping release so effect is persisted")
            persist_fx_count -= 1
            return
        sample_activated = is_pushed(selected_sample) or any(
            key_held[k] for k in AUTOPLAY_KEYS)
        if not selected_sample.looping and not sample_activated and autoplay_sample:
            selected_sample.mute()
            selected_sample.mute_override = False
        if handler:
            handler(selected_sample, *args)
    return fxrelease


def spice_up_press(selected):
    selected.spice_level.set(delta=0.1)
    logger.info(f"{selected.name} set spice to {selected.spice_level.value}")


def spice_down_press(selected):
    selected.spice_level.set(delta=-0.1)
    logger.info(f"{selected.name} set spice to {selected.spice_level.value}")


def pitch_up_press(selected):
    pitch_up_mod(selected)
    step_repeat_press(1, selected)
    return (Effect(functools.partial(pitch_up_release, selected)))


def pitch_down_press(selected):
    pitch_down_mod(selected)
    step_repeat_press(1, selected)
    return (Effect(functools.partial(pitch_down_release, selected)))


def pitch_up_release(selected):
    # todo cancel & gretel
    if key_held[K_PITCH_DOWN]:
        pitch_down_mod(selected)
    else:
        if selected.pitch.lfo:
            selected.pitch.lfo.enabled = False
        if not key_held[K_SHIFT]:
            selected.pitch.restore_default()
        step_repeat_release(1, selected)


def pitch_down_release(selected):
    if key_held[K_PITCH_UP]:
        pitch_up_mod(selected)
    else:
        if selected.pitch.lfo:
            selected.pitch.lfo.enabled = False
        if not key_held[K_SHIFT]:
            selected.pitch.restore_default()
        step_repeat_release(1, selected)


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
        return None

    prev_selected = selected_sample
    if prev_selected and sample.current_samples()[i] != prev_selected:
        logger.info(
            f"{prev_selected.name} clearing active effects, switch to {sample.current_samples()[i].name}")
        logger.info(
            f"{prev_selected.name} looping = {prev_selected.looping}, {selected_effects}")
        for effect in selected_effects:
            if effect is not None:
                effect.cancel()
        selected_effects.clear()
        if not prev_selected.looping and not is_pushed(prev_selected):
            prev_selected.mute()
            prev_selected.mute_override = False
    selected_sample = sample.current_samples()[i]

    if key_held[K_SHIFT]:
        selected_sample.looping = True
        logger.info(
            f"{selected_sample.name} looping set to {selected_sample.looping}")
    elif selected_sample.looping:
        selected_sample.looping = False
        logger.info(
            f"{selected_sample.name} looping set to {selected_sample.looping}")

    if not sequence.is_started:
        sequence.start_internal()

    selected_sample.mute_override = True
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
    return None


def sample_release(i):
    s = sample.current_samples()[i]
    if is_current := s == selected_sample:
        selected_effects.clear()
    if not s.looping and not (is_current and any(key_held[k] for k in AUTOPLAY_KEYS)):
        s.mute()
        s.mute_override = False


def shift_press(repeat):
    if repeat:
        return
    global persist_fx_count
    fx_keys_pressed = sum([1 for k in AUTOPLAY_KEYS if key_held[k]])
    persist_fx_count += fx_keys_pressed
    if persist_fx_count > 0 and selected_sample is not None:
        selected_sample.looping = True
    for s in [sample.current_samples()[i] for i, k in enumerate(SAMPLE_KEYS) if key_held[k]]:
        s.looping = True
        logger.info(f"{s.name} set looping to {s.looping}")
    if key_held[K_FX_CANCEL]:
        fx_cancel_press(False, shift=True)


def step_repeat_press(length, selected):
    selected.step_repeat_start(sequence.step, length)
    return (Effect(functools.partial(selected.step_repeat_stop, length)))


def step_repeat_release(length, selected):
    logger.info(f"releasing {length}")
    selected.step_repeat_stop(length)


def gate_period_up_press(*_):
    if selected_sample is None:
        return
    selected_sample.gate_period_increase()
    logger.info(
        f"set gate period to {selected_sample.gate_period.value} for {selected_sample.name}")


def gate_period_down_press(*_):
    if selected_sample is None:
        return
    selected_sample.gate_period_decrease()
    logger.info(
        f"set gate period to {selected_sample.gate_period.value} for {selected_sample.name}")


def gate_up_press(*_):
    if selected_sample is None:
        return
    selected_sample.gate_increase()
    logger.info(
        f"set gate to {selected_sample.gate.value} for {selected_sample.name}")


def gate_down_press(*_):
    if selected_sample is None:
        return
    selected_sample.gate_decrease()
    logger.info(
        f"set gate to {selected_sample.gate.value} for {selected_sample.name}")


def gate_invert_press(*_):
    if selected_sample is None:
        return
    selected_sample.gates = selected_sample.invert_gates()
    logger.info(f"inverted gates for {selected_sample.name}")


def gate_follow_press(*_):
    if selected_sample is None:
        return
    if key_held[K_SHIFT] and (mirror := selected_sample.gate_mirror):
        selected_sample.gate_mirror = None
        mirror.gate_mirror = None
        selected_sample.default_gates()
        mirror.default_gates()


def ht_press(selected):
    selected.halftime = True
    return (Effect(selected.stop_halftime))


def qt_press(selected):
    selected.quartertime = True
    return (Effect(selected.stop_quartertime))


def ht_release(selected):
    selected.halftime = False


def qt_release(selected):
    selected.quartertime = False


def dice_press(repeat):
    if selected_sample is None or repeat:
        return
    selected_sample.dice()


def fx_cancel_press(repeat, shift=None):
    if shift is None:
        shift = key_held[K_SHIFT]
    if repeat:
        return
    if shift:
        for s in sample.current_samples():
            s.cancel_fx()
    elif selected_sample is not None:
        selected_sample.cancel_fx()
    selected_effects.clear()


def record_press(selected):
    selected.start_recording()
    return Effect(selected.stop_recording)


def record_release(selected):
    selected.stop_recording()


def oneshot_press(selected):
    selected.clear_sound_queue()
    sequence.last_queued_step -= 1
    selected.trigger_oneshot(
        sequence.step, time.time() - sequence.step_time(sequence.step))
    return Effect(selected.stop_oneshot)


def oneshot_release(selected):
    selected.stop_oneshot()


def erase_press(_):
    if selected_sample is None:
        return
    selected_sample.recorded_steps[s := (
        sequence.step + 2) % len(selected_sample.recorded_steps)] = None
    logger.info(f"{selected_sample.name} erasing step {s}")


def make_handler(handler, x):
    def f(*args):
        handler(x, *args)
    return f


press = {
    K_TS_UP: increase_ts_time,
    K_TS_DOWN: decrease_ts_time,
    K_SPICE_UP: momentary_fx_press(spice_up_press, shift_persist=False),
    K_SPICE_DOWN: momentary_fx_press(spice_down_press, shift_persist=False),
    K_HT: momentary_fx_press(ht_press),
    K_QT: momentary_fx_press(qt_press),
    K_PITCH_UP: momentary_fx_press(pitch_up_press, shift_persist=False),
    K_PITCH_DOWN: momentary_fx_press(pitch_down_press, shift_persist=False),
    K_GATE_DOWN: gate_down_press,
    K_GATE_UP: gate_up_press,
    K_GATE_PERIOD_DOWN: gate_period_down_press,
    K_GATE_PERIOD_UP: gate_period_up_press,
    K_DICE_DOWN: dice_press,
    K_DICE_UP: dice_press,
    # K_GATE_INVERT: gate_invert_press,
    # K_GATE_FOLLOW: gate_follow_press,
    K_SHIFT: shift_press,
    K_RECORD: momentary_fx_press(record_press, autoplay_sample=False),
    K_ONESHOT: momentary_fx_press(oneshot_press),
    K_ERASE: erase_press,
    K_FX_CANCEL: fx_cancel_press,
    **dict(zip(SAMPLE_KEYS, [make_handler(sample_press, i) for i in range(len(SAMPLE_KEYS))], strict=True)),
    **{sr_key: momentary_fx_press(make_handler(step_repeat_press, length)) for sr_key, length in SR_KEYS.items()},
}

release = {
    K_HT: momentary_fx_release(ht_release),
    K_QT: momentary_fx_release(qt_release),
    K_SPICE_UP: momentary_fx_release(),
    K_SPICE_DOWN: momentary_fx_release(),
    K_PITCH_UP: momentary_fx_release(pitch_up_release, shift_persist=False),
    K_PITCH_DOWN: momentary_fx_release(pitch_down_release, shift_persist=False),
    K_RECORD: momentary_fx_release(record_release, autoplay_sample=False),
    K_ONESHOT: momentary_fx_release(oneshot_release),
    **dict(zip(SAMPLE_KEYS, [make_handler(sample_release, i) for i in range(len(SAMPLE_KEYS))], strict=True)),
    **{sr_key: momentary_fx_release(make_handler(step_repeat_release, length)) for sr_key, length in SR_KEYS.items()},
}


def key_pressed(e):
    logger.debug(f"start press handler for {e.name}")

    if all(key_held[k] for k in RESET_KEYS):
        logger.warning("restarting program!!!")
        utility.restart_program()

    if e.name in press:
        press[e.name](key_held[e.name])

    if key_held[(e.name)]:
        logger.debug(f"{e} already active, doing nothing")
        return
    key_held[e.name] = True

    if e.name == K_STOP:
        # cancel held keys
        for s in sample.current_samples():
            s.mute()
            s.clear_sound_queue()
            if s.channel and (sound := s.channel.get_sound()):
                sound.stop()
            s.looping = False
        if sequence.is_internal():
            sequence.stop_internal()

    if e.name == K_NEXT_BANK:
        looping_index = None
        old_samples = sample.current_samples()
        for i, s in enumerate(old_samples):
            if not s.is_muted() and not key_held[(SAMPLE_KEYS[i])]:
                looping_index = i
                # print(f"looping index {i}")
        # cancel held keys
        delta = -1 if key_held[K_SHIFT] else 1
        sample.bank.set((sample.bank.get() + delta) % sample.NUM_BANKS)
        for new_sample, old_sample in zip(sample.current_samples(), old_samples, strict=True):
            new_sample.swap_channel(old_sample)
        if looping_index is not None:
            sample.current_samples()[looping_index].looping = True

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

    logger.debug(f"finish release handler for {k}")
