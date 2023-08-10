import random
import time
from datetime import datetime
from multiprocessing import Process

import display
import dtxpro
import keyboard
import keys
import lights
import sample
import utility
from blink import update_dmx
from dtxpro import DtxPad
from sequence import sequence

import midi

logger = utility.get_logger(__name__)
current_time = datetime.now().strftime("%H:%M:%S")
logger.debug(f"Start time = {current_time}")


def on_key(e):
    if e.event_type == keyboard.KEY_DOWN:
        keys.key_pressed(e)
    elif e.event_type == keyboard.KEY_UP:
        keys.key_released(e)


keyboard.hook(on_key)

# control.init()
sample.load_samples()
display.init(sample.all_samples())
midi.connect()

sequence.on_step(lambda s: sample.Sample.audio_executor.submit(update_dmx, s))

subs = [
    Process(target=lights.run, args=(lights.q,)),
    Process(target=display.run, args=(display.q,)),
]
for sub in subs:
    sub.start()

# blink.Light.set_brightness(50)
last_dmx = time.time()
last_dmx_step = None


def update():
    # control.update()
    midi_status, midi_data = midi.get_status()
    sequence.update(midi_status)
    sample.play_samples(sequence.step_duration())
    sample_states = [sample.SampleState.of(
        s, keys.selected_sample, sequence.step, i, dtxpro.selected_sample) for i, s in enumerate(sample.loaded_samples)]
    dtxpro.update()

    # if lights.refresh_ready(samples_on):
    # lights.refreshing = True
    # lights.update(samples_on)

    if midi.is_note_on(midi_status):
        # logger.info(f"{data} {status}")
        note_number = midi_data[0]
        velocity = midi_data[1]
        if (dtxpad := dtxpro.struck_pad(note_number)) is not None and velocity != 0:
            smpl = keys.selected_sample if not dtxpro.selected_sample else dtxpro.selected_sample
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
                case DtxPad.HAT:
                    logger.info(f"hit hat")
                    step = min(63, round(random.random() * 64))
                    smpl.scatter_queue = step
                case DtxPad.CLOSED_HAT:
                    logger.info(f"closed hat")
                    smpl.dice()
                    smpl.spice_level.set_gradient(1, 0, 5)
                case DtxPad.TOM1:
                    velocity_threshold = 40
                    if velocity < velocity_threshold:
                        unmute_time *= 2
                        smpl.start_halftime(duration=unmute_time)
                    else:
                        unmute_time *= 4
                        smpl.start_quartertime(duration=unmute_time)
                case DtxPad.TOM2:
                    smpl.start_latch_repeat(4, duration=unmute_time * 4)
                    unmute_time *= 1.5
                case DtxPad.TOM3:
                    if dtxpad.roll_detected():
                        smpl.looping = not smpl.looping
                case DtxPad.CRASH1:
                    smpl.pitch_mod(1, sequence.step, duration=unmute_time * 1.5)
                case DtxPad.CRASH2:
                    smpl.pitch_mod(-1, sequence.step, duration=unmute_time * 1.5)
                case DtxPad.RIDE:
                    unmute_time *= 2
                    smpl.stop_latch_repeat()
                    smpl.stop_stretch()
            smpl.unmute(duration=unmute_time)
            logger.debug(f"hit combo: {dtxpro.total_hit_count()}")

    try:
        lights.q.put(sample_states, block=False)
        display.q.put(sample_states, block=False)
    except:
        pass

    if midi.lost_connection():
        if sequence.midi_started:
            print("lost midi connection")
            sequence.stop_midi()
        midi.reconnect(suppress_output=True)

    time.sleep(0.0005)


while True:
    try:
        update()
    except Exception:
        for sub in subs:
            sub.terminate()
        raise
