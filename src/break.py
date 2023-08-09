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
        s, keys.selected_sample, sequence.step, i, dtxpro.selected_sample) for i, s in enumerate(sample.current_samples())]
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
                keys.selected_sample = sample.current_samples()[0]
                smpl = keys.selected_sample

            if not sequence.is_started:
                sequence.start_internal()

            last_step_time = sequence.step_time(
                sequence.step) - sequence.step_duration()
            offset = time.time() - last_step_time
            if offset > sequence.step_duration():
                offset -= sequence.step_duration()

            hit_gate = 2 * sequence.step_duration() - offset

            # todo: move to dtxpro.py
            logger.info(f"hit DTX pad {dtxpad}")
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
                        hit_gate *= 2
                        smpl.start_halftime(duration=hit_gate)
                    else:
                        hit_gate *= 4
                        smpl.start_quartertime(duration=hit_gate)
                case DtxPad.TOM2:
                    # smpl.step_repeat_start(
                    #     sequence.step, 4, duration=hit_gate * 2)
                    hit_gate *= 2
                    smpl.start_latch_repeat(4, duration=hit_gate)
                case DtxPad.TOM3:
                    if dtxpad.roll_detected():
                        smpl.looping = not smpl.looping
                case DtxPad.CRASH1:
                    smpl.pitch_mod(1, sequence.step, duration=hit_gate * 1.5)
                case DtxPad.CRASH2:
                    smpl.pitch_mod(-1, sequence.step, duration=hit_gate * 1.5)
                case DtxPad.RIDE:
                    hit_gate *= 2
                    pass
                    # max_velocity = 127
                    # intensity = velocity / max_velocity
                    # gate = 0.1 + 1 - intensity
                    # smpl.gate.set_gradient(gate, 1, duration=hit_gate * 2)

                    # gate_period = 2 if dtxpad.hit_count < 5 else 1
                    # smpl.gate_period.set(gate_period)
                    # smpl.update_gates()
                    # hit_gate *= 2

            smpl.unmute(duration=hit_gate, step=(sequence.step - 1) %
                        sequence.steps, offset=offset)
            logger.info(f"hit combo: {dtxpro.total_hit_count()}")
            # spiciness = dtxpro.total_hit_count() / dtxpro.PRO_HIT_COUNT
            # if spiciness > smpl.spice_level.get():
            #     smpl.spice_level.set_gradient(spiciness, 0, 5)
    elif midi.is_control_change(midi_status):
        logger.info(
            f"received CC #{(cc_num := midi_data[0])}: {(cc_value := midi_data[1])}")
        dtxpro.update_bank(cc_num, cc_value)
    elif midi.is_program_change(midi_status):
        prog_num = midi_data[0]
        kit = dtxpro.kit_index(prog_num)
        logger.info(f"received program change {prog_num} -> {kit}")
        keys.select_sample(kit)

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
