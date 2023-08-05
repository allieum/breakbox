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

# import rtmidi
# from ctypes import *
# from contextlib import contextmanager
# import display
# import lights
# from dmx import DMXInterface
from blink import update_dmx
from dtxpro import DtxPad
from sequence import sequence

import midi

# with DMXInterface() as interface:
#


logger = utility.get_logger(__name__)
current_time = datetime.now().strftime("%H:%M:%S")
logger.debug(f"Start time = {current_time}")


def on_key(e):
    # logger.info(f"{e.name} {e.event_type}")
    if e.event_type == keyboard.KEY_DOWN:
        keys.key_pressed(e)
    elif e.event_type == keyboard.KEY_UP:
        keys.key_released(e)


keyboard.hook(on_key)

# control.init()
sample.init()
display.init(sample.all_samples())
midi.connect()
# midi.load_midi_files()
# sequence.control_bpm(control.encoder)

# if (now := time.time()) - last_dmx > 0.050: # and last_dmx_step != sequence.step:
# TODO move into own file
# sample.Sample.audio_executor.submit(dmx_interface.send_update)
# logger.info(f"dmx frame send took {time.time() - now}s")
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
    status, data = midi.get_status()
    sequence.update(status)
    sample.play_samples(sequence.step_duration())
    sample_states = [lights.SampleState.of(
        s, keys.selected_sample, sequence.step, i) for i, s in enumerate(sample.current_samples())]
    dtxpro.update()

    # if lights.refresh_ready(samples_on):
    # lights.refreshing = True
    # lights.update(samples_on)

    if midi.is_note_on(status):
        # logger.info(f"{data} {status}")
        note_number = data[0]
        velocity = data[1]
        if (dtxpad := dtxpro.struck_pad(note_number)) is not None and velocity != 0:
            if keys.selected_sample is None:
                keys.selected_sample = sample.current_samples()[0]
            smpl = keys.selected_sample

            if not sequence.is_started:
                sequence.start_internal()

            last_step_time = sequence.step_time(
                sequence.step) - sequence.step_duration()
            offset = time.time() - last_step_time
            if offset > sequence.step_duration():
                offset -= sequence.step_duration()

            # TODO quantize based on finishing a round eighth
            hit_gate = 3 * sequence.step_duration() - offset

            logger.info(f"hit DTX pad {dtxpad}")
            match dtxpad.pad:
                case DtxPad.SNARE:
                    smpl.drum_trigger(sequence.step)
                case DtxPad.HAT:
                    logger.info(f"hit hat")
                    pass
                case DtxPad.TOM1:
                    smpl.start_halftime(duration=hit_gate)
                case DtxPad.TOM2:
                    # smpl.drum_trigger(sequence.step, pitched=False)
                    # TODO why not work
                    smpl.step_repeat_start(
                        sequence.step, 4, duration=hit_gate * 2)
                case DtxPad.TOM3:
                    if dtxpad.roll_detected():
                        smpl.looping = not smpl.looping
                case DtxPad.CRASH1:
                    smpl.pitch_mod(1, sequence.step, duration=hit_gate)
                case DtxPad.CRASH2:
                    smpl.pitch_mod(-1, sequence.step, duration=hit_gate)
                case DtxPad.RIDE:
                    # add timer to gate param, velocity/position use
                    pass
            smpl.unmute(duration=hit_gate, step=(sequence.step - 1) %
                        sequence.steps, offset=offset)
            logger.info(f"hit combo: {dtxpro.total_hit_count()}")
            # smpl.drum_trigger(sequence.step, velocity / 127)
        # sample.Sample.audio_executor.submit(update_dmx, 0, note_number)
    elif midi.is_control_change(status):
        logger.info(
            f"received CC #{(cc_num := data[0])}: {(cc_value := data[1])}")
        dtxpro.update_bank(cc_num, cc_value)
    elif midi.is_program_change(status):
        prog_num = data[0]
        kit = dtxpro.kit_index(prog_num)
        logger.info(f"received program change {prog_num} -> {kit}")
        keys.select_sample(kit)

    try:
        lights.q.put(sample_states, block=False)
        display.q.put(sample_states, block=False)
    except:
        pass

    # if time.time() - last_dmx > 0.100:
    #     sample.Sample.audio_executor.submit(update_dmx, 0, 69)
    #     last_dmx = time.time()

    # logger.info(f"putting {samples_on} in queue")
    # f = sample.Sample.audio_executor.submit(lights.update, samples_on)
    # f.add_done_callback(lambda _: lights.refresh_done())

    # display.update(sequence.bpm.get(), sample_states)

    if midi.lost_connection():
        if sequence.midi_started:
            print("lost midi connection")
            sequence.stop_midi()
        midi.reconnect(suppress_output=True)

    # sys.stdout.flush()
    # sys.stderr.flush()
    # import keyboard
    # keyboard.hook(on_key)
    time.sleep(0.0005)


while True:
    try:
        update()
    except Exception:
        for sub in subs:
            sub.terminate()
        raise
