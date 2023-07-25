from concurrent.futures import thread
from multiprocessing import Process, Queue
from threading import Thread
import time
# import rtmidi
import sys
from types import TracebackType
import keyboard
# from ctypes import *
# from contextlib import contextmanager
from datetime import datetime

from sequence import sequence
import sample
import control
import keys
import midi
import utility
import display
import lights

from dmx import DMXInterface
import blink


# with DMXInterface() as interface:
#


logger = utility.get_logger(__name__)
current_time = datetime.now().strftime("%H:%M:%S")
logger.debug(f"Start time = {current_time}")

dmx_interface = None
try:
    dmx_interface = DMXInterface()
except:
    pass

def on_key(e):
    if e.event_type == keyboard.KEY_DOWN:
        keys.key_pressed(e)
    elif e.event_type == keyboard.KEY_UP:
        keys.key_released(e)
keyboard.hook(on_key)

control.init()
sample.load_samples()
display.init(sample.all_samples())
midi.connect()
midi.load_midi_files()
sequence.control_bpm(control.encoder)

def bounce(step):
    x = step % 32
    if x < 16:
        return x + 1
    return 16 - (x % 16)

def bounce_lights(step):
    # tri = modulation.triangle(15)
    # light = round(tri(step % 16) * 7)
    light = bounce(step)
    logger.debug(f"light {light}")
    return (light,)

def lights_for_step(step):
    light_index = step % 8 + 1 + 8
    mirror_index = -(light_index - 8) + 8 + 1
    return (light_index, mirror_index)

# if (now := time.time()) - last_dmx > 0.050: # and last_dmx_step != sequence.step:
# TODO move into own file
def update_dmx(step, note_number=None):
    if dmx_interface is None:
        return
    # last_dmx = now
    # last_dmx_step = sequence.step
    logger.debug(f"lighting dmx step {step}")
    color = [0, 0, 0, 0, 0, 0]
    time.sleep(0.020)

    blink.Light.scale(0.8)

    for i, s in enumerate(sample.current_samples()):
        if s.channel and s.channel.get_busy():
            source_step = sample.sound_data[s.channel.get_sound()].source_step
            if source_step != sequence.step:
                logger.debug(f"source step {source_step}")
            color[i] = 255
            for j in bounce_lights(source_step):
                blink.lights[(j + i * 3) % len(blink.lights)].absorb(color)
        # blink.lights[i].absorb([0,0,0,0,0,0])

    for note_number in midi.note_q:
        if note_number == 0:
            for light in blink.lights:
                # logger.info(f"bass flash {step} {note_number}")
                light.absorb([255, 0, 0, 255, 255, 255])
    midi.note_q.clear()

    # light_index = sequence.step % 8 + 1 + 8
    # mirror_index = -(light_index - 8) + 8 + 1
    # if any(color):
    #     blink.lights[light_index].set(color)
    #     blink.lights[mirror_index].set(color)
    dmx_interface.set_frame(list(blink.Light.data))
    now = time.time()
    dmx_interface.send_update()
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
    sample_states = [lights.SampleState.of(s, keys.selected_sample, sequence.step, i) for i, s in enumerate(sample.current_samples())]
    # if lights.refresh_ready(samples_on):
        # lights.refreshing = True
    # lights.update(samples_on)

    if midi.is_note_on(status):
        # logger.info(f"{data} {status}")
        note_number = data[0]
        sample.Sample.audio_executor.submit(update_dmx, 0, note_number)

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

    state = ()
    # display.update(sequence.bpm.get(), sample_states)

    if midi.lost_connection():
        if sequence.midi_started:
            print("lost midi connection")
            sequence.stop_midi()
        midi.reconnect(suppress_output = True)

    # sys.stdout.flush()
    # sys.stderr.flush()
    # import keyboard
    # keyboard.hook(on_key)
    time.sleep(0.0005)


while True:
    try:
        update()
    except Exception as e:
        for sub in subs:
            sub.terminate()
        raise e
