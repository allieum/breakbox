from concurrent.futures import thread
from multiprocessing import Process, Queue
from threading import Thread
import time
import sys
import keyboard
from datetime import datetime

from sequence import sequence
import sample
import control
import keys
import midi
import utility
import display
import lights
from light_patterns import BouncePattern

import blink

logger = utility.get_logger(__name__)
current_time = datetime.now().strftime("%H:%M:%S")
logger.debug(f"Start time = {current_time}")


def on_key(e):
    if e.event_type == keyboard.KEY_DOWN:
        keys.key_pressed(e)
    elif e.event_type == keyboard.KEY_UP:
        keys.key_released(e)
keyboard.hook(on_key)

display.init()
lights.init()
control.init()
sample.load_samples()
midi.connect()
sequence.control_bpm(control.encoder)

light_manager = blink.LightManager()
light_manager.add_pattern(BouncePattern([255, 0, 0, 0 ,0 ,0]))

def update_lights(step):
    light_manager.step(step)
sequence.on_step(lambda s: sample.Sample.audio_executor.submit(update_lights, s))

def dmx_updater():
    while True:
        light_manager.send_dmx()
        time.sleep(0.03) #TODO: tune? adjust for drift?

thread = Thread(target=dmx_updater, daemon=True)
thread.start()

lq = Queue(1)
p = Process(target=lights.run, args=(lq,))
p.start()

last_dmx = time.time()
last_dmx_step = None
while True:
    # control.update()
    sequence.update(midi.get_status())
    sample.play_samples(sequence.step_duration())
    sample_states = [lights.SampleState.of(s, keys.selected_sample, sequence.step) for s in sample.current_samples()]
    # if lights.refresh_ready(samples_on):
        # lights.refreshing = True
    # lights.update(samples_on)
    try:
        lq.put(sample_states, block=False)
    except:
        pass
    # logger.info(f"putting {samples_on} in queue")
        # f = sample.Sample.audio_executor.submit(lights.update, samples_on)
        # f.add_done_callback(lambda _: lights.refresh_done())

    # state = (sequence.bpm.get())
    # display.update(state)

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
