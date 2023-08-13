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

def init_selected_samples(bank_samples: list[sample.Sample]):
    BREAKS_START_PAD = 0
    VOCALS_START_PAD = 4
    keys.selected_sample = bank_samples[BREAKS_START_PAD % len(bank_samples)]
    dtxpro.selected_sample = bank_samples[VOCALS_START_PAD % len(bank_samples)]


keyboard.hook(on_key)

# control.init()
sample.load_samples_from_disk()
display.init(sample.all_samples())
midi.connect()
sample.on_load_bank(init_selected_samples)
init_selected_samples(sample.loaded_samples)

# sequence.on_step(lambda s: sample.Sample.audio_executor.submit(update_dmx, s))

subs = [
    Process(target=lights.run, args=(lights.q,)),
    Process(target=display.run, args=(display.display_queue, display.param_queue)),
]
for sub in subs:
    sub.start()

loop_counter = 0
def update():
    global loop_counter
    loop_counter += 1
    # control.update()
    midi_status, midi_data = midi.get_status()
    sequence.update(midi_status)
    sample.play_samples(sequence.step_duration())
    sample_states = [sample.SampleState.of(
        s, keys.selected_sample, sequence.step, i, dtxpro.selected_sample) for i, s in enumerate(sample.loaded_samples)]
    dtxpro.update()

    if midi.is_program_change(midi_status):
        prog_num = midi_data[0]
        if prog_num < sample.NUM_BANKS:
            sample.load_current_bank(prog_num)
        else:
            dtxpro.handle_program_change(prog_num)


    if midi.is_note_on(midi_status):
        note_number = midi_data[0]
        velocity = midi_data[1]
        if (dtxpad := dtxpro.struck_pad(note_number)) is not None and velocity != 0:
            dtxpro.hit_dtx_pad(sequence, dtxpad, velocity)

    if loop_counter % 5 == 0:
        try:
            lights.q.put(sample_states, block=False)
            display.display_queue.put(sample_states, block=False)
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
