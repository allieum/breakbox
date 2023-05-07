from collections import defaultdict
import math
import pygame
import pygame.event
import pygame.midi
# from pygame.locals import *
import pygame.key
import time
# import rtmidi
import os
import sys
import keyboard
from ctypes import *
from contextlib import contextmanager
from datetime import datetime

current_time = datetime.now().strftime("%H:%M:%S")
print("Start time =", current_time)

def restart_program():
    python = sys.executable
    os.execl(python, python, * sys.argv)

ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)

# sidestep ALSA underrun errors. not that cute
def py_error_handler(filename, line, function, err, fmt):
    # print("underrun CHOMP", fmt)
    if b'occurred' in fmt:
        print("we're done here")
        current_time = datetime.now().strftime("%H:%M:%S")
        print("Error time =", current_time)
        restart_program()
        # os.execl('/home/drum/breakbox/restart.sh', *sys.arg)

c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

@contextmanager
def noalsaerr():
    asound = cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
    yield
    asound.snd_lib_error_set_handler(None)

MAX_BEATS = 8
STEPS_PER_BEAT = 4
MAX_STEPS = MAX_BEATS * STEPS_PER_BEAT

K_STOP = 'delete'
K_NEXT_BANK = '#'

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
                                 [K_STOP, 'shift'],
                                 ['space', 'ctrl'],
                                 ['enter', 'alt'],
]

LOOP_KEYS = dactyl_keys[0]
TOGGLE_KEYS = dactyl_keys[1]
HOLD_KEYS = dactyl_keys[3]

class Sample:
    # low volume so poor pi soundcard doesn't clip playing multiple samples
    MAX_VOLUME = 0.3

    def __init__(self, file):
        self.queued = False
        self.step_repeat = False
        self.step_repeat_length = 0 # in steps
        self.step_repeat_index = 0  # which step to repeat
        self.step_repeat_channel = None
        self.step_repeat_queue = []
        self.load(file)

    def load(self, file):
        print(f"loading sample {file}")
        self.sound = pygame.mixer.Sound(file)
        self.sound.set_volume(0) # default mute
        wav = self.sound.get_raw()
        slice_size = math.ceil(len(wav) / MAX_STEPS)
        self.sound_slices = [pygame.mixer.Sound(wav[i:i + slice_size]) for i in range(0, len(wav), slice_size)]

    def step_repeat_start(self, index, length):
        if not self.step_repeat:
            self.step_repeat_was_muted = self.is_muted()
            self.step_repeat_length = length
            self.step_repeat_index = index - index % length
            self.step_repeat = True
            # print(f"starting step repeat at {self.step_repeat_index} with length {length}")
        elif length != self.step_repeat_length:
            self.step_repeat_length = length
        else:
            return

    def step_repeat_stop(self):
        if not self.step_repeat:
            return
        # print("stopping step repeat")
        self.step_repeat = False
        if self.step_repeat_channel is not None:
            self.step_repeat_channel.fadeout(15)
            self.step_repeat_channel = None
        self.step_repeat_queue = []
        self.set_mute(self.step_repeat_was_muted)

    def mute(self):
        self.sound.set_volume(0)

    def unmute(self):
        self.sound.set_volume(Sample.MAX_VOLUME)

    def set_mute(self, mute):
        if mute:
            self.mute()
        else:
            self.unmute()

    def is_muted(self):
        return self.sound.get_volume() == 0

    def toggle_mute(self):
        if self.is_muted():
            self.unmute()
        else:
            self.mute()

    def play(self):
        self.sound.stop()
        self.sound.play()


    def play_step(self, step):
        if not self.step_repeat:
            return
        if step in range(self.step_repeat_index % self.step_repeat_length, MAX_STEPS, self.step_repeat_length):
            self.mute()
            next_slice = self.sound_slices[self.step_repeat_index]
            if self.step_repeat_channel is None:
                self.step_repeat_channel = next_slice.play()
            else:
                self.step_repeat_channel.play(next_slice)
            # print(f"{step} playing {next_slice} on channel {self.step_repeat_channel}")
            self.step_repeat_channel.set_volume(Sample.MAX_VOLUME)
            self.step_repeat_queue = self.sound_slices[self.step_repeat_index + 1 : self.step_repeat_index + self.step_repeat_length]
        elif len(self.step_repeat_queue) > 0:
            next_slice = self.step_repeat_queue.pop(0)
            self.step_repeat_channel.play(next_slice)
            # print(f"{step} playing {next_slice} on channel {self.step_repeat_channel}")

# TODO try smaller buffer size for lower latency
pygame.mixer.init(buffer=1024)
pygame.mixer.set_num_channels(12)
dir_path = os.path.dirname(os.path.realpath(__file__))
bank = 0
BANK_SIZE = 6
NUM_BANKS = 2
samples = [Sample(dir_path + f'/samples/143-2bar-{i:03}.wav') for i in range(12)]

def current_samples():
   return samples[bank * BANK_SIZE : BANK_SIZE * (bank + 1)]

def play_samples(step = 0):
    if step == 0:
        for i, sample in enumerate(current_samples()):
            if sample.queued:
                for j, sample in enumerate(current_samples()):
                    sample.set_mute(i != j)
                    sample.queued = False
        pygame.mixer.stop()
        for i, sample in enumerate(current_samples()):
            # print(f"sample {i} volume: {sample.sound.get_volume()}")
            sample.play()
    for sample in current_samples():
        sample.play_step(step)

# unmute first sample
current_samples()[0].unmute()



hold_active = False
key_held = defaultdict(bool)
key_frozen = defaultdict(bool)
def key_active(key):
    return key_held[key] or key_frozen[key]

def key_pressed(e):
    global bank, step, hold_active
    # print(e.name, " ", e.scan_code)
    for i, key in enumerate(LOOP_KEYS):
        if key == e.name:
            for j, sample in enumerate(current_samples()):
                sample.queued = i == j
                # print(f"sample {j} queued: {sample.queued}")
            return

    for i, key in enumerate(TOGGLE_KEYS):
        if key != e.name:
            continue
        if not key_held[key] and not key_frozen[key]:
            current_samples()[i].toggle_mute()
        if not key_frozen[key]:
            key_held[key] = True
        if any([key_held[k] for k in HOLD_KEYS]):
            print(f"freezing {key}")
            key_frozen[key] = True
        elif key_frozen[key]:
            print(f"unfreezing {key}")
            key_frozen[key] = False
            process_release(key)
        # print(f"holding {key}")
        for step_repeat_key, length in SR_KEYS.items():
            if key_active(step_repeat_key):
                current_samples()[i].step_repeat_start(step, length)

    for step_repeat_key, length in SR_KEYS.items():
        if step_repeat_key != e.name:
            continue
        # print(f"holding {step_repeat_key}")
        if not key_frozen[step_repeat_key]:
            key_held[step_repeat_key] = True
        if any([key_held[k] for k in HOLD_KEYS]):
            key_frozen[step_repeat_key] = True
            print(f"freezing {step_repeat_key}")
        elif key_frozen[step_repeat_key]:
            print(f"unfreezing {step_repeat_key}")
            key_frozen[step_repeat_key] = False
            process_release(step_repeat_key)
        for i, key in enumerate(TOGGLE_KEYS):
            if key_active(key):
                current_samples()[i].step_repeat_start(step, length)

    if K_STOP == e.name:
        # cancel held keys
        for key in key_frozen:
            key_held[key] = False
            key_frozen[key] = False
        for sample in current_samples():
            sample.mute()

    if K_NEXT_BANK == e.name:
        looping_index = None
        for i, sample in enumerate(current_samples()):
            if not sample.is_muted() and not key_active(TOGGLE_KEYS[i]):
                looping_index = i
                # print(f"looping index {i}")
        # cancel held keys
        for key in key_frozen:
            key_held[key] = False
            key_frozen[key] = False
        bank = (bank + 1) % NUM_BANKS
        if looping_index is not None:
            current_samples()[looping_index].queued = True

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


def key_released(e):
    if not key_held[e.name]:
        return
    key_held[e.name] = False
    if key_frozen[e.name]:
        return
    process_release(e.name)

def process_release(k):
    for i, key in enumerate(TOGGLE_KEYS):
        if key == k:
            current_samples()[i].step_repeat_stop()
            current_samples()[i].toggle_mute()
    for key, length in SR_KEYS.items():
        if key != k:
            continue
        for sample in [s for s in current_samples() if s.step_repeat_length == length]:
            sample.step_repeat_stop()


def on_key(e):
    if e.event_type == keyboard.KEY_DOWN:
        key_pressed(e)
    elif e.event_type == keyboard.KEY_UP:
        key_released(e)

keyboard.hook(on_key)

# how come tr8 can't come second?
def connect_midi():
    global time_prev_midi_message
    device_id = None
    print("waiting for midi device...")
    while device_id is None:
        pygame.midi.init()
        for i in range(pygame.midi.get_count()):
            # print(i, pygame.midi.get_device_info(i))
            (_,name,inp,out,opened) = pygame.midi.get_device_info(i)
            if name == b"TR-8S MIDI 1" and inp == 1:
                device_id = i
                print(f"using device {i}: {name}")
                time_prev_midi_message = time.time()
                break
        if device_id is None:
            pygame.midi.quit()
            time.sleep(0.5)
    return pygame.midi.Input(device_id)

# pygame.midi.init()
# device_id = None
# print("waiting for midi device...")
# while device_id is None:
#     for i in range(pygame.midi.get_count()):
#         print(i, pygame.midi.get_device_info(i))
#         (_,name,inp,out,opened) = pygame.midi.get_device_info(i)
#         if name == b"TR-8S MIDI 1" and inp == 1:
#             device_id = i
#             print("using device ", i)
#     if device_id == None:
#         time.sleep(0.5)

CLOCK = 0b11111000
START = 0b11111010
STOP = 0b11111100

beat_interval = 42069
beat_start = time.time()

lag_time = 0.058
beat = 0

step_start = time.time()
step = 0
# step_interval = 42069
step_interval = 60 / 143 / 4

clock_count = 0
is_started = False
played_samples = False
played_step = False

midi = connect_midi()
time_prev_midi_message = time.time()

# with noalsaerr():
while True:
    step_predicted = time.time() - step_start >= step_interval - lag_time and not played_step
    if step_predicted and is_started:
        next_step = (step + 1) % MAX_STEPS
        play_samples(next_step)
        played_step = True
    events = midi.read(1)
    if len(events) == 1:
        (status, d1, d2, d3) = events[0][0]
        time_prev_midi_message = time.time()

        if status == START:
            is_started = True
            clock_count = 0
            beat = 0
            step = 0

        if status == STOP:
            is_started = False
            pygame.mixer.stop()

        if status == CLOCK:
            clock_count += 1

        if clock_count > 0 and clock_count % (24 / STEPS_PER_BEAT) == 0:
            step = (step + 1) % MAX_STEPS
            # print(step)
            now = time.time()
            # step_interval = now - step_start
            step_start = now
            played_step = False

        if clock_count == 24:
            beat = (beat + 1) % MAX_BEATS
            # print(beat + 1)
            clock_count = 0
            now = time.time()
            beat_interval = now - beat_start
            beat_start = now

    time.sleep(0.001)
    if time.time() - time_prev_midi_message > 6.0:
        pygame.midi.quit()
        is_started = False
        midi = connect_midi()
