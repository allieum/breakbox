from collections import defaultdict
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
from io import TextIOBase
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

STOP_KEY = 'delete'
NEXT_BANK_KEY = '#'
dactyl_keys =[
    ['esc',   '1', '2', '3', '4', '5'],
    ['`',     'q', 'w', 'e', 'r', 't'],
    ['tab',   'a', 's', 'd', 'f', 'g'],
    ['shift', 'z', 'x', 'c', 'v', 'b'],
                  ['tab', NEXT_BANK_KEY],
                                 [STOP_KEY, 'shift'],
                                 ['space', 'ctrl'],
                                 ['enter', 'alt'],
]

class Sample:
    def __init__(self, file):
        self.sound = pygame.mixer.Sound(file)
        self.sound.set_volume(0) # default mute
        self.queued = False

    def mute(self):
        self.sound.set_volume(0)

    def unmute(self):
        # low volume so poor pi soundcard doesn't clip playing multiple samples
        self.sound.set_volume(0.3)

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

pygame.mixer.init(buffer=1024)
dir_path = os.path.dirname(os.path.realpath(__file__))
bank = 0
BANK_SIZE = 6
NUM_BANKS = 2
samples = [Sample(dir_path + f'/samples/143-2bar-{i:03}.wav') for i in range(12)]

def current_samples():
   return samples[bank * BANK_SIZE : BANK_SIZE * (bank + 1)]

def play_samples():
    for i, sample in enumerate(current_samples()):
        if sample.queued:
            for j, sample in enumerate(current_samples()):
                sample.set_mute(i != j)
                sample.queued = False
    pygame.mixer.stop()
    for i, sample in enumerate(current_samples()):
        # print(f"sample {i} volume: {sample.sound.get_volume()}")
        sample.play()

# unmute first sample
current_samples()[0].unmute()

key_held = defaultdict(bool)
def key_pressed(e):
    global bank
    # print(e.name, " ", e.scan_code)
    for i, key in enumerate(dactyl_keys[0]):
        if key == e.name:
            for j, sample in enumerate(current_samples()):
                sample.queued = i == j
                # print(f"sample {j} queued: {sample.queued}")
            return
    for i, key in enumerate(dactyl_keys[1]):
        if key == e.name and not key_held[key]:
            current_samples()[i].toggle_mute()
            key_held[key] = True

    if STOP_KEY == e.name:
        # cancel ongoing mute toggles
        for key in dactyl_keys[1]:
            key_held[key] = False
        for sample in current_samples():
            sample.mute()

    if NEXT_BANK_KEY == e.name:
        looping_index = None
        for i, sample in enumerate(current_samples()):
            if not sample.is_muted() and not key_held[dactyl_keys[1][i]]:
                looping_index = i
                # print(f"looping index {i}")
        # cancel ongoing mute toggles
        for key in dactyl_keys[1]:
            key_held[key] = False
        bank = (bank + 1) % NUM_BANKS
        if looping_index is not None:
            current_samples()[looping_index].queued = True

def key_released(e):
    if not key_held[e.name]:
        return
    for i, key in enumerate(dactyl_keys[1]):
        if key == e.name:
            current_samples()[i].toggle_mute()
            key_held[key] = False

def on_key(e):
    if e.event_type == keyboard.KEY_DOWN:
        key_pressed(e)
    elif e.event_type == keyboard.KEY_UP:
        key_released(e)

keyboard.hook(on_key)

pygame.midi.init()
device_id = None
print("waiting for midi device...")
while device_id is None:
    for i in range(pygame.midi.get_count()):
        print(i, pygame.midi.get_device_info(i))
        (_,name,inp,out,opened) = pygame.midi.get_device_info(i)
        if name == b"TR-8S MIDI 1" and inp == 1:
            device_id = i
            print("using device ", i)
    if device_id == None:
        time.sleep(0.5)

CLOCK = 0b11111000
START = 0b11111010
STOP = 0b11111100

beat_interval = 42069
beat_start = time.time()

lag_time = 0.058
max_beats = 8
beat = 0
clock_count = 0
midi = pygame.midi.Input(device_id)
is_started = False
played_samples = False
with noalsaerr():
    while True:
        events = midi.read(1)
        if len(events) == 1:
            (status, d1, d2, d3) = events[0][0]

            if status == START:
                is_started = True
                clock_count = 0
                beat = 0

            if status == STOP:
                is_started = False
                pygame.mixer.stop()

            if status == CLOCK:
                clock_count += 1

            if clock_count == 24:
                beat = (beat + 1) % 8
                # print(beat + 1)
                clock_count = 0
                now = time.time()
                beat_interval = now - beat_start
                beat_start = now
                if not played_samples and beat == 0 and is_started:
                    play_samples()
                played_samples = False

            beat_predicted = time.time() - beat_start >= beat_interval - lag_time
            if beat == max_beats - 1 and beat_predicted and not played_samples and is_started:
                play_samples()
                played_samples = True
