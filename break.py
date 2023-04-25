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

class MyStdErr(TextIOBase):
    def write(self, s):
        if "underrun" in s:
            print("ahhhhhhh")
        print(s)
sys.stderr = MyStdErr()

dactyl_keys =[
    ['esc',   '1', '2', '3', '4', '5'],
    ['`',     'q', 'w', 'e', 'r', 't'],
    ['tab',   'a', 's', 'd', 'f', 'g'],
    ['shift', 'z', 'x', 'c', 'v', 'b'],
                  ['tab', '#'],
                                 ['delete', 'shift'],
                                 ['space', 'ctrl'],
                                 ['enter', 'alt'],
]

sound_index = 0

def key_pressed(e):
    global sound_index
    # print(e.name, " ", e.scan_code)
    for i, key in enumerate(['esc', '`', 'tab', 'shift']):
        if key == e.name:
            sound_index = i

keyboard.on_press(key_pressed)

# pygame.init()

# pygame.display.set_mode((0, 0))
pygame.midi.init()
pygame.mixer.init(buffer=1024)
device_id = 3
# device_id = pygame.midi.get_default_input_id()
for i in range(pygame.midi.get_count()):
    print(i, pygame.midi.get_device_info(i))
    (_,name,inp,out,opened) = pygame.midi.get_device_info(i)
    if name == "TR-8S MIDI 1" and input == 1:
        device_id = i
        print("using device ", i)


dir_path = os.path.dirname(os.path.realpath(__file__))
sounds = [pygame.mixer.Sound(dir_path + f'/143-2bar-00{i}.wav') for i in range(6)]

def current_sound():
    print(f'sound index {sound_index}')
    return sounds[sound_index]


# sound = pygame.mixer.Sound("click143.wav")
# while pygame.mixer.get_busy() == True:
#     continue

CLOCK = 0b11111000
START = 0b11111010
STOP = 0b11111100

# probably better to do this with milliseconds so it's agnostic to BPM
# CLOCK_LAG = 2

lag_time = 0.028

beat_interval = 42069
beat_start = time.time()

# lag_time = 0.028
lag_time = 0.058
max_beats = 8
beat = 0
clock_count = 0
midi = pygame.midi.Input(device_id)
is_started = False
played_sound = False
while True:
    # time.sleep(1)
    events = midi.read(1)
    if len(events) == 1:
        (status, d1, d2, d3) = events[0][0]

        if status == START:
            is_started = True
            clock_count = 0
            beat = 0
            # could play sample with later start time so it's not out of sync the first time around
            current_sound().play()

        if status == STOP:
            is_started = False


        if status == CLOCK:
            clock_count += 1

        if clock_count == 24:
            beat = (beat + 1) % 8
            print(beat + 1)
            clock_count = 0
            now = time.time()
            beat_interval = now - beat_start
            beat_start = now
            if not played_sound and beat == 0 and is_started:
                current_sound().play()
            played_sound = False

        # if beat == 0 and clock_count == 0 and is_playing:
        #     sound.play()
        beat_predicted = time.time() - beat_start >= beat_interval - lag_time
        if beat == max_beats - 1 and beat_predicted and not played_sound and is_started:
            current_sound().play()
            played_sound = True
    pygame.display.flip()
