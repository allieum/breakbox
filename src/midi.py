from dataclasses import dataclass
import mido
from mido import MidiFile, Message
import os
import pygame.midi
import re
import time

import utility

logger = utility.get_logger(__name__)

CLOCK = 0b11111000
START = 0b11111010
STOP = 0b11111100

TIMEOUT = 6

time_prev_midi_message = time.time()
midi_input = None
midi_output = None
last_connect_attempt = None

# @dataclass
# class NoteEvent:

MIDI_DIR = f"../midi"
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
logger.info(parent)
# adding the parent directory to
# the sys.path.

# print(mido.get_output_names())
def load_midi_files() -> list[MidiFile]:
    logger.info(parent)
    return [MidiFile(f"{MIDI_DIR}/{m.group()}")
                for f in os.listdir(MIDI_DIR)
                if (m := re.fullmatch(r".+\.mid", f))]

def send_midi(file: MidiFile):
    odd = True
    trip = 0
    for msg in file:
        print(f"midi message {msg}")
        time.sleep(msg.time * 120 / 143)
        if not msg.is_meta and midi_output is not None:
            if msg.type == 'note_on':
                midi_output.note_on(0, 100)

                odd = not odd

                if odd and trip == 0:
                    midi_output.note_on(2, 127)
                elif odd:
                    midi_output.note_on(2 , 70)

                trip += 1
                if trip == 4:
                    trip = 0
                    midi_output.note_on(4, 127)

                logger.info(f"note on")
            if msg.type == 'note_off':
                midi_output.note_off(0)
                logger.info(f"note off")

def is_note_on(status):
    return status is not None and 0b11110000 & status == 0b10010000

def connect(block = False, suppress_output = False):
    global time_prev_midi_message, midi_input, midi_output
    device_id = None

    if not suppress_output:
        print("waiting for midi device...")

    while midi_input is None:
        pygame.midi.init()
        logger.info(pygame.midi.get_count())
        for i in range(pygame.midi.get_count()):
            # print(i, pygame.midi.get_device_info(i))
            (_,name,inp,_,_) = pygame.midi.get_device_info(i)
            logger.info(f"{name} {inp}")
            if name == b"TR-8S MIDI 1" and inp == 1:
                device_id = i
                midi_input = pygame.midi.Input(device_id)
                print(f"using input device {i}: {name}")
                time_prev_midi_message = time.time()
            if name == b"TR-8S MIDI 1" and inp == 0:
                midi_output = pygame.midi.Output(i)
                print(f"using output device {i}: {name}")
        if midi_input is None:
            pygame.midi.quit()
            if not block:
                return midi_input, midi_output
            time.sleep(0.5)
    return midi_input, midi_output

def reconnect(block = False, suppress_output = False):
    global midi_input, midi_output
    pygame.midi.quit()
    midi_input, midi_output = connect(block, suppress_output)

note_q = []
def get_status():
    global time_prev_midi_message
    if midi_input is None:
        return None, None
    try:
        events = midi_input.read(1)
        msg = events[0][0] if len(events) == 1 else None
        if msg is not None:
            time_prev_midi_message = time.time()
            if is_note_on(status := msg[0]):
                note_q.append(msg[1])
            return msg[0], msg[1:]
    except Exception as e:
        logger.warn(f"{e}")
        pass

    return None, None

def lost_connection():
    if midi_input is None:
        return False
    return time.time() - time_prev_midi_message > TIMEOUT
