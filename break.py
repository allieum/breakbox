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
# from ctypes import *
# from contextlib import contextmanager
from datetime import datetime

from sequence import Sequence
import sample
import midi

sequence = Sequence()

current_time = datetime.now().strftime("%H:%M:%S")
print("Start time =", current_time)


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

key_held = defaultdict(bool)
key_frozen = defaultdict(bool)
def key_active(key):
    return key_held[key] or key_frozen[key]

def key_pressed(e):
    for i, key in enumerate(LOOP_KEYS):
        if key == e.name:
            for j, s in enumerate(sample.current_samples()):
                s.queued = i == j
                # print(f"sample {j} queued: {sample.queued}")

    for i, key in enumerate(TOGGLE_KEYS):
        if key != e.name:
            continue
        if not key_held[key] and not key_frozen[key]:
            sample.current_samples()[i].toggle_mute()
        if not key_frozen[key]:
            key_held[key] = True
        if any([key_held[k] for k in HOLD_KEYS]):
            # print(f"freezing {key}")
            key_frozen[key] = True
        elif key_frozen[key]:
            # print(f"unfreezing {key}")
            key_frozen[key] = False
            process_release(key)
        # print(f"holding {key}")
        for step_repeat_key, length in SR_KEYS.items():
            if key_active(step_repeat_key):
                sample.current_samples()[i].step_repeat_start(sequence.step, length)
    #
    # if loop or toggle and sequence not started, start it
    if not sequence.is_started:
        for key in LOOP_KEYS + TOGGLE_KEYS:
            if key == e.name:
                sequence.start_internal()

    for step_repeat_key, length in SR_KEYS.items():
        if step_repeat_key != e.name:
            continue
        # print(f"holding {step_repeat_key}")
        if not key_frozen[step_repeat_key]:
            key_held[step_repeat_key] = True
        if any([key_held[k] for k in HOLD_KEYS]): # make hold_active fn
            key_frozen[step_repeat_key] = True
            # print(f"freezing {step_repeat_key}")
        elif key_frozen[step_repeat_key]:
            # print(f"unfreezing {step_repeat_key}")
            key_frozen[step_repeat_key] = False
            process_release(step_repeat_key)
        for i, key in enumerate(TOGGLE_KEYS):
            if key_active(key):
                sample.current_samples()[i].step_repeat_start(sequence.step, length)

    if K_STOP == e.name:
        # cancel held keys
        for key in key_frozen:
            key_held[key] = False
            key_frozen[key] = False
        for s in sample.current_samples():
            s.mute()
        if sequence.is_internal():
            sequence.stop_internal()

    if K_NEXT_BANK == e.name:
        looping_index = None
        for i, s in enumerate(sample.current_samples()):
            if not s.is_muted() and not key_active(TOGGLE_KEYS[i]):
                looping_index = i
                # print(f"looping index {i}")
        # cancel held keys
        for key in key_frozen:
            key_held[key] = False
            key_frozen[key] = False
        sample.bank = (sample.bank + 1) % sample.NUM_BANKS
        for s in [s for s in sample.current_samples()]:
            s.step_repeat_stop()
        if looping_index is not None:
            sample.current_samples()[looping_index].queued = True

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
            sample.current_samples()[i].step_repeat_stop()
            sample.current_samples()[i].toggle_mute()
    for key, length in SR_KEYS.items():
        if key != k:
            continue
        sample.step_repeat_stop(length)

def on_key(e):
    if e.event_type == keyboard.KEY_DOWN:
        key_pressed(e)
    elif e.event_type == keyboard.KEY_UP:
        key_released(e)

keyboard.hook(on_key)

midi.connect()

while True:
    sequence.update(midi.get_status())

    if midi.lost_connection():
        if sequence.midi_started:
            print("lost midi connection")
            sequence.stop_midi()
        midi.reconnect(suppress_output = True)

    time.sleep(0.001)
