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
from threading import Thread

from sequence import sequence
import sample
import control
import midi
import utility

logger = utility.get_logger(__name__)
current_time = datetime.now().strftime("%H:%M:%S")
logger.debug(f"Start time = {current_time}")

def on_key(e):
    if e.event_type == keyboard.KEY_DOWN:
        control.key_pressed(e)
    elif e.event_type == keyboard.KEY_UP:
        control.key_released(e)
keyboard.hook(on_key)

sample.load_samples()
midi.connect()

while True:
    sequence.update(midi.get_status())
    sample.play_samples(sequence.step_duration())

    if midi.lost_connection():
        if sequence.midi_started:
            print("lost midi connection")
            sequence.stop_midi()
        midi.reconnect(suppress_output = True)

    sys.stdout.flush()
    sys.stderr.flush()
    time.sleep(0.0005)
