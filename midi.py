import pygame.midi
import time


CLOCK = 0b11111000
START = 0b11111010
STOP = 0b11111100
TIMEOUT = 6

time_prev_midi_message = time.time()
midi_input = None
last_connect_attempt = None

def connect(block = False, suppress_output = False):
    global time_prev_midi_message, midi_input
    device_id = None

    if not suppress_output:
        print("waiting for midi device...")

    while device_id is None:
        pygame.midi.init()
        for i in range(pygame.midi.get_count()):
            # print(i, pygame.midi.get_device_info(i))
            (_,name,inp,_,_) = pygame.midi.get_device_info(i)
            if name == b"TR-8S MIDI 1" and inp == 1:
                device_id = i
                print(f"using device {i}: {name}")
                time_prev_midi_message = time.time()
                break
        if device_id is None:
            pygame.midi.quit()
            if not block:
                return None
            time.sleep(0.5)
    midi_input = pygame.midi.Input(device_id)
    return midi_input

def reconnect(block = False, suppress_output = False):
    global midi_input
    pygame.midi.quit()
    midi_input = connect(block, suppress_output)

def get_status():
    global time_prev_midi_message
    if midi_input is None:
        return None
    events = midi_input.read(1)
    status = events[0][0][0] if len(events) == 1 else None
    if status is not None:
        time_prev_midi_message = time.time()
    return status

def lost_connection():
    if midi_input is None:
        return False
    return time.time() - time_prev_midi_message > TIMEOUT
