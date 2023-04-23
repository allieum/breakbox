import pygame
import pygame.midi
import time
# import rtmidi


pygame.midi.init()
pygame.mixer.init()
device_id = 3
# device_id = pygame.midi.get_default_input_id()
for i in range(pygame.midi.get_count()):
    print(i, pygame.midi.get_device_info(i))
    (_,name,inp,out,opened) = pygame.midi.get_device_info(i)
    if name == "TR-8S MIDI 1" and input == 1:
        device_id = i
        print("using device ", i)

sound = pygame.mixer.Sound("143-2bar-000.wav")
# sound = pygame.mixer.Sound("click143.wav")
# while pygame.mixer.get_busy() == True:
#     continue

CLOCK = 0b11111000
START = 0b11111010
STOP = 0b11111100

# probably better to do this with milliseconds so it's agnostic to BPM
# CLOCK_LAG = 2



beat_interval = 42069
beat_start = time.time()

lag_time = 0.028
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
            sound.play()

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
                sound.play()
            played_sound = False

        # if beat == 0 and clock_count == 0 and is_playing:
        #     sound.play()
        beat_predicted = time.time() - beat_start >= beat_interval - lag_time
        if beat == max_beats - 1 and beat_predicted and not played_sound and is_started:
            sound.play()
            played_sound = True
