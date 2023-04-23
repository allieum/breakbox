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
# while pygame.mixer.get_busy() == True:
#     continue

CLOCK = 0b11111000
START = 0b11111010
STOP = 0b11111100

# probably better to do this with milliseconds so it's agnostic to BPM
CLOCK_LAG = 2



beat_interval = None
beat_start = time.time()

max_beats = 8
beat = 0
clock_count = 0
midi = pygame.midi.Input(device_id)
is_playing = False
while True:
    # time.sleep(1)
    events = midi.read(1)
    if len(events) == 1:
        (status, d1, d2, d3) = events[0][0]

        if status == START:
            is_playing = True
            # head start to make up for lag
            clock_count = CLOCK_LAG
            beat = 0
            # could play sample with later start time so it's not out of sync the first time around
            sound.play()

        if status == STOP:
            is_playing = False


        if status == CLOCK:
            clock_count += 1

        if clock_count == 24:
            beat = (beat + 1) % 8
            print(beat + 1)
            clock_count = 0
            now = time.time()
            beat_interval = now - beat_start
            beat_start = now

        # if beat == 0 and clock_count == 0 and is_playing:
        #     sound.play()

        if beat == max_beats - 1 and time.time() - beat_start:
