import math
import pygame.mixer
import os

pygame.mixer.init(buffer=256)
pygame.mixer.set_num_channels(16)

dir_path = os.path.dirname(os.path.realpath(__file__))
bank = 0
BANK_SIZE = 6
NUM_BANKS = 2

class Sample:
    MAX_VOLUME = 1
    num_slices = 32

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
        slice_size = math.ceil(len(wav) / self.num_slices)
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
        if step in range(self.step_repeat_index % self.step_repeat_length, self.num_slices, self.step_repeat_length):
            self.mute()
            next_slice = self.sound_slices[self.step_repeat_index]
            if self.step_repeat_channel is None:
                self.step_repeat_channel = next_slice.play()
            else:
                # print(f"{self.step_repeat_channel.get_busy()}")
                # self.step_repeat_channel.stop() # todo bug this can be None
                self.step_repeat_channel.play(next_slice)
            # timer['played step'].tick()
            # t = time.time()
            # while next_slice.get_num_channels() > 0:
            #     pass
            # print(time.time() - t)

            # print(f"{step} playing {next_slice} on channel {self.step_repeat_channel}")
            self.step_repeat_channel.set_volume(self.MAX_VOLUME)
            self.step_repeat_queue = self.sound_slices[self.step_repeat_index + 1 : self.step_repeat_index + self.step_repeat_length]
        elif len(self.step_repeat_queue) > 0:
            next_slice = self.step_repeat_queue.pop(0)
            self.step_repeat_channel.play(next_slice)
            # print(f"{step} playing {next_slice} on channel {self.step_repeat_channel}")


samples = [Sample(f'{dir_path}/samples/143-2bar-{i:03}.wav') for i in range(12)]

def current_samples():
   return samples[bank * BANK_SIZE : BANK_SIZE * (bank + 1)]

# current_samples()[4].sound_slices[4].play(loops=32)

def play_samples(step = 0):
    # print(f'playing step {step}')
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

def step_repeat_stop(length):
    for sample in [s for s in current_samples() if s.step_repeat_length == length]:
        sample.step_repeat_stop()


# unmute first sample
current_samples()[0].unmute()
