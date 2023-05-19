import math
import pygame.mixer
import os
import time
from collections import deque
import concurrent.futures

import utility
from datetime import datetime

logger = utility.get_logger(__name__)

pygame.mixer.init(frequency=22050, buffer=256)
pygame.mixer.set_num_channels(32)
# if actual != 12:
#     print(f"requested {12} channels, got {actual}")

dir_path = os.path.dirname(os.path.realpath(__file__))
bank = 0
BANK_SIZE = 6
NUM_BANKS = 2

# copied from sequence.py b/c circular import and unmotivated
step_interval = 60 / 143 / 4

channels = set()

class Sample:
    MAX_VOLUME = 1
    num_slices = 32
    timeout = 0.005
    lookahead = 0.001
    audio_executor = concurrent.futures.ThreadPoolExecutor(max_workers=6)

    def __init__(self, file):
        self.name = file.split("samples/")[1]
        self.looping = False
        self.step_repeat = False    # mode active
        self.step_repeating = False # currently repeating steps
        self.step_repeat_length = 0 # in steps
        self.step_repeat_index = 0  # which step to repeat
        self.channel = None
        self.sound_queue = deque()
        self.muted = True
        self.halftime = False
        self.load(file)
        self.last_printed = 0

    def load(self, file):
        logger.debug(f"loading sample {file}")
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

    def swap_channel(self, other):
        self.channel, other.channel = other.channel, self.channel

    def step_repeat_stop(self):
        if not self.step_repeat:
            return
        self.step_repeat = False
        self.step_repeating = False
        # if self.channel is not None:
        #     self.channel.fadeout(15)
        self.sound_queue.clear()
        # self.set_mute(self.step_repeat_was_muted)

    def mute(self):
        logger.debug(f"{self.name} muted {len(self.sound_queue)}")
        # self.sound.set_volume(0)
        self.muted = True
        # self.sound_queue = []

    def unmute(self):
        logger.debug(f"{self.name} unmuted {len(self.sound_queue)}")
        # self.sound.set_volume(Sample.MAX_VOLUME)
        self.muted = False

    def set_mute(self, mute):
        if mute:
            self.mute()
        else:
            self.unmute()

    def is_muted(self):
        # return self.sound.get_volume() == 0
        return self.muted

    def toggle_mute(self):
        if self.is_muted():
            self.unmute()
        else:
            self.mute()

    def play(self):
        pass
        # self.sound.stop()
        # self.sound.play()

    def queue(self, sound, t):
        logger.info(f"queued sound in {self.name} for {datetime.fromtimestamp(t)}")
        _, prev_t = self.sound_queue[len(self.sound_queue) - 1] if len(self.sound_queue) > 0 else None, None
        self.sound_queue.append((sound, t))
        if prev_t and prev_t > t:
            logger.error(f"{self.name} saw out of order sound queue")

    def warn_dropped(self, dropped, now):
        if (n := len(dropped)) == 0:
            return
        _, scheduled = dropped[0]
        msg = f"{self.name} dropped {n}/{n+len(self.sound_queue)} samples stale by: {1000 * (now - scheduled - self.timeout):10.6}ms"
        msg += f" sched. {datetime.fromtimestamp(scheduled)}"
        # for _, scheduled in dropped:
        #     msg += f" {1000 * (now - scheduled - self.timeout):10.6}ms"
        logger.warn(msg)

    # returns callable to do the sound making
    def process_queue(self, now):
        logger.debug(f"{self.name} start process queue")
        if len(self.sound_queue) == 0:
            logger.debug(f"{self.name}: queue empty")
            return None


        if (size := len(self.sound_queue)) > 1:
            if not self.last_printed == size and not self.step_repeat:
                self.last_printed = size
                # print(f"{self.filename} has queue of {size}; {self.channel.get_busy()} {self.channel.get_queue()}")

        _sound, t = self.sound_queue[0]

        dropped = []
        while now > t + self.timeout:
            dropped.append(self.sound_queue.popleft())
            if len(self.sound_queue) == 0:
                self.warn_dropped(dropped, now)
                return None
            _sound, t = self.sound_queue[0]
        self.warn_dropped(dropped, now)

        in_play_window = now >= t - self.lookahead
        in_queue_window = now >= t - self.lookahead - step_interval
        if not in_queue_window:
            logger.debug(f"{self.name}: too early for queue her")
            return None
        if not in_play_window and self.channel is None:
            logger.debug(f"{self.name}: too early for channel her")
            return None

        if self.channel and not self.channel.get_busy() and not in_play_window: # and in_queue_window:
            return None

        if not in_play_window and self.channel and self.channel.get_busy() and self.channel.get_queue() is not None:
            logger.info(f"{self.name}: channel full")
            return None

        if self.channel and not self.channel.get_busy() and self.channel.get_queue() is not None:
            logger.warn(f"{self.name} weird state, ghost queue? maybe it will go away on its own")
            return None

        logger.info(f"{self.name} processing {datetime.fromtimestamp(t)}")
        sound, _ = self.sound_queue.popleft()
        if sound is not _sound:
            logger.error("sound is not sound!")
        if self.channel is None:
            # self.channel = sound.play()
            logger.debug(f"{self.name}: played sample on new channel")
            return lambda: self.play_sound_new_channel(sound)
        if in_play_window:
            if self.channel.get_busy():
                logger.warn(f"{self.name} interrupted sample")
                # self.channel.stop() # is this needed
            # self.channel.play(sound)
            # print(f'{self.filename} played {sound} on existing {self.channel}')
            logger.debug(f"{self.name}: played sample")
            return lambda: self.channel.play(sound)
        if self.channel.get_queue() is None and in_queue_window:
            # utility.timer("queue")
            self.channel.queue(sound)
            # print(f'{self.filename} queued {sound} on existing {self.channel}')
            logger.debug(f"{self.name}: queued sample")
            return lambda: self.channel.queue(sound)

        # logger.info(f"done process queue for {self.name}: how you get here?")
        logger.warn(f"what wrong? {self.name} {now - t} busy:{self.channel.get_busy()} channel.queue: {self.channel.get_queue()} is_queue {in_queue_window} is_play {in_play_window}")
        msg = ""
        for i in range(len(self.sound_queue) - 1, -1, -1):
            _, t = self.sound_queue[i]
            msg += f"{now - t} "
        logger.warn(f"queue contents: {msg}")

    def play_sound_new_channel(self, sound):
        self.channel = sound.play()
        channels.add(self.channel)
        print(f"seen {(n := len(channels))} channels")
        pygame.mixer.set_reserved(n)

    def queue_step(self, step, t):
        if self.step_repeat and step in range(self.step_repeat_index % self.step_repeat_length, self.num_slices, self.step_repeat_length):
            self.step_repeating = True
            # self.mute()
            self.sound_queue.clear()
            for i, s in enumerate(self.sound_slices[self.step_repeat_index: self.step_repeat_index + self.step_repeat_length]):
                self.queue(s, t + i * step_interval)
        if not self.step_repeating:
            if not self.is_muted() or self.looping:
                sound = self.sound_slices[step]
                if self.halftime:
                    if step % 2 != 0:
                        return
                    self.queue_async(lambda: timestretch(sound, 0.5), t)
                else:
                    self.queue(sound, t)
            return

    # call provided fn to create sound and add to queue
    def queue_async(self, generate_sound, t):
        self.audio_executor.submit(lambda: self.queue(generate_sound(), t))

            #
            # next_slice = self.sound_slices[self.step_repeat_index]
            # if self.channel is None:
            #     self.channel = next_slice.play()
            # else:
                # print(f"{self.step_repeat_channel.get_busy()}")
                # self.step_repeat_channel.stop() # todo bug this can be None
                # self.channel.play(next_slice)
            # timer['played step'].tick()
            # t = time.time()
            # while next_slice.get_num_channels() > 0:
            #     pass
            # print(time.time() - t)

            # print(f"{step} playing {next_slice} on channel {self.step_repeat_channel}")
        # elif len(self.sound_queue) > 0:
        #     next_slice = self.sound_queue.pop(0)
        #     self.channel.play(next_slice)
            # print(f"{step} playing {next_slice} on channel {self.step_repeat_channel}")


samples = [Sample(f'{dir_path}/samples/143-2bar-{i:03}.wav') for i in range(12)]

def current_samples():
   return samples[bank * BANK_SIZE : BANK_SIZE * (bank + 1)]

def play_samples():
    # while True:
        # t = time.time()
        # futures = [Sample.audio_executor.submit(s.process_queue) for s in current_samples()]
        # concurrent.futures.wait(futures)
        # elapsed = time.time() - t
        # print(f"play_samples took {elapsed}s")
    logger.debug("playing samples")
    # logger.info(queues_to_string())
    now = time.time()
    play_hooks = [s.process_queue(now) for s in current_samples()]
    for hook in play_hooks:
        if hook is not None:
            hook()
    # if changed and not queues_empty():
    #     logger.info(queues_to_string())
    # time.sleep(0.010)

def queues_empty():
    return all([len(s.sound_queue) == 0] for s in current_samples())

def queue_samples(step, t):
    # if step == 0:
    #     pygame.mixer.stop()
    # utility.timer['step'].tick()
    for sample in current_samples():
        sample.queue_step(step, t)
        # utility.timer['interstep'].tick()

def queues_to_string():
     s = "\n============\n"
     for sample in current_samples():
         s += " " + "o" * len(sample.sound_queue) + "\n"
     s += "============\n"
     return s


def step_repeat_stop(length):
    for sample in [s for s in current_samples() if s.step_repeat_length == length]:
        sample.step_repeat_stop()

def stop_halftime():
    for s in current_samples():
        if s.halftime:
            s.sound_queue.clear()
            s.halftime = False

def make_even(x):
    if x % 2 == 1:
        x -= 1
    return x

TS_TIME_DEFAULT = 0.048
TS_TIME_DELTA = 0.001
ts_time = TS_TIME_DEFAULT
def increase_ts_time():
    global ts_time
    ts_time += TS_TIME_DELTA

def decrease_ts_time():
    global ts_time
    if ts_time > TS_TIME_DELTA:
        ts_time -= TS_TIME_DELTA
    else:
        ts_time /= 2

def reset_ts_time():
    global ts_time
    ts_time = TS_TIME_DEFAULT

def timestretch(sound, rate):
    chunk_time = ts_time
    wav = sound.get_raw()
    new_wav = bytearray(math.ceil(len(wav) / rate))

    chunk_size = math.ceil(chunk_time * 22050) * 2 # 2 bytes per sample
    growth_factor =  1 / rate

    print(f"chunk_time {chunk_time} chunk_size {chunk_size} growth_factor {growth_factor}")

    for i in range(0, len(wav), chunk_size):
        chunks = wav[i:i + chunk_size] * math.floor(growth_factor)
        partial_factor = growth_factor - math.floor(growth_factor)
        leftover_chunk_size = make_even(math.floor(chunk_size * partial_factor))
        leftover_chunk = wav[i:i + leftover_chunk_size]
        stretched_bytes = chunks + leftover_chunk
        j = make_even(math.floor(i * growth_factor))
        new_wav[j:j + len(stretched_bytes)] = stretched_bytes

    return pygame.mixer.Sound(new_wav)


def change_rate(sound, rate):
    wav = sound.get_raw()
    new_wav = bytearray(math.ceil(len(wav) / rate))

    # from new_wav -> wav
    def convert_sample_index(i):
        j = make_even(math.floor(i * rate))
        return j

    for i in range(0, len(new_wav), 2):
        j = convert_sample_index(i)
        new_wav[i:i + 2] = wav[j:j + 2]

    return pygame.mixer.Sound(new_wav)
    
# unmute first sample
# current_samples()[0].unmute()

# for s in current_samples():
#     s.unmute()

# orig = current_samples()[0].sound
# print(len(orig.get_raw()) / 4 / 22050)
# stretched = timestretch(current_samples()[0].sound, 0.4)
# pitched = change_rate(current_samples()[0].sound, 0.6)
# print(len(stretched.get_raw()) / 4 / 22050)
# print(len(orig.get_raw()) / len(pitched.get_raw()))
# # stretched.play()
# pitched.play()

# ch = None
# for slice in current_samples()[3].sound_slices:
#     if ch is None:
#         ch = slice.play()
#     else:
#         while ch.get_queue():
#             time.sleep(0.001)
#         ch.queue(slice)
