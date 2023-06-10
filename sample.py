import math
import sys
import pygame.mixer
import os
import time
from collections import deque, defaultdict
import concurrent.futures
from datetime import datetime
from pydub import AudioSegment
from pydub.utils import db_to_float
import re
from dataclasses import dataclass
from typing import Optional, List

import modulation
import utility

logger = utility.get_logger(__name__)
# logger.setLevel('WARN')

bank = 0
BANK_SIZE = 6
NUM_BANKS = 4
SAMPLE_RATE = 22050

pygame.mixer.init(frequency=SAMPLE_RATE, buffer=256, channels=1)
pygame.mixer.set_num_channels(32)
logger.info(pygame.mixer.get_init())

@dataclass
class SoundData:
    playtime: float
    bpm: int
    step: int
    source: Optional['SoundData']

    def __init__(self) -> None:
        pass

playtime = {}
sound_data = defaultdict(SoundData)

def remaining_time(sound):
    if sound is None:
        return 0
    if sound.get_num_channels() == 0:
        logger.error("sound not playing")
        return 0
    if sound not in sound_data:
        return 0
    return max(0, sound.get_length() - (time.time() - sound_data[sound].playtime))

def write_wav(soundbytes, filename):
    AudioSegment(
        soundbytes,
        sample_width=2,
        frame_rate=SAMPLE_RATE,
        channels=1
    ).export(filename, format='wav')

dir_path = os.path.dirname(os.path.realpath(__file__))
samples = []
def load_samples():
    sample_dir = f'{dir_path}/samples'
    for f in sorted(os.listdir(sample_dir)):
        if m := re.fullmatch(r"([0-9]{2,3}).+([0-9]{3})?.wav", f):
            print(f)
            bpm = int(m.group(1))
            samples.append(Sample(f"{sample_dir}/{m.group()}", bpm))
        else:
            logger.warn(f"wrong filename format for {f}, not loaded")
    print([s.name for s in current_samples()])

def current_samples() -> List['Sample']:
   return samples[bank * BANK_SIZE : BANK_SIZE * (bank + 1)]

channels = set()

class Sample:
    MAX_VOLUME = 1
    num_slices = 32
    timeout = 0.005
    lookahead = 0.001
    audio_executor = concurrent.futures.ThreadPoolExecutor(max_workers=6)

    def __init__(self, file, bpm):
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
        self.bpm = bpm
        self.load(file)
        self.last_printed = 0
        self.pitch = modulation.Param(0)
        self.gate = modulation.Param(1.0)
        self.gate_period = modulation.Param(1)
        self.gate_count = 0
        self.cache = {
            'pitch': {
                **dict.fromkeys(range(self.num_slices), {})
            },
            'halftime': {
                **dict.fromkeys(range(self.num_slices), {})
            }
        }

    def load(self, file):
        logger.debug(f"loading sample {file}")
        self.sound = pygame.mixer.Sound(file)
        # self.sound.sound_data = "test"
        # self.sound = pygame.mixer.Sound(file)
        self.sound.set_volume(0) # default mute
        step_time = 60 / self.bpm / 4
        wav = self.sound.get_raw()
        num_steps = round(self.sound.get_length() / step_time)
        slice_size = math.ceil(len(wav) / num_steps)
        self.sound_slices = [pygame.mixer.Sound(wav[i:i + slice_size]) for i in range(0, len(wav), slice_size)]
        for i, s in enumerate(self.sound_slices):
            sound_data[s].bpm = self.bpm
            sound_data[s].step = i
            sound_data[s].source = None

    def step_repeat_start(self, index, length):
        index %= len(self.sound_slices)
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

    def gate_increase(self):
        if self.gate.value == 1:
            return
        self.gate.value += 0.1

    def gate_decrease(self):
        if self.gate.value <= 0.1:
            return
        self.gate.value -= 0.1

    def gate_period_increase(self):
        self.gate_period.value += 1

    def gate_period_decrease(self):
        if self.gate_period.value <= 1:
            return
        self.gate_period.value -= 1

    def transform_slices(self, f):
        for i in range(len(self.sound_slices)):
            future = self.audio_executor.submit(self.transform_slice, i, f)
            future.add_done_callback(future_done)
            yield future

    def transform_slice(self, i, f):
        self.sound_slices[i] = f(self.sound_slices[i])

    def set_and_queue_slice(self, i, t, sound_generator):
        self.sound_slices[i] = sound_generator()
        self.queue(self.sound_slices[i], t, i)
        return self.sound_slices[i]

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
        logger.debug(f"{self.name} muted {len(self.sound_queue)}") # self.sound.set_volume(0)
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

    def queue(self, sound, t, step):
        logger.debug(f"queued sound in {self.name} for {datetime.fromtimestamp(t)}")
        # _, prev_t = self.sound_queue[len(self.sound_queue) - 1] if len(self.sound_queue) > 0 else None, None
        if self.gate_count == 0:
            sound.set_volume(1.0)
        self.gate_count = (self.gate_count + 1) % self.gate_period.get(step)
        self.sound_queue.append((sound, t, step))
        # if prev_t and prev_t > t:
        #     logger.error(f"{self.name} saw out of order sound queue")

    # call provided fn to create sound and add to queue
    def queue_async(self, generate_sound, t, step):
        logger.debug(f"{self.name} scheduling async sound for {t}")
        future = self.audio_executor.submit(lambda: self.queue(generate_sound(), t, step))
        future.add_done_callback(future_done)

    def queue_and_replace_async(self, generate_sound, t, step):
        logger.debug(f"{self.name} scheduling async sound for {t}")
        future = self.audio_executor.submit(self.set_and_queue_slice, step, t, generate_sound)
        future.add_done_callback(future_done)
        return future

    def warn_dropped(self, dropped, now):
        if (n := len(dropped)) == 0:
            return
        _, scheduled, step = dropped[0]
        msg = f"{self.name} dropped {n}/{n+len(self.sound_queue)} samples stale by: {1000 * (now - scheduled - self.timeout):10.6}ms for step {step}"
        msg += f" sched. {datetime.fromtimestamp(scheduled)}"
        # for _, scheduled in dropped:
        #     msg += f" {1000 * (now - scheduled - self.timeout):10.6}ms"
        logger.warn(msg)

    # returns callable to do the sound making
    def process_queue(self, now, step_duration):
        if self.channel and (playing := self.channel.get_sound()) is not None:
            # TODO input step into gate
            gate_time = self.gate.get() * playing.get_length()
            if playing in sound_data and now - sound_data[playing].playtime >= gate_time:
                playing.set_volume(0)

        logger.debug(f"{self.name} start process queue")
        if len(self.sound_queue) == 0:
            logger.debug(f"{self.name}: queue empty")
            return None

        if (size := len(self.sound_queue)) > 1:
            if not self.last_printed == size and not self.step_repeat:
                self.last_printed = size
                # print(f"{self.filename} has queue of {size}; {self.channel.get_busy()} {self.channel.get_queue()}")

        _sound, t, _ = self.sound_queue[0]

        dropped = []
        while now > t + self.timeout:
            dropped.append(self.sound_queue.popleft())
            if len(self.sound_queue) == 0:
                self.warn_dropped(dropped, now)
                return None
            _sound, t, _ = self.sound_queue[0]
        self.warn_dropped(dropped, now)


        in_play_window = now >= t - self.lookahead
        in_queue_window = now >= t - self.lookahead - step_duration
        if not in_queue_window:
            logger.debug(f"{self.name}: too early for queue her")
            return None
        if not in_play_window and self.channel is None:
            logger.debug(f"{self.name}: too early for channel her")
            return None

        if self.channel and not self.channel.get_busy() and not in_play_window: # and in_queue_window:
            return None

        if not in_play_window and self.channel and self.channel.get_busy() and self.channel.get_queue() is not None:
            logger.debug(f"{self.name}: channel full")
            return None

        if self.channel and not self.channel.get_busy() and (s := self.channel.get_queue()) is not None:
            logger.warn(f"{self.name} weird state, ghost queue? let's try clear it")
            self.channel.stop()
            # return lambda: self.channel.play(s)

        logger.debug(f"{self.name} processing {datetime.fromtimestamp(t)}")
        if len(self.sound_queue) == 0:
            logger.warn(f"{self.name}: queue cleared by other thread?")
            return None
        sound, t, step = self.sound_queue.popleft()
        if sound is not _sound:
            logger.error("sound is not sound!")
        if self.channel is None:
            logger.debug(f"{self.name}: played sample on new channel")
            return self.play_step(lambda s: self.play_sound_new_channel(s), sound, step, t)
        if in_play_window:
            if self.channel.get_busy():
                playing = self.channel.get_sound()
                logger.warn(f"{self.name} interrupted sample with {remaining_time(playing)}s left")
                logger.warn(f"sample length {playing.get_length()} vs step length {step_duration}")
            logger.debug(f"{self.name}: played sample")
            return self.play_step(lambda s: self.play_sound(s), sound, step, t)
        if self.channel.get_queue() is None and in_queue_window:
            playing = self.channel.get_sound()
            predicted_finish = time.time() + remaining_time(playing)
            if (error := predicted_finish - t) > 0.010:
                logger.debug(f"{self.name} queueing sample would make it late by {error}, putting back on queue")
                self.sound_queue.appendleft((sound, t, step))
                return None
            if error < -0.010:
                logger.warn(f"queueing sample would make it early by {-error}, putting back on queue")
                self.sound_queue.appendleft((sound, t, step))
                return None
            self.channel.queue(sound)
            sound_data[sound].playtime = predicted_finish
            # self.channel.queue(sound)
            logger.debug(f"{self.name}: queued sample")
            return None
            # return self.play_step(lambda s: self.queue_sound(s, t), sound, step, t)

        logger.warn(f"what wrong? {self.name} {now - t} busy:{self.channel.get_busy()} channel.queue: {self.channel.get_queue()} is_queue {in_queue_window} is_play {in_play_window}")
        msg = ""
        for i in range(len(self.sound_queue) - 1, -1, -1):
            _, t = self.sound_queue[i]
            msg += f"{now - t} "
        logger.warn(f"queue contents: {msg}")

    def queue_sound(self, sound, t):
        playing = self.channel.get_sound()
        self.channel.queue(sound)
        sound_data[sound].playtime = time.time() + remaining_time(playing)
        # todo if we're gonna be late, don't do it!!!
        if abs(sound_data[sound].playtime - t) > 0.005:
            logger.warn(f"scheduled for {sound_data[sound].playtime} vs ideal {t}")

    def play_step(self, sound_player, sound, step, t):
        def fn():
            sound_player(sound)
            return step, t
        return fn

    def play_sound(self, sound):
        self.channel.play(sound)
        sound_data[sound].playtime = time.time()

    def play_sound_new_channel(self, sound):
        self.channel = sound.play()
        sound_data[sound].playtime = time.time()
        channels.add(self.channel)
        print(f"seen {(n := len(channels))} channels")
        pygame.mixer.set_reserved(n)

    def queue_step(self, step, t, step_interval):
        srlength = self.step_repeat_length * 2 if self.halftime else self.step_repeat_length
        if self.step_repeat and step in range(self.step_repeat_index % srlength, len(self.sound_slices), srlength):
            self.step_repeating = True
            # self.mute()
            self.sound_queue.clear()
            slices = self.sound_slices[self.step_repeat_index: self.step_repeat_index + self.step_repeat_length]
            for i, s in enumerate(slices):
                ts =  t + i * step_interval
                slice_step = step + i
                if self.halftime:
                    ts = t + i * step_interval * 2
                    self.queue_async(lambda: timestretch(s, 0.5, stretch_fade), ts, slice_step)
                    logger.debug(f"queueing {s}")
                elif (p := self.pitch.get(step + i)) != 0:
                    self.queue_async(lambda: self.change_pitch(self.step_repeat_index + i, s, p), ts, slice_step)
                else:
                    self.queue(s, ts, slice_step)
        if not self.step_repeating:
            if not self.is_muted() or self.looping:
                sound = self.sound_slices[step % len(self.sound_slices)]
                if self.halftime:
                    if step % 2 != 0:
                        return
                    sound = self.sound_slices[(step // 2) % len(self.sound_slices)]
                    self.queue_async(lambda: timestretch(sound, 0.5, stretch_fade), t, step)
                elif (p := self.pitch.get(step)) != 0:
                    logger.debug(f"{self.name} setting pitch to {p}")
                    self.queue_async(lambda: self.change_pitch(step, sound, p), t, step)
                else:
                    if (sound_bpm := sound_data[sound].bpm) != self.bpm:
                        logger.info(f"{self.name} step {step} stretching sample from {sound_bpm} to {self.bpm}")
                        future = self.queue_and_replace_async(lambda: timestretch(sound, self.bpm / sound_bpm), t, step)
                        future.add_done_callback(lambda f: set_sound_bpm(f.result(), self.bpm))
                    else:
                        self.queue(sound, t, step)

    def change_pitch(self, step, sound, semitones):
        logger.info(f"{self.name}: step {step} by {semitones} semitones")
        # if semitones in (cache := self.cache['pitch'][sound]):
        #     logger.info(f"{self.name}: got from cache step {step} by {semitones} semitones")
        #     return cache[semitones]
        pitched_sound = pitch_shift(sound, semitones)
        # cache[semitones] = pitched_sound
        logger.info(f"{self.name}: caching step {step} by {semitones} semitones")
        # logger.info(f"cache {cache}")
        return pitched_sound

    def pitch_mod(self, sequence):
        self.modulate(sequence, self.pitch, 16, modulation.Lfo.Shape.TRIANGLE, 8, 8)

    def cancel_pitch_mod(self):
        pass
        # self.pitch.lfo = None

    def modulate(self, sequence, param, period, shape, amount, steps):
        lfo = modulation.Lfo(period, shape)
        param.modulate(lfo, amount, steps)

def set_sound_bpm(sound, bpm):
    sound_data[sound].bpm = bpm

def future_done(f):
    if (e := f.exception()):
        raise e


def play_samples(step_duration):
    logger.debug("playing samples")
    now = time.time()
    play_hooks = [s.process_queue(now, step_duration) for s in current_samples()]
    played_steps = [hook() if hook else hook for hook in play_hooks]
    if any(played_steps):
        logger.debug(played_steps_string(played_steps))

def played_steps_string(played_steps):
     s = "\n" * 10 + "============\n"
     for nt in played_steps:
         if nt:
            n,t = nt
            s += f"{datetime.fromtimestamp(t)} - {n}\n"
         else:
            s += "--\n"
     s += "============\n"
     return s

def queues_empty():
    return all([len(s.sound_queue) == 0] for s in current_samples())

def queue_samples(step, t, step_duration):
    for sample in current_samples():
        sample.queue_step(step, t, step_duration)

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

TS_TIME_DEFAULT = 0.060
TS_TIME_DELTA = 0.001
ts_time = TS_TIME_DEFAULT
stretch_fade = 0.005
def increase_ts_time():
    global ts_time, stretch_fade
    # ts_time += TS_TIME_DELTA
    stretch_fade += .001


def decrease_ts_time():
    global ts_time, stretch_fade
    if ts_time > TS_TIME_DELTA:
        # ts_time -= TS_TIME_DELTA
        stretch_fade -= .001
    else:
        ts_time /= 2

def reset_ts_time():
    global ts_time
    ts_time = TS_TIME_DEFAULT

# def fade(soundbytes, start, end, gainfn):
def fade(soundbytes, start, end, gain_start=0, gain_end=0):
    # samples = math.floor(fade_time * SAMPLE_RATE)
    # logger.info(f"")
    if start == end:
        return
    gain_delta = db_to_float(gain_end) - db_to_float(gain_start)
    gain_step_delta = gain_delta / (end - start)
    startsize = len(soundbytes)
    if end - start > len(soundbytes):
        logger.debug(f"fade length longer than sample ({len(soundbytes)})")
        return
    for i in range(start, end, 2):
        val = int.from_bytes(soundbytes[i:i + 2], sys.byteorder, signed=True)
        j = i - start
        # use it
        # step_gain = gain_step_delta * j + gain_start
        # newval = math.floor(val * db_to_float(step_gain))
        step_gain = gain_step_delta * j + db_to_float(gain_start)
        newval = math.floor(val * step_gain)
        sb = bytes([newval & 0xff, (newval >> 8) & 0xff])
        soundbytes[i:i + 2] = sb
        # logger.debug(f"{step_gain}db {db_to_float(step_gain)}")
    # if len(soundbytes) != startsize:
    #     logger.error(f"{start} {end} {gainfn} {startsize} grew by {len(soundbytes) - startsize}")

def fade_in(soundbytes, fade_time):
    samples = math.floor(fade_time * SAMPLE_RATE)
    fade(soundbytes, 0, samples * 2, gain_start=-120)

def fade_out(soundbytes, fade_time):
    samples = math.floor(fade_time * SAMPLE_RATE)
    start = len(soundbytes) - samples * 2
    fade(soundbytes, start, len(soundbytes), gain_end=-120)
    return soundbytes

def fadeinout(soundbytes, fade_time):
    # logger.debug(f"fading by {fade_time} samples {fade_time * SAMPLE_RATE}")
    fade_in(soundbytes, fade_time)
    fade_out(soundbytes, fade_time)
    return soundbytes

def timestretch(sound, rate, fade_time=0.005):
    logger.info(f"start stretch x{rate} ({fade_time} fade)")
    chunk_time = ts_time
    wav = sound.get_raw()
    new_wav = bytearray(make_even(math.ceil(len(wav) / rate)))
    logger.debug(f"{len(wav)} {len(new_wav)} vs {len(wav) / rate}")

    chunk_size = math.ceil(chunk_time * SAMPLE_RATE) * 2 # 2 bytes per sample
    growth_factor =  1 / rate

    # print(f"chunk_time {chunk_time} chunk_size {chunk_size} growth_factor {growth_factor}")

    for i in range(0, len(wav), chunk_size):
        finout = fadeinout(bytearray(wav[i:i + chunk_size]), fade_time)
        if i == 0:
            fout = fade_out(bytearray(wav[i:i + chunk_size]), fade_time)
            chunks = fout + finout * max(0, math.floor(growth_factor) - 1)
        else:
            chunks = finout * math.floor(growth_factor)
        partial_factor = growth_factor - math.floor(growth_factor)
        leftover_chunk_size = make_even(math.floor(chunk_size * partial_factor))
        stretched_bytes = chunks
        if leftover_chunk_size != 0:
            leftover_chunk = fadeinout(bytearray(wav[i:i + leftover_chunk_size]), fade_time)
            stretched_bytes += leftover_chunk
        j = make_even(math.floor(i * growth_factor))
        k = j + len(stretched_bytes)
        if k > len(new_wav):
            k = len(new_wav)
            stretched_bytes = fade_out(stretched_bytes[:k - j], fade_time)
        new_wav[j:k] = stretched_bytes
    # write_wav(new_wav, "ts.wav")

    if len(new_wav) % 2 == 1:
        logger.info(f"odd one out")

    new_sound = pygame.mixer.Sound(new_wav)
    logger.debug(f"finish stretch x{rate} ({fade_time} fade) {sound.get_length() / rate} calculated length vs {new_sound.get_length()} actual")
    sound_data[new_sound].source = sound
    sound_data[new_sound].step = sound_data[sound].step
    return new_sound

# s = current_samples()[0]
# b = bytearray(s.sound.get_raw())
# fade_out(b, 4)
# pygame.mixer.Sound(b).play()

# for fs in range(0,10,2):
#     fs /= 1000
#     ts = timestretch(s.sound_slices[0], 0.5, fs)
#     write_wav(ts.get_raw(), f"ts{fs}.wav")

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

    new_sound = pygame.mixer.Sound(new_wav)
    sound_data[new_sound].source = sound
    sound_data[new_sound].step = sound_data[sound].step
    return new_sound

def pitch_shift(sound, semitones):
    if semitones == 0:
        return sound
    ratio = 1.05946 # 12th root of 2
    rate = ratio ** semitones
    inverse = ratio ** -semitones

    stretch_fade = 0.002
    if semitones > 0:
        return timestretch(change_rate(sound, rate), inverse, stretch_fade)
    return change_rate(timestretch(sound, inverse, stretch_fade), rate)

def stretch_samples(target_bpm):
    # schedulers = [sample.transform_slices(lambda s: timestretch(s, target_bpm / sample.bpm)) for sample in samples]
    # while any([get_next(scheduler) for scheduler in schedulers]):
    #     pass
    for s in samples:
        s.bpm = target_bpm

def get_next(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None

# unmute first sample
# current_samples()[0].unmute()

# for s in current_samples():
#     s.unmute()
# orig = current_samples()[0].sound
# print(len(orig.get_raw()) / 4 / SAMPLE_RATE)
# stretched = timestretch(current_samples()[0].sound, 0.4)
# pitched = change_rate(current_samples()[0].sound, 0.6)
# print(len(stretched.get_raw()) / 4 / SAMPLE_RATE)
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
