import math
import sys
import time

import pygame
from pydub.utils import db_to_float
from utility import get_logger, make_even

logger = get_logger(__name__)

TS_TIME_DEFAULT = 0.030
TS_TIME_DELTA = 0.001
ts_time = TS_TIME_DEFAULT
stretch_fade = 0.005


def increase_ts_time(*_):
    global ts_time
    ts_time += TS_TIME_DELTA


def decrease_ts_time(*_):
    global ts_time
    if ts_time > TS_TIME_DELTA:
        ts_time -= TS_TIME_DELTA


def reset_ts_time():
    global ts_time
    ts_time = TS_TIME_DEFAULT


def fade(soundbytes, start, end, gain_start=0, gain_end=0):
    if start == end:
        return
    gain_delta = db_to_float(gain_end) - db_to_float(gain_start)
    gain_step_delta = gain_delta / (end - start)
    if end - start > len(soundbytes):
        logger.debug(f"fade length longer than sample ({len(soundbytes)})")
        return
    for i in range(start, end, 2):
        val = int.from_bytes(soundbytes[i:i + 2], sys.byteorder, signed=True)
        j = i - start
        step_gain = gain_step_delta * j + db_to_float(gain_start)
        newval = math.floor(val * step_gain)
        sb = bytes([newval & 0xff, (newval >> 8) & 0xff])
        soundbytes[i:i + 2] = sb


def fade_in(soundbytes, fade_time, sample_rate=22050):
    num_samples = math.floor(fade_time * sample_rate)
    fade(soundbytes, 0, num_samples * 2, gain_start=-120)


def fade_out(soundbytes, fade_time, sample_rate=22050):
    num_samples = math.floor(fade_time * sample_rate)
    start = len(soundbytes) - num_samples * 2
    fade(soundbytes, start, len(soundbytes), gain_end=-120)
    return soundbytes


def fadeinout(soundbytes, fade_time):
    # logger.debug(f"fading by {fade_time} samples {fade_time * SAMPLE_RATE}")
    fade_in(soundbytes, fade_time)
    fade_out(soundbytes, fade_time)
    return soundbytes


def timestretch(sound, rate, sound_data, fade_time=0.005, sample_rate=22050):
    logger.info(
        f"start stretch x{rate} ({fade_time} fade), {ts_time}ms chunks")
    chunk_time = ts_time
    wav = sound.get_raw()
    new_wav = bytearray(make_even(math.ceil(len(wav) / rate)))
    logger.debug(f"{len(wav)} {len(new_wav)} vs {len(wav) / rate}")

    chunk_size = math.ceil(chunk_time * sample_rate) * 2  # 2 bytes per sample
    growth_factor = 1 / rate

    # print(f"chunk_time {chunk_time} chunk_size {chunk_size} growth_factor {growth_factor}")

    for i in range(0, len(wav), chunk_size):
        finout = fadeinout(bytearray(wav[i:i + chunk_size]), fade_time)
        if i == 0:
            fout = fade_out(bytearray(wav[i:i + chunk_size]), fade_time)
            chunks = fout + finout * max(0, math.floor(growth_factor) - 1)
        else:
            chunks = finout * math.floor(growth_factor)
        partial_factor = growth_factor - math.floor(growth_factor)
        leftover_chunk_size = make_even(
            math.floor(chunk_size * partial_factor))
        stretched_bytes = chunks
        if leftover_chunk_size != 0:
            leftover_chunk = fadeinout(
                bytearray(wav[i:i + leftover_chunk_size]), fade_time)
            stretched_bytes += leftover_chunk
        j = make_even(math.floor(i * growth_factor))
        k = j + len(stretched_bytes)
        if k > len(new_wav):
            k = len(new_wav)
            stretched_bytes = fade_out(stretched_bytes[:k - j], fade_time)
        new_wav[j:k] = stretched_bytes
    # write_wav(new_wav, "ts.wav")

    new_sound = pygame.mixer.Sound(new_wav)
    logger.debug(
        f"finish stretch x{rate} ({fade_time} fade) {sound.get_length() / rate} calculated length vs {new_sound.get_length()} actual")
    sound_data[new_sound].source = sound
    sound_data[new_sound].source_step = sound_data[sound].source_step
    return new_sound


def change_rate(sound, rate, sound_data):
    wav = sound.get_raw()
    new_wav = bytearray(math.ceil(len(wav) / rate))

    # from new_wav -> wav
    def convert_sample_index(i):
        return make_even(math.floor(i * rate))

    for i in range(0, len(new_wav), 2):
        j = convert_sample_index(i)
        new_wav[i:i + 2] = wav[j:j + 2]

    new_sound = pygame.mixer.Sound(new_wav)
    sound_data[new_sound].source = sound
    sound_data[new_sound].source_step = sound_data[sound].source_step
    return new_sound


def pitch_shift(sound, semitones, sound_data):
    start = time.time()
    if semitones == 0:
        return sound
    ratio = 1.05946  # 12th root of 2
    rate = ratio ** semitones
    inverse = ratio ** -semitones

    fade_time = 0.002
    if semitones > 0:
        new_sound = timestretch(change_rate(
            sound, rate, sound_data), inverse, sound_data, fade_time)
    else:
        new_sound = change_rate(timestretch(
            sound, inverse, sound_data, fade_time), rate, sound_data)
    logger.info(
        f"shifting by {semitones} semitones took {time.time() - start}s")
    return new_sound
