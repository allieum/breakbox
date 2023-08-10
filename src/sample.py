import concurrent.futures
import functools
import math
import os
import re
import threading
import time
from collections import defaultdict, namedtuple, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from queue import PriorityQueue
from random import random
from typing import Optional

import modulation
import pygame.mixer
from effects import pitch_shift, stretch_fade, timestretch
from modulation import Lfo, Param
from pydub import AudioSegment
from utility import TimeInterval, get_logger, make_even

logger = get_logger(__name__)
NUM_BANKS = 10
BANK_SIZE = 6
SAMPLE_RATE = 22050
bank = Param(0, min_value=0, max_value=NUM_BANKS - 1, round=True)

pygame.mixer.init(frequency=SAMPLE_RATE, buffer=256, channels=1)
pygame.mixer.set_num_channels(32)
logger.info(pygame.mixer.get_init())

dir_path = os.path.dirname(os.path.realpath(__file__))


# TODO: does it make sense to create banks.py for sample loading / selection?
#
loaded_samples = []
sample_banks = []
channels = set()
def load_samples():
    for i in range(1, NUM_BANKS + 1):
        sample_dir = f'{dir_path}/samples/{i}'
        sample_banks.append(bnk := [])
        for f in sorted(os.listdir(sample_dir)):
            if m := re.fullmatch(r"([0-9]{2,3}).+.wav", f):
                # print(f)
                bpm = int(m.group(1))
                bnk.append(Sample(f"{sample_dir}/{m.group()}", bpm, i - 1))
            else:
                logger.warn(f"wrong filename format for {f}, not loaded")
        logger.info([s.name for s in bnk])
    load_bank(0)


def all_samples() -> list['Sample']:
    return functools.reduce(lambda a, b: a + b, sample_banks)

def load_bank(bank_index):
    global loaded_samples
    old_samples = loaded_samples
    bank.set(bank_index)
    loaded_samples = sample_banks[bank_index].copy()
    if len(old_samples) == 0:
        return
    for new_sample, old_sample in zip(loaded_samples, old_samples, strict=True):
        new_sample.swap_channel(old_sample)

@dataclass
class SampleState:
    playing: bool = field(default=False)
    bank: int = field(default=0)
    length: float = field(default=0, compare=False)
    steps: int = field(default=0, compare=False)
    selected: bool = field(default=False)
    dtx_selected: bool = field(default=False)
    recording: bool = field(default=False)
    step: int | None = field(compare=False, default=None)
    pad: int = field(default=0)

    @staticmethod
    def of(sample: 'Sample', selected_sample: Optional['Sample'], step, pad, dtx_selected_sample: Optional['Sample']):
        if sample is None:
            return SampleState()

        selected = selected_sample == sample
        dtx_selected = dtx_selected_sample == sample
        length = sum(s.get_length() for s in sample.get_sound_slices())
        length = sample.sound.get_length()
        steps = len(sample.sound_slices)
        if sample.step_repeating:
            length *= sample.step_repeat_length / len(sample.sound_slices)
            steps = sample.step_repeat_length
        progress = (step % steps) / steps
        length *= (1 - progress)
        length -= 0.5
        return SampleState(sample.is_playing(), sample.bank,
                           length, steps, selected, dtx_selected, sample.recording, step, pad)


@dataclass
class SoundData:
    playtime: float = field(default=0)
    bpm: int = field(default=143)
    source_step: int = field(default=0)
    step: int | None = field(default=None)
    semitones: int = field(default=0)
    source: Optional[pygame.mixer.Sound] = field(default=None)


sound_data = defaultdict(SoundData)


# Slice of a sound, with effects applied
@dataclass(order=True)
class SampleSlice:
    start_time: float
    step: int = field(compare=False)
    sound: pygame.mixer.Sound = field(compare=False)
    fx: list[Callable[[pygame.mixer.Sound],
                      pygame.mixer.Sound]] = field(compare=False, default_factory=list)

    def apply_fx(self):
        if self.fx is None or time.time() > self.start_time:
            return
        while len(self.fx) > 0:
            effect = self.fx[0]
            self.sound = effect(self.sound)
            # fx is shared with another thread, don't pop until effect is applied
            self.fx.pop(0)

    def t_string(self):
        return datetime.fromtimestamp(self.start_time)


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
        channels=1,
    ).export(filename, format='wav')



class Sample:
    MAX_VOLUME = 1
    slices_per_loop = 64
    timeout = 0.005
    lookahead = 0.001
    audio_executor = concurrent.futures.ThreadPoolExecutor(max_workers=6)
    SpiceParams = namedtuple('SpicedParams', [
        'skip_gate', 'extra_gate', 'stretch_chance', 'gate_length', 'volume', 'pitch', 'scatter',
    ])

    @dataclass
    class LatchRepeat:
        steps: deque[SampleSlice] = field(default_factory=deque)
        timer: threading.Timer = field(default_factory=lambda: threading.Timer(0, lambda: 0))
        active: bool = field(default=False)
        is_rotating: bool = field(default=False)

    @dataclass
    class Roll:
        last_hit: float
        sound: pygame.mixer.Sound
        pitch_delta: int
        step: int

        def update(self, gen_sound):
            def set_sound():
                self.sound = gen_sound()
            Sample.audio_executor.submit(set_sound)

    def __init__(self, file, bpm, bank):
        self.name = file.split("samples/")[1]
        self.bank = bank
        self.looping = False
        self.step_repeat = False    # mode active
        self.step_repeating = False  # currently repeating steps
        self.step_repeat_length = 0  # in steps
        self.step_repeat_lengths = []
        self.step_repeat_index = 0  # which step to repeat
        self.seq_start = 0
        self.scatter_queue: int | None = None
        self.channel: pygame.mixer.Channel | None = None
        self.sound_queue: PriorityQueue[SampleSlice] = PriorityQueue()
        self.muted = True
        self.mute_timer: None | threading.Timer = None
        self.unmute_intervals: list[TimeInterval] = []
        self.mute_override = False
        self.oneshot_start_step = 0
        self.oneshot_offset = 0.0
        self.step_repeat_was_muted = False
        self.step_repeat_timer = None
        self.played_steps: deque[SampleSlice] = deque(maxlen=Sample.slices_per_loop)
        self.latch = Sample.LatchRepeat()

        self.roll: Sample.Roll | None = None

        # TODO rate param affected by these two
        self.halftime = False
        self.halftime_timer: Optional[threading.Timer] = None
        self.quartertime_timer: Optional[threading.Timer] = None
        self.quartertime = False

        self.bpm = bpm
        self.load(file)
        self.recorded_steps: list[None | pygame.mixer.Sound] = [
            None] * len(self.sound_slices)
        self.recording = False
        self.last_printed = 0
        self.gate_mirror = None
        self.gates = [1] * len(self.sound_slices)
        self.gate_fade = 0.020
        self.unspiced_gates = self.gates
        self.spice_level = modulation.Param(0, min_value=0, max_value=1)
        self.spice_level.add_change_handler(self.spice_gates)
        self.spices_param = self.SpiceParams(
            skip_gate=modulation.SpiceParams(
                max_chance=0.05, max_delta=0, spice=self.spice_level, step_data=None),
            extra_gate=modulation.SpiceParams(
                max_chance=0.7, max_delta=0, spice=self.spice_level, step_data=None),
            stretch_chance=modulation.SpiceParams(
                max_chance=0.2, max_delta=0, spice=self.spice_level, step_data=None),
            gate_length=modulation.SpiceParams(
                max_chance=0.5, max_delta=0.25, spice=self.spice_level, step_data=None),
            volume=modulation.SpiceParams(
                max_chance=0.3, max_delta=0.25, spice=self.spice_level, step_data=None),
            pitch=modulation.SpiceParams(
                max_chance=0.1, max_delta=3, spice=self.spice_level, step_data=None),
            scatter=modulation.SpiceParams(
                max_chance=0.2, max_delta=16, spice=self.spice_level, step_data=None, integer=True),
        )

        self.gate = modulation.Param(1.0, min_value=0.25, max_value=1)
        self.gate.add_change_handler(self.update_gates)
        self.gate_period = modulation.Param(2, min_value=1, max_value=32)
        self.volume = modulation.Param(
            1, min_value=0, max_value=1).spice(self.spices_param.volume)
        self.pitch = modulation.Param(
            0, min_value=-12, max_value=12, round=True).spice(self.spices_param.pitch)
        self.pitch_timer: None | threading.Timer = None

        self.dice()

    def load(self, file):
        logger.debug(f"loading sample {file}")
        self.sound = pygame.mixer.Sound(file)
        step_time = 60 / self.bpm / 4
        samples_per_step = round(step_time * SAMPLE_RATE)
        num_steps = round(self.sound.get_length() / step_time)
        wav = self.sound.get_raw()[:2 * samples_per_step * num_steps]
        slice_size = make_even(math.ceil(len(wav) / num_steps))
        logger.info(f"{self.name} slice size {slice_size}")
        self.sound_slices: list[pygame.mixer.Sound] = [pygame.mixer.Sound(
            buffer=wav[i:i + slice_size]) for i in range(0, len(wav), slice_size)]
        self.stretched_slices = self.sound_slices.copy()
        for i, s in enumerate(self.sound_slices):
            sound_data[s].bpm = self.bpm
            sound_data[s].source_step = i
            sound_data[s].source = None
            sound_data[s].semitones = 0

    def elapsed_sequence_time(self):
        return time.time() - self.seq_start

    def trigger_oneshot(self, step, offset):
        self.oneshot_start_step = step
        self.oneshot_offset = offset

    def drum_trigger(self, step: int, pitched=True, volume=0.5):
        max_roll_interval = 0.200
        if self.roll and time.time() - self.roll.last_hit > max_roll_interval:
            self.roll = None
        elif self.roll:
            self.roll.last_hit = time.time()

        slices = self.get_sound_slices()
        slice_i = step % len(slices)
        if slice_i % 2 == 1:
            slice_i -= 1
        if not self.roll:
            queued_sound = self.get_step_sound(
                slice_i, time.time(), force=True)
            # TODO apply_fx never called for sound
            if not queued_sound:
                return
            delta = 1 if random() > 0.5 else -1  # noqa: PLR2004
            if pitched:
                self.roll = Sample.Roll(
                    time.time(), queued_sound.sound, delta, slice_i)
            else:
                self.play_sound(queued_sound.sound)

        if not self.roll:
            return

        next_sound = self.get_step_sound(slice_i + 1, time.time(), force=True)
        if not next_sound:
            return
        self.roll.sound.set_volume(volume + 0.25)
        self.play_sound(self.roll.sound)
        if self.channel:  # and not stretch:
            self.channel.queue(next_sound.sound)

        self.roll.update(functools.partial(self.change_pitch,
                         slice_i, self.roll.sound, self.roll.pitch_delta))

    # TODO maybe all step repeat could be implemented with latch?
    def start_latch_repeat(self, length: int, duration: Optional[float]=None):
        if self.latch.active:
            self.latch.timer = self.after_delay(self.stop_latch_repeat, self.latch.timer, duration)
            return
        if len(self.played_steps) < length:
            logger.info(f"{self.name} not enough recent steps to start latch repeat")
            return
        timer = self.after_delay(self.stop_latch_repeat, None, duration)
        latched_steps = deque(self.played_steps, maxlen=length)
        self.latch = Sample.LatchRepeat(latched_steps, timer, active=True)

    def stop_latch_repeat(self):
        self.latch.active = False

    def pitch_mod(self, delta, step, duration=None):
        self.step_repeat_start(step, 1, duration)
        self.pitch_timer = self.after_delay(
            self.pitch.mod_cancel, self.pitch_timer, duration)
        # TODO better way to tell if lfos are equivalent
        if (lfo := self.pitch.lfo) and lfo.enabled and lfo.shape == lfo.Shape.INC and self.pitch.scale == delta:
            return
        self.modulate(self.pitch, 1, Lfo.Shape.INC, delta)

    def pitch_mod_cancel(self):
        self.pitch.mod_cancel()

    def stop_oneshot(self):
        self.oneshot_start_step = 0
        self.oneshot_offset = 0

    def start_recording(self):
        self.recording = True
        if self.mute_override and not self.is_muted():
            logger.info(
                f"{self.name} start unmute interval [{self.elapsed_sequence_time()}]")
            self.unmute_intervals.append(
                TimeInterval(self.elapsed_sequence_time()))

    def stop_recording(self):
        self.recording = False
        if len(self.unmute_intervals) > 0 and not (last := self.unmute_intervals[-1]).has_end():
            last.end = self.elapsed_sequence_time()
            logger.info(
                f"{self.name} finished interval [{last.start} {last.end}]")

    def dice(self):
        for param in self.spices_param:
            data = []
            while len(data) < self.slices_per_loop:
                flip = random() > 0.5  # noqa: PLR2004
                grain_size = 2 if flip else 4
                if self.spices_param.skip_gate is param:
                    grain_size = 2
                data.extend([(random(), random())] * grain_size)
            param.dice(data[:self.slices_per_loop])
        self.spice_gates()

    def spice_gates(self, *_):
        spiced_gates = []
        for step, gate in enumerate(self.unspiced_gates):
            spicy_gate = gate
            if gate == 0 and self.spices_param.extra_gate.toss(step):
                spicy_gate = self.spices_param.gate_length.value(0.75, step)
            elif self.spices_param.skip_gate.toss(step):
                spicy_gate = 0
            else:
                spicy_gate = self.spices_param.gate_length.value(gate, step)
            spiced_gates.append(spicy_gate)
        self.gates = spiced_gates

    def start_halftime(self, duration=None):
        logger.info(f"{self.name} halftime started")
        self.halftime = True
        self.halftime_timer = self.after_delay(
            self.stop_halftime, self.halftime_timer, duration)

    def start_quartertime(self, duration=None):
        logger.info(f"{self.name} quartertime started")
        self.quartertime = True
        self.quartertime_timer = self.after_delay(
            self.stop_quartertime, self.quartertime_timer, duration)

    def stop_halftime(self, *_):
        logger.info(f"{self.name} halftime stopped")
        self.halftime = False

    def stop_quartertime(self, *_):
        logger.info(f"{self.name} quartertime stopped")
        self.quartertime = False

    def step_repeat_start(self, index, length, duration=None):
        self.step_repeat_timer = self.after_delay(
            self.step_repeat_stop, self.step_repeat_timer, duration)
        logger.info(
            f"starting step repeat at {self.step_repeat_index} with length {length}")
        if length in self.step_repeat_lengths:
            return
        index %= len(self.sound_slices)
        self.step_repeat_was_muted = self.is_muted()
        self.step_repeat_lengths.append(length)
        self.update_step_repeat(index)
        self.step_repeat = True

    def update_step_repeat(self, index):
        self.step_repeat_length = sum(self.step_repeat_lengths)
        quantized_length = 2 if self.step_repeat_length == 1 else self.step_repeat_length
        self.step_repeat_index = index - index % quantized_length

    def step_repeat_stop(self, length=None):
        if not self.step_repeat:
            return
        if length is None:
            self.step_repeat_lengths.clear()
            logger.info(f"{self.name} removing all lengths from step repeat")
        elif length not in self.step_repeat_lengths:
            return
        else:
            self.step_repeat_lengths.remove(length)
            logger.info(
                f"{self.name} removing {length} from step repeats {self.step_repeat_lengths}")
        self.step_repeating = False
        if len(self.step_repeat_lengths) == 0:
            self.step_repeat = False
            self.clear_sound_queue()
            logger.info(f"{self.name} step repeat off")
        else:
            self.update_step_repeat(self.step_repeat_index)

    def cancel_fx(self):
        logger.info(f"{self.name} cancelling sample fx")
        self.pitch.restore_default()
        self.spice_level.restore_default()
        self.volume.restore_default()
        self.default_gates()
        self.stop_latch_repeat()
        self.stop_stretch()
        self.step_repeat_stop()
        for i, sound in enumerate(self.sound_slices):
            self.sound_slices[i] = self.source_sound(sound)

    def invert_gates(self):
        def invert(gate):
            if gate == 0:
                return 1
            if gate == 1:
                return 0
            return -gate
        return [invert(g) for g in self.gates]

    def get_rate(self) -> float:
        rate = 1.0
        if self.halftime:
            rate *= 0.5
        if self.quartertime:
            rate *= 0.25
        return rate

    # TODO could use on_change for update_gates?
    def gate_increase(self):
        self.gate.set(delta=0.25)
        self.update_gates()

    def gate_decrease(self):
        self.gate.set(delta=-0.25)
        self.update_gates()

    def gate_period_increase(self):
        if self.gate_period.set(self.gate_period.value * 2) and self.gate.get() == 1:
            self.gate.set(0.5)
        self.update_gates()

    def gate_period_decrease(self):
        if self.gate_period.set(self.gate_period.value // 2) and self.gate.get() == 1:
            self.gate.set(0.5)
        self.update_gates()

    def default_gates(self):
        self.gate_period.restore_default()
        self.gate.restore_default()
        self.update_gates()

    def update_gates(self, gate=None):
        gates = [0.0] * len(self.unspiced_gates)
        step_length = self.sound_slices[0].get_length()
        period = self.gate_period.get()
        max_gate = period * step_length
        if gate is None:
            gate = self.gate.value
        gate_length = max_gate * gate
        steps = gate_length / step_length
        whole_steps = math.floor(steps)
        step_gate = steps - whole_steps
        for i in range(0, len(gates), period):
            gates[i:i + 1 + whole_steps] = [1] * whole_steps + [step_gate]
        self.unspiced_gates = gates
        self.spice_gates()
        if self.gate_mirror:
            self.gate_mirror.gates = self.invert_gates()

    def stop_stretch(self):
        if self.halftime or self.quartertime:
            self.clear_sound_queue()
            self.halftime = False
            self.quartertime = False

    def transform_slices(self, f):
        for i in range(len(self.sound_slices)):
            future = self.audio_executor.submit(self.transform_slice, i, f)
            future.add_done_callback(future_done)
            yield future

    def transform_slice(self, i, f):
        self.sound_slices[i] = f(self.sound_slices[i])

    def set_and_queue_slice(self, i, t, sound_generator):
        self.sound_slices[i % len(self.sound_slices)] = sound_generator()
        slicey = self.sound_slices[i % len(self.sound_slices)]
        self.queue_sound(SampleSlice(t, i, slicey))
        return slicey

    def set_slice(self, i, sound_generator):
        self.sound_slices[i % len(self.sound_slices)] = sound_generator()
        return self.sound_slices[i % len(self.sound_slices)]

    def swap_channel(self, other):
        self.channel, other.channel = other.channel, self.channel

    def mute(self, suppress_recording=False):
        if self.is_muted():
            return
        logger.info(f"{self.name} muted")
        self.muted = True
        if not suppress_recording:
            logger.debug(f"{self.unmute_intervals}")
        if self.recording and not suppress_recording \
                and len(self.unmute_intervals) > 0 \
                and not (last := self.unmute_intervals[-1]).has_end():
            last.end = self.elapsed_sequence_time()
            logger.info(
                f"{self.name} finished interval [{last.start} {last.end}]")
            logger.info(f"{self.name} intervals: {self.unmute_intervals}")
        self.clear_sound_queue()

    def unmute(self, duration=None, suppress_recording=False, step=None, offset=None):
        self.mute_timer = self.after_delay(
            self.mute, self.mute_timer, duration)
        if not self.is_muted():
            return
        logger.info(f"{self.name} unmuted duration={duration}")
        self.muted = False

        # if step is not None and offset is not None:
        #     self.partial_trigger(step, offset)

        if self.recording and not suppress_recording:
            logger.info(
                f"{self.name} start mute interval [{self.elapsed_sequence_time()}, ]")
            self.unmute_intervals.append(
                TimeInterval(self.elapsed_sequence_time()))

    def partial_trigger(self, step, offset):
        queued_sound = self.get_step_sound(step, time.time(), force=True)
        if not queued_sound:
            return
        done_ratio = offset / queued_sound.sound.get_length()
        wav = queued_sound.sound.get_raw()
        start_index = make_even(round(done_ratio * len(wav)))
        start_index %= len(wav)
        logger.info(
            f"partial trigger step {step} offset {offset} starting at {start_index} of {len(wav)}")
        sound = pygame.mixer.Sound(buffer=wav[start_index:])
        self.play_sound(sound)
        next_sound = self.get_step_sound(step + 1, time.time(), force=True)
        self.queue_sound(next_sound)

    def after_delay(self, action, timer: threading.Timer | None, duration) -> threading.Timer:
        if duration is None:
            return threading.Timer(0, lambda: 0)
        if timer is not None:
            timer.cancel()
        timer = threading.Timer(duration, action)
        timer.start()
        return timer

    def set_mute(self, mute):
        if mute:
            self.mute()
        else:
            self.unmute()

    def is_muted(self):
        return self.muted

    def toggle_mute(self):
        if self.is_muted():
            self.unmute()
        else:
            self.mute()

    # Is sound happening?
    def is_playing(self):
        return self.channel is not None and (sound := self.channel.get_sound()) and sound.get_volume() > 0

    def clear_sound_queue(self):
        while not self.sound_queue.empty():
            self.sound_queue.get()

    def queue_sound(self, sample_slice: SampleSlice | None):
        if sample_slice is None:
            return
        sample_slice.start_time += self.oneshot_offset
        logger.debug(
            f"queued sound in {self.name} for step {sample_slice.step} {datetime.fromtimestamp(sample_slice.start_time)}")
        if self.recording:
            self.recorded_steps[i := sample_slice.step %
                                len(self.recorded_steps)] = sample_slice.sound
            logger.info(f"{self.name} recording sound for step {i}")
        sound_data[sample_slice.sound].step = sample_slice.step
        step_gate = self.gates[sample_slice.step % len(self.gates)]
        prev_step_gate = self.gates[(sample_slice.step - 1) % len(self.gates)]
        if self.step_repeating and step_gate == 0:
            step_gate = 0.5
        if step_gate > 0 and prev_step_gate == 1:
            sample_slice.sound.set_volume(self.volume.get(sample_slice.step))
        else:
            sample_slice.sound.set_volume(0)
        self.sound_queue.put(sample_slice)
        self.audio_executor.submit(
            sample_slice.apply_fx).add_done_callback(future_done)

    # call provided fn to create sound and add to queue
    def queue_async(self, generate_sound, t: float, step: int):
        logger.debug(f"{self.name} scheduling async sound for {t}")
        future = self.audio_executor.submit(
            lambda: self.queue_sound(SampleSlice(t, step, generate_sound())))
        future.add_done_callback(future_done)

    def queue_and_replace_async(self, generate_sound, t: float, step: int):
        logger.debug(f"{self.name} scheduling async sound for {t}")
        future = self.audio_executor.submit(
            self.set_and_queue_slice, step, t, generate_sound)
        future.add_done_callback(future_done)
        return future

    def replace_async(self, generate_sound, step: int):
        logger.debug(f"{self.name} scheduling async sound for {step}")
        future = self.audio_executor.submit(
            self.set_slice, step, generate_sound)
        future.add_done_callback(future_done)
        return future

    def warn_dropped(self, dropped: list[SampleSlice], now: float):
        if (n := len(dropped)) == 0:
            return
        dropped_sample = dropped[0]
        msg = f"{self.name} dropped {n}/{n+self.sound_queue.qsize()} samples stale by: {1000 * (now - dropped_sample.start_time - self.timeout):10.6}ms for step {dropped_sample.step}"
        msg += f" sched. {datetime.fromtimestamp(dropped_sample.start_time)}"
        logger.debug(msg)


    def unmute_active_intervals(self):
        for interval in self.unmute_intervals:
            if interval.contains(self.elapsed_sequence_time()):
                self.unmute(suppress_recording=True)
            elif not self.mute_override:
                self.mute(suppress_recording=True)

    def apply_gate(self, playing_sound: pygame.mixer.Sound):
        if self.channel and playing_sound in sound_data and sound_data[playing_sound].step is not None:
            start_volume = playing_sound.get_volume()
            playing_step = sound_data[playing_sound].step
            if playing_step is None:
                # Shouldnt happen, but this makes red squiggles happy
                return
            step_gate = self.gates[playing_step % len(self.gates)]
            if (inverted := step_gate < 0):
                step_gate *= -1
            if self.step_repeating and step_gate == 0:
                step_gate = 0.5
            gate_time = step_gate * playing_sound.get_length()
            time_playing = time.time() - sound_data[playing_sound].playtime
            time_fading = time_playing - gate_time
            prev_step_gate = self.gates[(playing_step - 1) % len(self.gates)]
            start_fade = step_gate > 0 and prev_step_gate != 1
            if step_gate != 0 and step_gate != 1 and time_playing > gate_time:
                ratio = max(0, min(1, time_fading / self.gate_fade))
                if not inverted:
                    ratio = 1 - ratio
                volume = ratio * self.volume.get(playing_step)
                if playing_sound.get_volume() != volume:
                    playing_sound.set_volume(volume)
                    logger.debug(
                        f"{self.name} volume to {volume}, {ratio}% faded")
            elif start_fade and playing_sound.get_volume() != self.volume.get(playing_step):
                ratio = min(1, time_playing / self.gate_fade)
                playing_sound.set_volume(
                    volume := self.volume.get(playing_step) * ratio)
                logger.debug(
                    f"{self.name} volume to {volume}, {ratio}% faded in")
            end_volume = playing_sound.get_volume()
            logger.debug(f"{self.name} channel vol={self.channel.get_volume()} changed volume from {start_volume} to {end_volume}")

    # returns callable to do the sound making
    def process_queue(self, now, step_duration) -> Callable | None:
        self.unmute_active_intervals()
        playing_sound: pygame.mixer.Sound | None = self.channel.get_sound() if self.channel else None
        if playing_sound:
            self.apply_gate(playing_sound)

        logger.debug(f"{self.name} start process queue")
        if self.sound_queue.empty():
            logger.debug(f"{self.name}: queue empty")
            return None

        qsound = self.sound_queue.get()

        dropped = []
        while now > qsound.start_time + self.timeout:
            dropped.append(qsound)
            if self.sound_queue.empty():
                self.warn_dropped(dropped, now)
                return None
            qsound = self.sound_queue.get()
        self.warn_dropped(dropped, now)

        in_play_window = now >= qsound.start_time - self.lookahead
        in_queue_window = now >= qsound.start_time - self.lookahead - step_duration
        if not in_queue_window:
            logger.debug(f"{self.name}: too early for queue her")
            self.sound_queue.put(qsound)
            return None
        if not in_play_window and self.channel is None:
            logger.debug(f"{self.name}: too early for channel her")
            self.sound_queue.put(qsound)
            return None

        if self.channel and not self.channel.get_busy() and not in_play_window:
            self.sound_queue.put(qsound)
            return None

        if not in_play_window and self.channel and self.channel.get_busy() and self.channel.get_queue() is not None:
            logger.debug(f"{self.name}: channel full")
            self.sound_queue.put(qsound)
            return None

        if self.channel and not self.channel.get_busy() and self.channel.get_queue() is not None:
            logger.warn(
                f"{self.name} weird state, ghost queue? let's try clear it")
            self.channel.stop()

        logger.debug(
            f"{self.name} processing {datetime.fromtimestamp(qsound.start_time)}")
        # todo: move this return past all the early "None" returns to facilitate consolidation in refactor
        if in_play_window:
            if self.channel and self.channel.get_busy() and playing_sound:
                logger.warn(
                    f"{self.name} interrupted sample with {remaining_time(playing_sound)}s left")
                logger.warn(
                    f"sample length {playing_sound.get_length()} vs step length {step_duration}")
            logger.debug(f"{self.name}: played sample")
            return self.step_player(qsound)
        if self.channel and self.channel.get_queue() is None:
            predicted_finish = time.time() + remaining_time(playing_sound)
            max_start_discrepancy = 0.015
            if not should_queue(self.name, qsound, predicted_finish, max_start_discrepancy):
                self.sound_queue.put(qsound)
                return None
            self.channel.queue(qsound.sound)
            sound_data[qsound.sound].playtime = predicted_finish
            logger.debug(
                f"{self.name}: queued sample on {self.channel} {qsound.t_string()} {qsound}")
            return None

        logger.info(f"{self.name} fell through, putting back on queue")
        self.sound_queue.put(qsound)
        return None

    def get_playing(self):
        if self.channel:
            return self.channel.get_sound()
        return None

    def step_player(self, sample_slice: SampleSlice):
        def fn():
            self.play_sound(sample_slice.sound)
            self.played_steps.append(sample_slice)
            return sample_slice.step, sample_slice.start_time
        return fn

    def play_sound(self, sound):
        if self.channel is None:
            self.play_sound_new_channel(sound)
        else:
            self.channel.play(sound)
            sound_data[sound].playtime = time.time()

    def play_sound_new_channel(self, sound):
        self.channel = sound.play()
        sound_data[sound].playtime = time.time()
        channels.add(self.channel)
        print(f"seen {(n := len(channels))} channels")
        pygame.mixer.set_reserved(n)

    def get_sound_slices(self):
        slices = [s if rec is None else rec for s, rec in zip(
            self.sound_slices, self.recorded_steps, strict=True)]
        for i in range(0, len(slices) - 2, 2):
            scatter_offset = (
                val := self.spices_param.scatter.value(0, i)) - val % 4
            scatter_offset %= len(slices)
            j = i + scatter_offset
            j %= len(slices)
            slices[j:j + 4] = slices[i:i + 4]
        return slices

    def get_step_sound(self, step, t, source_step=None, force=False) -> SampleSlice | None:
        # TODO LatchRepeat method to get SampleSlice
        if self.latch.active:
            sample_slice = self.latch.steps[0]
            # TODO could get rid of is_rotating? check step alignment instead
            if self.latch.is_rotating:
                self.latch.steps.rotate(-1)
            else:
                length = len(self.latch.steps)
                count = 0
                while step % length != sample_slice.step % length:
                    count += 1
                    if count > len(self.latch.steps):
                        break
                    self.latch.steps.rotate(-1)
                    sample_slice = self.latch.steps[0]
                self.latch.is_rotating = True
            return SampleSlice(t, step, sample_slice.sound)

        spice_factor = 2 if self.spices_param.stretch_chance.toss(
            step) else 1
        rate = self.get_rate() / spice_factor
        steps_per_slice = round(1 / rate)
        if step % steps_per_slice != 0 and not force:
            return None
        sound_slices = self.get_sound_slices()
        if source_step is None:
            source_step = step
        if self.scatter_queue is not None:
            source_step = self.scatter_queue
            self.scatter_queue = None

        sound = sound_slices[(source_step // steps_per_slice) %
                             len(sound_slices)]

        fx = []
        if rate != 1:
            fx.append(lambda sound: timestretch(
                sound, rate, sound_data, stretch_fade))
        if (pitch := self.pitch.get(step)):
            logger.debug(f"{self.name} setting pitch to {pitch}")
            fx.append(lambda sound: self.change_pitch(
                source_step, sound, pitch))
        if (sound_bpm := sound_data[sound].bpm) != self.bpm:
            logger.info(
                f"{self.name} step {step} stretching sample from {sound_bpm} to {self.bpm}")
            future = self.queue_and_replace_async(lambda: timestretch(
                self.source_sound(sound), self.bpm / sound_bpm, sound_data), t, step)
            future.add_done_callback(
                lambda f: set_sound_bpm(f.result(), self.bpm))

        return SampleSlice(t, step, sound, fx)

    def queue_step_repeat_steps(self, step: int, step_time: float, step_interval: float):
        self.step_repeating = True
        self.clear_sound_queue()
        # TODO could save work here, only process fx once for each loop slice
        subslices = [range(self.step_repeat_index, self.step_repeat_index + length)
                     for length in self.step_repeat_lengths]
        all_slices = []
        for subslice in subslices:
            all_slices.extend(subslice)
        logger.info(
            f"{self.name} has {len(all_slices)} step repeat slices for sr length {self.step_repeat_length}, index {self.step_repeat_index}")
        for i, substep in enumerate(all_slices):
            spice_factor = 2 if self.spices_param.stretch_chance.toss(
                step) else 1
            rate = self.get_rate() / spice_factor
            ts = step_time + i * step_interval / rate
            for_step = step + i
            qs = self.get_step_sound(for_step, ts, source_step=substep)
            logger.info(
                f"step repeat queueing {substep} for step {for_step} {qs.t_string() if qs else '?'}")
            self.queue_sound(qs)

    def queue_step(self, step: int, step_time: float, step_interval: float):
        if step == 0:
            self.seq_start = step_time
        step = (step - self.oneshot_start_step) % 64
        srlength = round(self.step_repeat_length / self.get_rate())
        do_step_repeat = self.step_repeat and (
            self.looping or not self.is_muted())
        if step % 2 == 1 and self.spices_param.stretch_chance.toss(step - 1 % self.slices_per_loop):
            # last step is stretched 2x, skip this one to give it time to finish
            return
        if do_step_repeat and step in range(self.step_repeat_index % srlength, self.slices_per_loop, srlength):
            self.queue_step_repeat_steps(step, step_time, step_interval)

        should_be_queued = (not self.is_muted()
                            or self.looping or self.step_repeat)
        if not self.step_repeating and should_be_queued:
            self.queue_sound(self.get_step_sound(step, step_time))

    @staticmethod
    def source_sound(sound: pygame.mixer.Sound):
        logger.info(f"getting source for {sound}")
        if sound not in sound_data or (src := sound_data[sound].source) is None:
            return sound
        return Sample.source_sound(src)

    def change_pitch(self, step: float, sound: pygame.mixer.Sound, semitones: int):
        if semitones == 0:
            return sound
        logger.info(f"{self.name}: step {step} by {semitones} semitones")
        pitched_sound = pitch_shift(sound, semitones, sound_data)
        sound_data[pitched_sound].bpm = sound_data[sound].bpm
        sound_data[pitched_sound].semitones = semitones
        return pitched_sound

    @staticmethod
    def modulate(param, period: int, shape, amount, steps=None):
        lfo = modulation.Lfo(period, shape)
        param.modulate(lfo, amount, steps)


def should_queue(name, qsound, predicted_finish, max_start_discrepancy) -> bool:
    if (error := predicted_finish - qsound.start_time) > max_start_discrepancy:
        logger.debug(
            f"{name} queueing sample would make it late by {error}, putting back on queue")
        return False
    if error < -1 * max_start_discrepancy:
        logger.debug(
            f"{name} queueing sample would make it early by {-error}, putting back on queue")
        return False
    if len(qsound.fx) > 0:
        logger.info(
            f"{name} not queueing yet because fx haven't finished applying")
        return False
    return True


def set_sound_bpm(sound, bpm):
    sound_data[sound].bpm = bpm


def future_done(f):
    if (e := f.exception()):
        raise e


def play_samples(step_duration):
    logger.debug("playing samples")
    now = time.time()
    play_hooks = [s.process_queue(now, step_duration)
                  for s in loaded_samples]
    played_steps = [hook() if hook else hook for hook in play_hooks]
    if any(played_steps):
        logger.debug(played_steps_string(played_steps))


def played_steps_string(played_steps):
    s = "\n" * 10 + "============\n"
    for nt in played_steps:
        if nt:
            n, t = nt
            s += f"{datetime.fromtimestamp(t)} - {n}\n"
        else:
            s += "--\n"
            s += "============\n"
    return s


def queues_empty():
    return all([s.sound_queue.qsize() == 0] for s in loaded_samples)


def queue_samples(step, t, step_duration):
    for sample in loaded_samples:
        sample.queue_step(step, t, step_duration)


def queues_to_string():
    s = "\n============\n"
    for sample in loaded_samples:
        s += " " + "o" * sample.sound_queue.qsize() + "\n"
        s += "============\n"
    return s


def step_repeat_stop(length):
    for sample in [s for s in loaded_samples if length in s.step_repeat_lengths]:
        sample.step_repeat_stop(length)


def stop_halftime():
    for s in loaded_samples:
        s.stop_stretch()


def stretch_samples(target_bpm):
    for s in all_samples():
        s.bpm = target_bpm
