import dataclasses
import math
import time
from dataclasses import dataclass, field
from multiprocessing import Queue

import adafruit_ws2801
import board
import utility
from sample import SampleState

logger = utility.get_logger(__name__)

q = Queue(1)

AMARANTH = 0x9f2b68
AMARANTH = (0x9f, 0x2b, 0x68)
CERISE = 0xde3163
TEAL = 0xacddde
PURPLE = (138, 43, 226)

# palette = [
#     AMARANTH,
#     BISQUE,
#     CERISE,
#     TEAL,
#     MINT,
#     PURPLE,
# ]

palette = [
    0xff00c1,
    0x9600ff,
    0x4900ff,
    0x00b8ff,
    0x00fff9,
    0x00ff85,
]
# ff00c1,
# 9600ff,
# 4900ff,
# 00b8ff,
# 00fff9,
# 00ff85,
# TODO colors module
palette = [
    # 4deeea,
    0x4deeea,
    # 74ee15,
    0x74ee15,
    # ffe700,
    0xffe700,
    # f000ff,
    0xf000ff,
    # 001eff,
    0x001eff,
    # FF7A53,
    0xFF7A53,
]
OFF = 0x0


def average(a, b, b_weight=0.5):
    a_weight = 1 - b_weight
    return math.floor(a * a_weight + b * b_weight)


def to_tuple(color):
    if type(color) is tuple:
        return color
    return (
        (color & 0xff0000) >> 4 * 4,
        (color & 0x00ff00) >> 2 * 4,
        (color & 0x0000ff) >> 0,
    )

# @dataclass
# class SampleState:
#     playing: bool = field(default=False)
#     bank: int = field(default=0)
#     length: float = field(default=0, compare=False)
#     steps: int = field(default=0, compare=False)
#     selected: bool = field(default=False)
#     recording: bool = field(default=False)
#     step: int | None = field(compare=False, default=None)
#     pad: int = field(default=0)

#     @staticmethod
#     def of(sample: Sample, selected_sample: Sample, step, pad):
#         if sample is None:
#             return None

#         selected = selected_sample == sample
#         # if selected:
#         #     logger.info(f"{selected_sample} vs {sample} {selected}")
#         length = sum(map(lambda s: s.get_length(), sample.get_sound_slices()))
#         length = sample.sound.get_length()
#         steps = len(sample.sound_slices)
#         if sample.step_repeating:
#             length *= sample.step_repeat_length / len(sample.sound_slices)
#             steps = sample.step_repeat_length
#         progress = (step % steps) / steps
#         length *= (1 - progress)
#         length -= 0.5
#         state = SampleState(sample.is_playing(), sample.bank, length, steps, selected, sample.recording, step, pad)
#         # grab step from sequencer
#         # if sample.channel and (playing := sample.channel.get_sound()) in sound_data:
#         #     state.step = sound_data[playing].step
#         # state.step = step
#         return state


@dataclass
class LedState:
    i: int = field(compare=False)
    leds: adafruit_ws2801.WS2801 = field(compare=False)
    color: tuple[int, int, int] = field(default=(0, 0, 0))
    fade_goals: list[tuple[tuple[int, int, int], float, float, float]] = field(
        compare=False, default_factory=list)

    def update(self):
        for item in self.fade_goals:
            fade_color, start, duration, strength = item
            if time.time() - start > duration:
                self.fade_goals.remove(item)
            self.mix(fade_color, strength)
        self.mix(0x0, 0.08)

    def write(self):
        logger.debug(f"{self.i} setting to {self.color}")
        self.leds[self.i] = self.color

    def fade(self, color, duration, strength=0.25):
        self.fade_goals.append(
            (to_tuple(color), time.time(), duration, strength))

    def mix(self, color, strength=0.25):
        color = to_tuple(color)
        self.color = tuple(average(*ab, strength)
                           for ab in zip(self.color, color, strict=True))


def run(lights_q: Queue):
    # global last_update, last_state, refreshing
    logger.info(f"lights baby")
    # odata = board.D5
    # oclock = board.D6
    odata = board.MOSI
    oclock = board.SCK
    numleds = 6
    bright = 0.5
    try:
        leds = adafruit_ws2801.WS2801(
            oclock, odata, numleds, brightness=bright, auto_write=False)
    except:
        logger.info("failed to initialize lights")
        return
    leds.fill(0)
    leds.show()
    sample_lights_offset = 0
    led_states = [LedState(j + sample_lights_offset, leds)
                  for j in range(numleds)]
    min_refresh = 2
    max_refresh = 0.010
    last_update = time.time()
    sample_states: list[SampleState] = [SampleState()] * 6
    while True:
        # make a copy
        prev_samples = list(map(dataclasses.replace, sample_states))
        last_state = list(map(dataclasses.replace, led_states))
        sample_states = lights_q.get()
        logger.debug(f"got {sample_states} from queue")
        bank_changed = any(((new_bank := sample.bank) != prev.bank for sample, prev in zip(
            sample_states, prev_samples, strict=True)))
        for sample, led, prev in zip(sample_states, led_states, prev_samples, strict=True):
            # if sample.selected and prev.step != sample.step and sample.step % 4 == 0:
            #     color = 0xff0000 if sample.recording else 0x00ff00
            #     led.fade(color, 0.3, 0.5)
            # pulse on sample loop
            # strength = 0.1 if sample.selected else 0.3
            strength = 0.3
            first_step = sample.step is not None and sample.step % sample.steps == 0
            new_step = prev.step != sample.step
            if sample.playing and new_step and first_step:
                led.fade(palette[sample.bank % len(palette)],
                         sample.length, strength)
            elif sample.playing and not prev.playing:
                led.fade(palette[sample.bank % len(palette)],
                         sample.length, strength)
            if not sample.playing and prev.playing:
                led.fade_goals.clear()
                led.fade(OFF, 1)
            # if not sample.selected and not sample.recording:
            if bank_changed and led.i != new_bank % 6:
                led.fade(palette[new_bank % len(palette)], 2)
            led.update()

        elapsed = time.time() - last_update
        updating = led_states != last_state and elapsed > max_refresh
        if updating or elapsed > min_refresh:
            last_update = time.time()
            for led in led_states:
                led.write()
            leds.show()
            # logger.info(f"{selected_sample} vs ")
            # light_str = ["on" if led.color != OFF else "off" for led in led_states]
            # logger.debug(f"updating lights {light_str}")
        else:
            led_states = last_state
            sample_states = prev_samples
            # logger.info(f"updating lights to {led_states}")
        # if prev_samples != sample_states:
        #     prev_samples = sample_states
        #     logger.info(f"sample state changed")
        # if updating:
        #     logger.info(f"updating lights")
        time.sleep(0.005)

# last_state = None
# refreshing = False
# REFRESH_INTERVAL = 0.010
# def update(samples_on):
#     global last_update, last_state, refreshing
#     # if (now := time.time()) - last_update < REFRESH_INTERVAL or samples_on == last_samples:
#     #     return
#     # logger.info(f"updating sample status leds {samples_on} vs {last_samples}")


#     sample_lights_offset = 1
#     palette = [AMARANTH] * 6
#     colors = [palette[i] if sample_on else (0,0,0) for i, sample_on in enumerate(samples_on)]
#     for i in range(len(samples_on)):
#         leds[sample_lights_offset + i] = colors[i]
#     leds.show()
#     for i in range(len(samples_on)):
#         if leds[sample_lights_offset + i] != colors[i]:
#             logger.info(f"tried to set {colors[i]} but light is {leds[sample_lights_offset + i]}")
#     last_update = time.time()
#     last_state = samples_on


# def refresh_done():
#     global refreshing
#     refreshing = False
#     logger.debug("finished refresh")
