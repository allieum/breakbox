from dataclasses import dataclass, field
import dataclasses
from multiprocessing import Queue
import time
import board
import adafruit_ws2801

from sample import Sample, sound_data
import utility

logger = utility.get_logger(__name__)

# odata = board.D5
# oclock = board.D6
# numleds = 25
# bright = 1.0
# leds = adafruit_ws2801.WS2801(oclock, odata, numleds, brightness=bright, auto_write=False)

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
    #ff00c1,
    #9600ff,
    #4900ff,
    #00b8ff,
    #00fff9,
    #00ff85,
# TODO colors module
palette = [
    #4deeea,
    0x4deeea,
    #74ee15,
    0x74ee15,
    #ffe700,
    0xffe700,
    #f000ff,
    0xf000ff,
    #001eff,
    0x001eff,
    #FF7A53,
    0xFF7A53,
]
OFF = 0x0

def average(a, b, b_weight=0.5):
    a_weight = 1 - b_weight
    return round(a * a_weight + b * b_weight)

def to_tuple(color):
    if type(color) is tuple:
        return color
    return (
        (color & 0xff0000) >> 4 * 4,
        (color & 0x00ff00) >> 2 * 4,
        (color & 0x0000ff) >> 0
    )

@dataclass
class SampleState:
    playing: bool
    bank: int
    step: int | None = field(compare=False, default=None)

    @staticmethod
    def of(sample: Sample):
        state = SampleState(sample.is_playing(), sample.bank)
        if sample.channel and (playing := sample.channel.get_sound()) in sound_data:
            state.step = sound_data[playing].step
        return state

@dataclass
class LedState:
    i: int = field(compare=False)
    leds: adafruit_ws2801.WS2801 = field(compare=False)
    color: tuple[int, int, int] = field(default=(0, 0, 0))
    fade_goal: tuple[int, int, int] | None = field(compare=False, default=None)

    def update(self):
        color = self.color
        if self.fade_goal:
            self.mix(self.fade_goal)
        if self.fade_goal == self.color:
            self.fade_goal = None
        # if changed := self.color != color:
        #     self.leds[self.i] = self.color
        # return changed
    def write(self):
        # logger.info(f"{self.i} setting to {self.color}")
        self.leds[self.i] = self.color

    def fade(self, color):
        self.fade_goal = to_tuple(color)

    def mix(self, color):
        color = to_tuple(color)
        self.color = tuple(map(lambda ab: average(*ab), zip(self.color, color)))

def init():
    # TODO conditional import / init
    pass
    # leds.fill(0)
    # leds.show()

# def refresh_ready(samples_on):
#     # return time.time() - last_update > REFRESH_INTERVAL and samples_on != last_samples and not refreshing
#     # too_soon = time.time() - last_update < REFRESH_INTERVAL
#     return samples_on != last_state

def run(lights_q: Queue):
    # global last_update, last_state, refreshing
    logger.info(f"lights baby")
    odata = board.D5
    oclock = board.D6
    numleds = 7
    bright = 0.75
    leds = adafruit_ws2801.WS2801(oclock, odata, numleds, brightness=bright, auto_write=False)
    leds.fill(0)
    leds.show()
    sample_lights_offset = 1
    led_states = [LedState(j + sample_lights_offset, leds) for j in range(numleds - 1)]
    min_refresh = 2
    max_refresh = 0.025
    last_update = time.time()
    prev_samples = None
    while True:
        # make a copy
        last_state = list(map(dataclasses.replace, led_states))
        sample_states = lights_q.get()
        logger.debug(f"got {sample_states} from queue")
        for sample, led in zip(sample_states, led_states):
            led.fade(palette[sample.bank] if sample.playing else OFF)
            led.update()
        elapsed = time.time() - last_update
        updating = led_states != last_state and elapsed > max_refresh
        if updating or elapsed > min_refresh:
            last_update = time.time()
            for led in led_states:
                led.write()
            leds.show()
            # logger.info(f"updating lights")
        else:
            led_states = last_state
            # logger.info(f"updating lights to {led_states}")
        if prev_samples != sample_states:
            prev_samples = sample_states
            logger.info(f"sample state changed")
        # if updating:
        #     logger.info(f"updating lights")
        time.sleep(0.005)

# last_update = time.time()
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
