import time
import board
import adafruit_ws2801

import utility

logger = utility.get_logger(__name__)

odata = board.D5
oclock = board.D6
numleds = 25
bright = 1.0
leds = adafruit_ws2801.WS2801(oclock, odata, numleds, brightness=bright, auto_write=False)

AMARANTH = 0x9f2b68
AMARANTH = (0x9f, 0x2b, 0x68)
BISQUE = 0xF2D2BD
CERISE = 0xde3163

def init():
    # TODO conditional import / init
    leds.fill(0)
    leds.show()

def refresh_ready(samples_on):
    # return time.time() - last_update > REFRESH_INTERVAL and samples_on != last_samples and not refreshing
    # too_soon = time.time() - last_update < REFRESH_INTERVAL
    return samples_on != last_samples

last_update = time.time()
last_samples = None
refreshing = False
REFRESH_INTERVAL = 0.010
def update(samples_on):
    global last_update, last_samples, refreshing
    # if (now := time.time()) - last_update < REFRESH_INTERVAL or samples_on == last_samples:
    #     return
    # logger.info(f"updating sample status leds {samples_on} vs {last_samples}")


    sample_lights_offset = 1
    palette = [AMARANTH] * 6
    colors = [palette[i] if sample_on else (0,0,0) for i, sample_on in enumerate(samples_on)]
    for i in range(len(samples_on)):
        leds[sample_lights_offset + i] = colors[i]
    leds.show()
    for i in range(len(samples_on)):
        if leds[sample_lights_offset + i] != colors[i]:
            logger.info(f"tried to set {colors[i]} but light is {leds[sample_lights_offset + i]}")
    last_update = time.time()
    last_samples = samples_on


def refresh_done():
    global refreshing
    refreshing = False
    logger.debug("finished refresh")
