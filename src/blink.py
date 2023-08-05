import array
import time

import sample
import sequence
import utility

import midi

logger = utility.get_logger(__name__)


class Light:
    dmx_frame = array.array('B', [0] * 119)
    start_channel = 12
    brightness = 255

    def __init__(self, i):
        self.i = i
        self.start_index = self.start_channel - 1 + self.i * 6
        self.set_brightness(self.brightness)

    def set_color(self, color):
        for j, value in enumerate(color):
            self.dmx_frame[self.start_index + j] = value

    def absorb(self, color):
        for j, value in enumerate(color):
            new_value = value
            old = self.dmx_frame[self.start_index + j]
            if old == round(0.8 * value) and old != 0:
                logger.info(
                    f"setting light {self.i} to same color, toggling off")
                new_value = 0
            elif value != 0:
                new_value = (old + value) // 2
            self.dmx_frame[self.start_index + j] = new_value

    @staticmethod
    def scale(factor):
        for i in range(Light.start_channel - 1, len(Light.dmx_frame)):
            Light.dmx_frame[i] = round(Light.dmx_frame[i] * factor)

    @staticmethod
    def all_off():
        Light.dmx_frame[Light.start_channel -
                        1:] = array.array('B', [0] * (119 - (Light.start_channel - 1)))

    @staticmethod
    def set_brightness(level):
        Light.dmx_frame[0] = level

    @staticmethod
    def send_frame(interface):
        interface.set_frame(list(Light.dmx_frame))
        interface.send_update()


lights = [Light(i) for i in range(18)]


def bounce(step):
    x = step % 32
    if x < 16:  # noqa: PLR2004
        return x + 1
    return 16 - (x % 16)


def bounce_lights(step):
    # tri = modulation.triangle(15)
    # light = round(tri(step % 16) * 7)
    light = bounce(step)
    logger.debug(f"light {light}")
    return (light,)


def lights_for_step(step):
    light_index = step % 8 + 1 + 8
    mirror_index = -(light_index - 8) + 8 + 1
    return (light_index, mirror_index)


dmx_interface = None
# try:
#     dmx_interface = DMXInterface()
# except:
#     pass


def update_dmx(step, note_number=None):
    if dmx_interface is None:
        return
    # last_dmx = now
    # last_dmx_step = sequence.step
    logger.debug(f"lighting dmx step {step}")
    color = [0, 0, 0, 0, 0, 0]
    time.sleep(0.020)

    Light.scale(0.8)

    for i, s in enumerate(sample.current_samples()):
        if s.channel and s.channel.get_busy():
            source_step = sample.sound_data[s.channel.get_sound()].source_step
            if source_step != sequence.step:
                logger.debug(f"source step {source_step}")
            color[i] = 255
            for j in bounce_lights(source_step):
                lights[(j + i * 3) % len(lights)].absorb(color)
        # lights[i].absorb([0,0,0,0,0,0])

    for note_number in midi.note_q:
        if note_number == 0:
            for light in lights:
                # logger.info(f"bass flash {step} {note_number}")
                light.absorb([255, 0, 0, 255, 255, 255])
    midi.note_q.clear()

    # light_index = sequence.step % 8 + 1 + 8
    # mirror_index = -(light_index - 8) + 8 + 1
    # if any(color):
    #     lights[light_index].set(color)
    #     lights[mirror_index].set(color)
    dmx_interface.set_frame(list(Light.data))
    time.time()
    dmx_interface.send_update()
