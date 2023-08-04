import array

import utility

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
