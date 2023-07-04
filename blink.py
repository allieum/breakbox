import array
import utility
import time

from dmx import DMXInterface

logger = utility.get_logger(__name__)

class Light:
    start_channel = 12
    brightness = 255

    def __init__(self):
        self.color = [0, 0, 0, 0, 0, 0]

    def absorb(self, color):
        for j, value in enumerate(color):
            self.color[j] = (self.color[j] + value) // 2


class LightManager:
    start_channel = 12
    def __init__(self):
        self.patterns = []
        self.lights = [Light() for i in range(18)]
        self.brightness = 255
        self.dmx_interface = None
        try:
            self.dmx_interface = DMXInterface()
        except:
            pass

    def add_pattern(self, pattern):
        self.patterns.append(pattern)
    
    def step(self, step):
        for pattern in self.patterns:
            pattern.update()

            for i, light in self.lights:
                light.absorb(pattern.lights[i].color)
    
    def send_dmx(self):
        if not self.dmx_interface:
            return

        self.dmx_interface.set_frame(self.get_light_data())
        now = time.time()
        self.dmx_interface.send_update()
        logger.debug(f"dmx frame send took {time.time() - now}s")

    def get_light_data(self):
        data = array.array('B', [0] * 119)
        data[0] = self.brightness
        for i, light in self.lights:
            start_idx = self.start_channel - 1
            light_idx = i * 6
            for color_idx, color in light.color:
                data[start_idx + light_idx + color_idx] = color
        return list(data)


