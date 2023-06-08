import array
import utility

logger = utility.get_logger(__name__)

class Light:
    data = array.array('B', [0] * 119)
    start_channel = 12
    brightness = 255

    def __init__(self, i):
        self.i = i
        self.start_index = self.start_channel - 1 + self.i * 6
        self.set_brightness(self.brightness)

    def set(self, color):
        for j, value in enumerate(color):
            self.data[self.start_index + j] = value

    def absorb(self, color):
        for j, value in enumerate(color):
            old = self.data[self.start_index + j]
            if old == round(0.8 * value) and old != 0:
                logger.info(f"setting light {self.i} to same color, toggling off")
                value = 0
            elif value != 0:
                value = (old + value) // 2
            self.data[self.start_index + j] = value

    @staticmethod
    def scale(factor):
        for i in range(Light.start_channel - 1, len(Light.data)):
            Light.data[i] = round(Light.data[i] * factor)

    @staticmethod
    def all_off():
        Light.data[Light.start_channel - 1:] = array.array('B', [0] * (119 - (Light.start_channel - 1)))

    @staticmethod
    def set_brightness(level):
       Light.data[0] = level

    @staticmethod
    def send_frame(interface):
        interface.set_frame(list(Light.data))
        interface.send_update()

lights = [Light(i) for i in range(18)]
