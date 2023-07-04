import blink
import array

# Patterns:

# Alternate
# XXX000XXX000XXX000
# 000XXX000XXX000XXX

# XXXXXXXXX000000000
# 000000000XXXXXXXXX

# Chase
# XXX000XXX000XXX000
# 0XXX000XXX000XXX00
# 00XXX000XXX000XXX0
# For effect, direction change or different amounts of lights:
# X000X000X000X000X0
# 0X000X000X000X000X
# 00X000X000X000X000
# X0XXX0XXX0XXX0XXX0

# Level
# 00000000XX00000000
# 000000XXXXX0000000
# 0000XXXXXXXXXX0000
# 000000XXXXX0000000
# or
# X0000000000000000X
# XX00000000000000XX
# XXX000000000000XXX



class LightPattern:
    def __init__(self):
        self.lights = [blink.Light(i) for i in range(18)]
        pass
    
    def step(self, step):
        pass

    # utilities
    def scale(self, factor):
        for light in self.lights:
          for i in range(light.start_channel - 1, len(light.data)):
              light.data[i] = round(light.data[i] * factor)
    
    def all_off(self):
        for light in self.lights:
          light.data[light.start_channel - 1:] = array.array('B', [0] * (119 - (light.start_channel - 1)))