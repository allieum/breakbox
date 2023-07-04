import utility

from light_pattern import LightPattern
import sample
from sequence import sequence

logger = utility.get_logger(__name__)

class BouncePattern(LightPattern):
    def __init__(self, color):
      self.color = color
      pass

    def done(self):
       pass

    def update(self):
        self.scale(0.8)

        for i, s in enumerate(sample.current_samples()):
            if s.channel and s.channel.get_busy():
                source_step = sample.sound_data[s.channel.get_sound()].source_step
                if source_step != sequence.step:
                    logger.debug(f"source step {source_step}")
                for j in self.bounce_lights(source_step):
                    self.lights[j].absorb(self.color)
    
    def bounce(step):
      x = step % 32
      if x < 16:
          return x + 1
      return (16 - (x % 16),)