from enum import Enum

import utility

logger = utility.get_logger(__name__)

class Lfo:
    class Shape(Enum):
        TRIANGLE = 1
        SIN = 2
        SQUARE = 3
        SAW = 4

    def __init__(self, period, shape):
        self.period = period
        self.shape = shape
        self.steps = 0
        self.enabled = True
        match shape:
            case self.Shape.TRIANGLE:
                self.fn = triangle(period)
            case _:
                logger.error(f"unknown LFO shape {shape}")

    def step(self):
        self.steps = (self.steps + 1) % self.period

    def value(self, x = None):
        if x is None:
            x = self.steps
        return self.fn(x)

class Param:
    def __init__(self, value) -> None:
        self.value = value
        self.lfo = None
        self.start_step = None
        self.steps = None

    def modulate(self, lfo, amount, steps=None):
        logger.info(f"modulating param with {lfo} x {amount}")
        self.lfo = lfo
        self.scale = amount
        self.steps = steps
        self.start_step = None

    def get(self, step):
        value = self.value
        if self.steps is not None:
            self.steps -= 1
        if self.lfo and self.lfo.enabled:
            if self.start_step is None:
                self.start_step = step
            lfo_step = (step - self.start_step) % self.lfo.period
            value += round(self.lfo.value(lfo_step) * self.scale)
            logger.info(f"LFO {self.lfo} value {value} step {step} lfo_step {lfo_step} start_step {self.start_step}")
        if self.steps == 0:
             self.lfo.enabled = False
        return value

def triangle(period):
    return lambda x: x / x2 if x <= (x2 := period / 2) else 1 - (x - x2) / x2

for i in range(8):
    print(round(triangle(8)(i) * 4))


def sin(period):
    pass