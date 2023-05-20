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
        # # match shape:
        #     case self.Shape.TRIANGLE:
        #         self.fn = triangle(period)
        #     case _:
        #         logger.error(f"unknown LFO shape {shape}")

    def step(self):
        self.steps += (self.steps + 1) % self.period

    def value(self):
        return self.fn(self.steps)

class Param:
    def __init__(self, value) -> None:
        self.value = value

    def modulate(self, lfo, amount):
        self.lfo = lfo
        self.scale = amount

    def get(self):
        value = self.value
        if self.lfo:
            value += round(self.lfo.value() * self.scale)
        return value

def triangle(period):
    return lambda x: x / x2 if x <= (x2 := period / 2) else 1 - (x - x2) / x2

def sin(period):
    pass
