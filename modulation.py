from enum import Enum

# import control
import utility

logger = utility.get_logger(__name__)

class Lfo:
    class Shape(Enum):
        TRIANGLE = 1
        SIN = 2
        SQUARE = 3
        SAW = 4
        SAW_DESC = 5
        INC = 6
        DEC = 7

    def __init__(self, period, shape):
        self.period = period
        self.shape = shape
        self.steps = 0
        self.enabled = True
        match shape:
            case self.Shape.TRIANGLE:
                self.fn = triangle(period)
            case self.Shape.SAW:
                self.fn = saw(period)
            case self.Shape.SAW_DESC:
                self.fn = lambda x: -saw(period)(x)
            case self.Shape.INC:
                self.fn = inc(period)
            case self.Shape.DEC:
                self.fn = dec(period)
            case _:
                logger.error(f"unknown LFO shape {shape}")

    def step(self):
        self.steps = (self.steps + 1) % self.period

    def value(self, x = None):
        if x is None:
            x = self.steps
        return self.fn(x)

class Param:
    def __init__(self, value, min_value=None, max_value=None) -> None:
        self.value = value
        self.lfo = None
        self.start_step = None
        self.steps = None
        self.encoder = None
        self.encoder_scale = None
        self.encoder_prev = None
        self.scale = None
        self.on_change = None
        self.last_value = None
        self.min_value = min_value
        self.max_value = max_value

    def modulate(self, lfo, amount, steps=None):
        logger.info(f"modulating param with {lfo} x {amount}")
        self.lfo = lfo
        self.scale = amount
        self.steps = steps
        self.start_step = None
        if self.last_value is not None:
            self.value = self.last_value
        return self

    def control(self, encoder, scale, on_change=None):
        self.encoder = encoder
        self.encoder_scale = scale
        self.encoder_prev = encoder.value()
        self.on_change = on_change
        return self

    def get(self, step = -1):
        if step == -1:
            if self.encoder:
                delta = self.encoder_scale * (self.encoder.value() - self.encoder_prev)
                self.value += delta
                self.encoder_prev = self.encoder.value()
                if delta != 0 and self.on_change is not None:
                    self.on_change(self.value)
        
        value = self.value
        if self.steps is not None:
            self.steps -= 1
        if self.lfo and self.lfo.enabled:
            if self.start_step is None:
                self.start_step = step
            lfo_step = (step - self.start_step) % self.lfo.period
            value += round(self.lfo.value(lfo_step) * self.scale)
            logger.debug(f"LFO {self.lfo} value {value} step {step} lfo_step {lfo_step} start_step {self.start_step}")
        if self.steps == 0:
            self.lfo.enabled = False
        if self.lfo and not self.lfo.enabled and self.last_value is not None:
            value = self.last_value
        if self.min_value is not None:
            value = max(value, self.min_value)
        if self.max_value is not None:
            value = min(value, self.max_value)
        self.last_value = value
        return value

class Counter:
    def __init__(self, value, delta=1):
        self.value = value
        self.delta = delta

    def next(self):
        value = self.value
        self.value += self.delta
        logger.info(f"counter next value {value}, (delta {self.delta})")
        return value

    
def inc(_):
    counter = Counter(0)
    return lambda _: counter.next()

def dec(_):
    counter = Counter(0, -1)
    return lambda _: counter.next()

def saw(period):
    return lambda x: x % period

def triangle(period):
    return lambda x: x / x2 if x <= (x2 := period / 2) else 1 - (x - x2) / x2
