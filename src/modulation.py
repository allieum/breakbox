from dataclasses import dataclass, field
from enum import Enum
from threading import Timer
import time
from typing import Any

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

    def value(self, x=None):
        if x is None:
            x = self.steps
        return self.fn(x)


class Param:
    def __init__(self, value, min_value=None, max_value=None, round=False) -> None:
        self.gradient_timer = None
        self.value = value
        self.default_value = value
        self.lfo: None | Lfo = None
        self.start_step = None
        self.steps = None
        self.encoder = None
        self.encoder_scale = None
        self.encoder_prev = None
        self.scale = None
        self.on_change = []
        self.last_value = None
        self.min_value = min_value
        self.max_value = max_value
        self.round = round
        self.spice_params: None | SpiceParams = None

    def restore_default(self):
        self.value = self.default_value
        self.last_value = None

    def spice(self, spice_params):
        self.spice_params = spice_params
        return self

    def modulate(self, lfo, amount, steps=None):
        logger.info(f"modulating param with {lfo} x {amount}")
        self.lfo = lfo
        self.scale = amount
        self.steps = steps
        self.start_step = None
        if self.last_value is not None:
            self.value = self.last_value
        return self

    def mod_cancel(self, restore=True):
        if self.lfo:
            self.lfo.enabled = False
        if restore:
            self.restore_default()

    def control(self, encoder, scale, on_change=None):
        self.encoder = encoder
        self.encoder_scale = scale

        self.encoder_prev = encoder.value()
        self.add_change_handler(on_change)
        return self

    def add_change_handler(self, handler):
        self.on_change.append(handler)

    def get(self, step=-1):
        if step == -1 and self.encoder:
            delta = self.encoder_scale * \
                (self.encoder.value() - self.encoder_prev)
            self.value += delta
            self.encoder_prev = self.encoder.value()
        value = self.value
        if self.steps is not None:
            self.steps -= 1
        if self.lfo and self.lfo.enabled:
            if self.start_step is None:
                self.start_step = step
            lfo_step = (step - self.start_step) % self.lfo.period
            value += self.lfo.value(lfo_step) * self.scale
            if self.round:
                value = round(value)
            logger.debug(
                f"LFO {self.lfo} value {value} step {step} lfo_step {lfo_step} start_step {self.start_step}")
        if self.steps == 0 and self.lfo:
            self.lfo.enabled = False
        if self.lfo and not self.lfo.enabled and self.last_value is not None:
            value = self.last_value
        if self.min_value is not None:
            value = max(value, self.min_value)
        if self.max_value is not None:
            value = min(value, self.max_value)
        if self.round:
            value = round(value)
        if value != self.last_value:
            for handler in self.on_change:
                handler(value)
        self.last_value = value
        if self.spice_params is not None and step >= 0:
            value = self.spice_params.value(value, step)
            if self.round:
                value = round(value)
        return value

    def set(self, value=None, delta=None):
        if (value is None and delta is None) or (value is not None and delta is not None):
            logger.error(f"must specify just 1 of value and delta")
            return None
        if delta is None:
            delta = 0
        if value is None:
            value = self.value + delta
        if self.min_value is not None and value is not None:
            value = max(value, self.min_value)
        if self.max_value is not None and value is not None:
            value = min(value, self.max_value)
        if (changed := value != self.value):
            for handler in self.on_change:
                handler(value)
        self.value = value
        return changed

    def set_gradient(self, start_value: float | int, end_value: float | int, duration: float):
        self.set(start_value)
        update_rate = 0.1
        if self.gradient_timer is not None:
            self.gradient_timer.cancel()
        gradient_delta = end_value - start_value
        update_delta = gradient_delta * 0.1 / duration
        self.gradient_timer = Timer(update_rate, self.set_gradient,
                                    [start_value + update_delta, end_value, duration - update_rate])
        self.gradient_timer.start()

    def normalize(self, value, scale=1.0):
        if self.max_value is None or self.min_value is None:
            logger.error(
                f"tried to normalize param with unbounded range: default {self.default_value}, [{self.min_value}, {self.max_value}]")
            return value
        range = self.max_value - self.min_value
        norm = value - self.min_value
        ratio = norm / range
        return ratio * scale


@dataclass
class SpiceParams:
    max_chance: float
    max_delta: float | int
    spice: Param
    step_data: None | list
    integer: bool = field(default=False)

    def toss(self, step):
        if self.step_data is None:
            return False
        if step > len(self.step_data):
            # TODO why this happen
            logger.debug(f"step {step} and step_data {self.step_data}")
        _, step_chance = self.step_data[step % len(self.step_data)]
        chance = self.spice.value * self.max_chance
        return step_chance < chance

    def value(self, original, step):
        if self.step_data is None:
            return original
        if step > len(self.step_data):
            logger.debug(f"step {step} and step_data {self.step_data}")
        step_intensity, step_chance = self.step_data[step % len(
            self.step_data)]
        val = original
        chance = self.spice.value * self.max_chance
        delta = 2 * self.max_delta * step_intensity - self.max_delta
        if step_chance < chance:
            val += delta
        if self.integer:
            val = round(val)
        return val

    def dice(self, step_data):
        self.step_data = step_data


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
