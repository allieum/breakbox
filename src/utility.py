from dataclasses import dataclass, field
from math import inf
import os
import sys
import logging


def get_logger(name):
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s %(threadName)s:%(funcName)s: %(message)s',
        # filename='break.log'
    )
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


@dataclass
class TimeInterval:
    start: float
    end: float = field(default=inf)

    def contains(self, t):
        return self.has_end and t >= self.start and t <= self.end

    def combine(self, other):
        if self.contains(other.start):
            return TimeInterval(self.start, other.end)
        if self.contains(other.end):
            return TimeInterval(other.start, self.end)
        return self

    def has_end(self):
        return self.end != inf


def restart_program():
    python = sys.executable
    os.execl(python, python, * sys.argv)


def make_even(x):
    if x % 2 == 1:
        x -= 1
    return x
