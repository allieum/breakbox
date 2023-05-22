import time
from collections import defaultdict
import pygame.mixer

from midi import START, STOP, CLOCK
import sample
import modulation
import utility

logger = utility.get_logger(__name__)

MAX_BEATS = 8
STEPS_PER_BEAT = 4
MAX_STEPS = MAX_BEATS * STEPS_PER_BEAT

step_interval = 60 / 143 / 4
midi_lag_time = 0.039
lookahead_time = 0.100
# lookahead_time = 0.100

class Sequence:

    def __init__(self):
        self.is_started = False
        self.clock_count = 0
        self.beat = 0
        self.step = 0
        self.steps = MAX_STEPS
        self.measure_start = time.time()
        self.internal_start_time = None
        self.midi_started = False
        self.played_step = False
        self.last_queued_step = -1
        self.lfos = []

    def start_internal(self):
        self.internal_start_time = time.time()
        self._start()
        # sample.queue_samples(0, self.internal_start_time)
        print("starting internal clock")

    def stop_internal(self):
        if not self.is_internal():
            return
        self.internal_start_time = None
        self._stop()
        print("stopping internal clock")

    def is_internal(self):
        return self.internal_start_time is not None

    def start_midi(self):
        self.stop_internal()
        self.midi_started = True
        self._start()

    def stop_midi(self):
        self.midi_started = False
        self._stop()

    def _start(self):
        self.is_started = True
        self.clock_count = 0
        self.beat = 0
        self.step = 0
        self.measure_start = time.time()
        self.last_queued_step = -1

    def _stop(self):
        self.is_started = False
        pygame.mixer.stop() # todo only stop own samples

    def update(self, midi_status=None):
        if midi_status is not None:
            if midi_status == START:
                self.start_midi()

            if midi_status == STOP:
                self.stop_midi()

            if midi_status == CLOCK and self.midi_started:
                self.clock_count += 1
                # print(f"got {self.clock_count} clocks")
                if self.clock_count > 0 and self.clock_count % (24 / STEPS_PER_BEAT) == 0:
                    self.step_forward(time.time())
                if self.clock_count == 24:
                    self.clock_count = 0
                    # self.beat = (self.beat + 1) % MAX_BEATS
                    # if self.beat == 0:
                    #     self.measure_start = time.time()
        now = time.time()
        prev = self.last_queued_step
        for i in (self.inc(prev), self.inc(prev, 2), self.inc(prev, 3)):
            if self.is_started and now + lookahead_time >= (t := self.step_time(i)):
                logger.debug(f"------- queuing step {i} === {t - self.measure_start}")
                sample.queue_samples(i, t)
                self.last_queued_step = i

        next_step_time = self.step_time(self.inc(self.step))
        step_predicted = now >= next_step_time and not self.played_step
        if step_predicted and self.is_started:
            next_step = (self.step + 1) % MAX_STEPS
            self.played_step = True
            if self.is_internal():
                # print(f'internal step {next_step}')
                self.step_forward(now)

    def inc(self, step, n=1):
        return (step + n) % self.steps

    def step_time(self, step):
        next_step_time = self.measure_start + step_interval * step
        if step < self.step:
            next_step_time += step_interval * self.steps
        lag_time = midi_lag_time if self.midi_started else 0
        return next_step_time - lag_time


    def modulate(self, param, lfo, amount):
        param.modulate(lfo, amount)

    def step_forward(self, t):
        self.step = self.inc(self.step)
        self.played_step = False
        if self.step == 0:
            self.measure_start = t
        for lfo in self.lfos:
            lfo.step()

    def make_lfo(self, period, shape):
        self.lfos.append(lfo := modulation.Lfo(period, shape))
        return lfo

sequence = Sequence()
