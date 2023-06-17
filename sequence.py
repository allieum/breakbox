import time
import pygame.mixer

from midi import START, STOP, CLOCK
import sample
import modulation
import utility

logger = utility.get_logger(__name__)

MAX_BEATS = 16
STEPS_PER_BEAT = 4
MAX_STEPS = MAX_BEATS * STEPS_PER_BEAT

midi_lag_time = 0.039
lookahead_time = 0.100

class Sequence:

    def __init__(self):
        self.is_started = False
        self.clock_count = 0
        self.beat = 0
        self.step = 0
        self.steps = MAX_STEPS
        self.measure_start = time.time()
        self.last_midi_beat = time.time()
        self.midi_bpm = 143
        self.provisional_midi_bpm = 143
        self.midi_stable_beats = 0
        self.internal_start_time = None
        self.midi_started = False
        self.played_step = False
        self.last_queued_step = -1
        self.lfos = []
        self.bpm = modulation.Param(143)
        self.callback = None

    def control_bpm(self, encoder):
        self.bpm.control(encoder, 1, sample.stretch_samples)

    def step_duration(self):
        bpm = self.midi_bpm if self.midi_started else self.bpm.get()
        return 60 / bpm / STEPS_PER_BEAT

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
        self.last_midi_beat = time.time()

    def calculate_midi_bpm(self, beat_time):
        sec_per_beat = beat_time - self.last_midi_beat
        logger.debug(f"sec per beat {sec_per_beat}")
        bpm = round(60 / sec_per_beat)
        return bpm

    def update_midi_bpm(self, t):
        bpm = self.calculate_midi_bpm(t)
        if bpm != self.midi_bpm:
            if bpm == self.provisional_midi_bpm:
                self.midi_stable_beats += 1
            else:
                self.midi_stable_beats = 0
                self.provisional_midi_bpm = bpm
            if self.midi_stable_beats > 5:
                sample.stretch_samples(bpm)
                self.midi_bpm = bpm
                logger.info(f"midi bpm changed to {bpm} on step {self.step}")

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
        pygame.mixer.stop()

    def update(self, midi_status=None):
        now = time.time()
        if midi_status is not None:
            if midi_status == START:
                self.start_midi()

            if midi_status == STOP:
                self.stop_midi()

            if midi_status == CLOCK and self.midi_started:
                self.clock_count = (self.clock_count + 1) % 24
                if self.clock_count == 0:
                    self.update_midi_bpm(now)
                    self.last_midi_beat = now
                if self.clock_count % (24 / STEPS_PER_BEAT) == 0:
                    self.step_forward(time.time())

        prev = self.last_queued_step
        for i in (self.inc(prev), self.inc(prev, 2), self.inc(prev, 3)):
            if self.is_started and now + lookahead_time >= (t := self.step_time(i)):
                logger.debug(f"------- queuing step {i} === {t - self.measure_start}")
                sample.queue_samples(i, t, self.step_duration())
                self.last_queued_step = i

        next_step_time = self.step_time(self.inc(self.step))
        step_predicted = now >= next_step_time and not self.played_step
        if step_predicted and self.is_started:
            self.played_step = True
            if self.is_internal():
                self.step_forward(now)

    def on_step(self, callback):
        self.callback = callback

    def inc(self, step, n=1):
        return (step + n) % self.steps

    def step_time(self, step):
        next_step_time = self.measure_start + self.step_duration() * step
        if step < self.step:
            next_step_time += self.step_duration() * self.steps
        lag_time = midi_lag_time if self.midi_started else 0
        return next_step_time - lag_time

    def step_forward(self, t):
        self.step = self.inc(self.step)
        self.played_step = False
        if self.step == 0:
            self.measure_start = t
        logger.debug(f"step {self.step}")
        if self.callback is not None:
            # TODO should be async
            self.callback(self.step)
        # for lfo in self.lfos:
        #     lfo.step()

    # def make_lfo(self, period, shape):
    #     self.lfos.append(lfo := modulation.Lfo(period, shape))
    #     return lfo

sequence = Sequence()
