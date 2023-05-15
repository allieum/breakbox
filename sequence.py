import time
import pygame.mixer

from midi import START, STOP, CLOCK
import sample

MAX_BEATS = 8
STEPS_PER_BEAT = 4
MAX_STEPS = MAX_BEATS * STEPS_PER_BEAT

step_interval = 60 / 143 / 4
midi_lag_time = 0.039


class Sequence:
    def __init__(self):
        self.is_started = False
        self.clock_count = 0
        self.beat = 0
        self.step = 0
        self.steps = MAX_STEPS
        self.measure_start = time.time()
        self.played_step = False
        self.internal_start_time = None
        self.midi_started = False

    def start_internal(self):
        self.internal_start_time = time.time()
        self._start()
        sample.play_samples(0, self.internal_start_time)
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
                    self.step_forward()
                if self.clock_count == 24:
                    self.clock_count = 0
                    # self.beat = (self.beat + 1) % MAX_BEATS
                    # if self.beat == 0:
                    #     self.measure_start = time.time()

        next_step_time = self.measure_start + step_interval * (self.step + 1)
        lag_time = midi_lag_time if self.midi_started else 0
        step_predicted = time.time() >= next_step_time - lag_time and not self.played_step
        if step_predicted and self.is_started:
            next_step = (self.step + 1) % MAX_STEPS
            sample.play_samples(next_step, next_step_time - lag_time)
            self.played_step = True
            if self.is_internal():
                # print(f'internal step {next_step}')
                self.step_forward()

        sample.update()

    def step_forward(self):
        self.step = (self.step + 1) % self.steps
        self.played_step = False
        if self.step == 0:
            self.measure_start = time.time()
        # print(f'step {self.step}')


sequence = Sequence()
