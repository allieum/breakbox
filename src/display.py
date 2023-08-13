import contextlib
import time
from collections import namedtuple
from dataclasses import dataclass, field
from multiprocessing import Queue

import adafruit_ssd1306
import board
import busio
import utility
from modulation import Param
from PIL import Image, ImageDraw, ImageFont
from sample import NUM_BANKS, Sample, SampleState, current_bank
from sequence import sequence

logger = utility.get_logger(__name__)


@dataclass
class ParamUpdate:
    name: str | None
    value: str
    show_bar: bool
    fullness: float
    time: float
    priority: int = field(default=0)  # higher is bigger priority

    def is_alive(self):
        return time.time() - self.time < UPDATE_LINGER

    def text(self):
        if self.name == PARAM_BANK:
            return ""
        label = f"{self.name}: " if self.name else ""
        return label + str(self.value)


Point = namedtuple("Point", ["x", "y"])
Size = namedtuple("Size", ["w", "h"])

REFRESH_RATE = 0.010
UPDATE_LINGER = 1

display_queue = Queue(1)
param_queue = Queue()


W = 128
H = 64

WHITE = 255
BLACK = 0

PARAM_BANK = 'bank'

def init(samples: list[Sample]):
    for sample in samples:
        for param_name in [
                "gate", "gate_period", "spice_level",
                "volume", "pitch"]:
            param = getattr(sample, param_name)
            display_on_change(param_queue, param, param_name, True)
    display_on_change(param_queue, current_bank, PARAM_BANK, priority=2)
    display_on_change(param_queue, sequence.bpm, "bpm")


def display_on_change(param_q: 'Queue[ParamUpdate]', param: Param, name: str | None = None, show_bar=False, priority=0, fn=None):
    def on_change(value):
        fullness = param.normalize(value) if show_bar else 0
        with contextlib.suppress(Exception):
            if fn:
                value = fn(value)
            param_q.put(ParamUpdate(name, value, show_bar, fullness, time.time(), priority=priority), block=False)

        logger.info(f"updated param {param}")
    param.add_change_handler(on_change)


def run(display_q: 'Queue[list[SampleState]]', param_q: 'Queue[ParamUpdate]'):
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        oled = adafruit_ssd1306.SSD1306_I2C(128, 64, i2c, addr=0x3d)
    except:
        logger.info("failed to initialize OLED")
        return

    last_refresh = time.time()
    last_text = None
    prev_sample_states = []
    selected_sample_name = Param("juice_fruit.wav")
    display_on_change(param_q, selected_sample_name, priority=1)
    last_changed_param = ParamUpdate(
        "dummy", "hot", show_bar=False, fullness=0.69, time=0)
    bank = "1"

    logger.info(f"display says hay")
    while True:
        sample_states = display_q.get()
        param_update = None
        try:
            param_update = param_q.get_nowait()
            is_param_update = True
        except:
            is_param_update = False

        if is_param_update and param_update:
            # logger.info(f"got {param_update.name}, (queue length {display_q.qsize()})")
            if param_update.name == PARAM_BANK and isinstance(param_update.value, int):
                bank = str(param_update.value + 1)
                param_update.value = str(int(param_update.value) + 1)
            if last_changed_param.priority >= param_update.priority \
                    and last_changed_param.is_alive():
                param_q.put(param_update)
            else:
                last_changed_param = param_update

        # if time.time() - last_refresh < REFRESH_RATE and not is_param_update:
        #     continue
        # last_refresh = time.time()

        if any((selected_sample := smpl).selected for smpl in sample_states):
            selected_sample_name.set(selected_sample.name)

        if last_changed_param.is_alive():
            text = last_changed_param.text()
        else:
            text = f"bank {bank}"

        if last_text == text and sample_states == prev_sample_states:
            continue

        last_text = text
        prev_sample_states = sample_states

        # Create blank image for drawing.
        # Make sure to create image with mode '1' for 1-bit color.
        image = Image.new("1", (oled.width, oled.height))

        # Get drawing object to draw on image.
        draw = ImageDraw.Draw(image)

        draw_sample_icons(draw, sample_states)

        if any((selected_sample := smpl).selected for smpl in sample_states):
            draw_step_indicator(draw, selected_sample, H - 2, 2)

        if any((dtx_selected_sample := smpl).dtx_selected for smpl in sample_states):
            draw_step_indicator(draw, dtx_selected_sample, H - 4, 2)

        if last_changed_param.is_alive():
            draw_param(draw, last_changed_param)
        else:
            font = ImageFont.truetype("DejaVuSans.ttf", size=16)

            (left, top, right, bottom) = font.getbbox(text)
            (font_width, font_height) = right - left, bottom - top
            logger.debug(f"drawing text {text} {font_width} x {font_height}")
            draw.text(
                (oled.width // 2 - font_width // 2,
                 oled.height // 2 - font_height // 2),
                text,
                font=font,
                fill=255,
            )

        # Display image
        oled.image(image)
        oled.show()


def draw_value_bar(draw, fullness):
    bar_height = 16
    bar_width = 100

    xmid = W // 2
    ymid = H // 2

    # Draw a white container for bar
    draw.rectangle((x1 := xmid - bar_width // 2,
                    y1 := ymid - bar_height // 2,
                    _x2 := xmid + bar_width // 2,
                    y2 := ymid + bar_height // 2),
                   outline=255, fill=0)

    # Draw rectangle for filled portion
    draw.rectangle((x1, y1, x1 + round(fullness * bar_width), y2),
                   outline=255, fill=255)


def draw_param(draw, param: ParamUpdate):
    # Load default font.
    name_font = ImageFont.truetype("DejaVuSans.ttf", size=12)
    text = param.text()

    (left, top, right, bottom) = name_font.getbbox(text)
    (font_width, font_height) = right - left, bottom - top
    logger.debug(f"drawing text {text} {font_width} x {font_height}")
    draw.text(
        (5, H * 3 // 4 - font_height // 2),
        text,
        font=name_font,
        fill=255,
    )

    if param.show_bar:
        draw_value_bar(draw, param.fullness)


def draw_step_indicator(draw: ImageDraw.ImageDraw, selected: SampleState, y: int, height: int):
    # do the things
    # start with love
    # two bars on the bottom of the screen:
    #   one, for each selected sample
    #   shows you what step it is
    if selected.step is None:
        return
    step_width = W // selected.steps
    x = (selected.step % selected.steps) * step_width
    draw.rectangle(((x, y), (x + step_width, y + height)), fill=WHITE)


def draw_sample_icons(draw, sample_states: list[SampleState]):
    xpad = 10
    ypad = 10
    y = ypad
    total_width = W - xpad * 2
    heart_size = Size(15, 12)
    big_dot_size = 10
    lil_dot_size = 1
    for x, sample_state in zip(range(xpad, W - xpad + 1, total_width // 5), sample_states, strict=False):
        position = Point(x, y)
        if sample_state.dtx_selected:
            draw_heart(draw, heart_size, position, WHITE)
        if sample_state.selected:
            draw_icon(draw, position, big_dot_size, fill=BLACK, outline=WHITE)
        draw_icon(draw, position, lil_dot_size, fill=BLACK, outline=WHITE)

        # show bank numbers
        # bank_font = ImageFont.truetype("DejaVuSans.ttf", size=12)

        # if sample_state.bank < NUM_BANKS:
        #     text = str(sample_state.bank + 1)
        # else:
        #     text = "*"

        # (left, top, right, bottom) = bank_font.getbbox(text)
        # (font_width, font_height) = right - left, bottom - top
        # logger.info(f"drawing text {text} {font_width} x {font_height}")
        # draw.text(
        #     (x, 0),
        #     text,
        #     font=bank_font,
        #     fill=255,
        # )



def draw_icon(draw: ImageDraw.ImageDraw, position: Point, diameter, fill, outline):
    r = diameter // 2
    draw.ellipse((position.x - r, position.y - r, position.x + r, position.y + r),
                 fill=fill, outline=outline, width=2)


def draw_heart(draw: ImageDraw.ImageDraw, size: Size, position: Point, fill):
    xmid = size.w // 2
    ymid = size.h // 2
    dx = position.x - xmid
    dy = position.y - ymid
    poly = [
        Point(size.w / 10, size.h / 3),
        Point(size.w / 10, 81 * size.h / 120),
        Point(size.w / 2, size.h),
        Point(size.w - size.w / 10, 81 * size.h / 120),
        Point(size.w - size.w / 10, size.h / 3),
    ]
    trans_poly = [(point.x + dx, point.y + dy) for point in poly]
    draw.polygon(trans_poly, fill=fill)
    draw.ellipse((dx, dy,  size.w / 2 + dx, 3 * size.h / 4 + dy), fill=fill)
    draw.ellipse((size.w / 2 + dx, dy,  size.w + dx,
                 3 * size.h / 4 + dy), fill=fill)
