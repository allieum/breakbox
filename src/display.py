import contextlib
import time
from dataclasses import dataclass
from multiprocessing import Queue

import adafruit_ssd1306
import board
import busio
import utility
from modulation import Param
from PIL import Image, ImageDraw, ImageFont
from sample import Sample, SampleState, bank
from sequence import sequence

logger = utility.get_logger(__name__)
logger.setLevel("WARN")


@dataclass
class ParamUpdate:
    name: str
    value: str
    show_bar: bool
    fullness: float
    time: float

    def is_visible(self):
        return time.time() - self.time < UPDATE_LINGER


REFRESH_RATE = 0.025
UPDATE_LINGER = 3

q = Queue(10)


W = 128
H = 64

WHITE = 255
BLACK = 0


def init(samples: list[Sample]):
    for sample in samples:
        for param_name in [
                "gate", "gate_period", "spice_level",
                "volume", "pitch"]:
            param = getattr(sample, param_name)
            param.add_change_handler(on_param_changed(param, param_name))
    bank.add_change_handler(on_param_changed(bank, "bank"))
    sequence.bpm.add_change_handler(on_param_changed(sequence.bpm, "bpm", show_bar=False))


def on_param_changed(param: Param, name: str, show_bar=True):
    def on_change(value):
        fullness = param.normalize(value) if show_bar else 0
        with contextlib.suppress(Exception):
            q.put(ParamUpdate(name, value, show_bar, fullness, time.time()), block=False)

        logger.info(f"updated param {param}")
    return on_change


def run(display_q: Queue):
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        oled = adafruit_ssd1306.SSD1306_I2C(128, 64, i2c, addr=0x3d)
    except:
        logger.info("failed to initialize OLED")
        return

    last_refresh = time.time()
    last_text = None
    prev_sample_states = []
    bpm = "BEYOND"
    last_changed_param = ParamUpdate("dummy", "hot", False, 0.69, 0)

    while True:
        item = display_q.get()
        if isinstance(item, ParamUpdate):
            last_changed_param = item
            sample_states = prev_sample_states
        else:
            sample_states = item

        if time.time() - last_refresh < REFRESH_RATE:
            continue
        last_refresh = time.time()

        if last_changed_param.is_visible():
            text = f"{last_changed_param.name}: {last_changed_param.value}"
        else:
            text = f"{bpm} bpm"

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

        if last_changed_param.is_visible():
            draw_param(draw, last_changed_param)
        else:
            font = ImageFont.truetype("DejaVuSans.ttf", size=16)

            (left, top, right, bottom) = font.getbbox(text)
            (font_width, font_height) = right - left, bottom - top
            logger.info(f"drawing text {text} {font_width} x {font_height}")
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
    text = f"{param.name}: {param.value}"

    (left, top, right, bottom) = name_font.getbbox(text)
    (font_width, font_height) = right - left, bottom - top
    logger.info(f"drawing text {text} {font_width} x {font_height}")
    draw.text(
        (5, H * 3 // 4 - font_height // 2),
        text,
        font=name_font,
        fill=255,
    )

    if param.show_bar:
        draw_value_bar(draw, param.fullness)


def draw_sample_icons(draw, sample_states: list[SampleState]):
    xpad = 10
    ypad = 10
    total_width = W - xpad * 2
    for x, sample_state in zip(range(xpad, W - xpad + 1, total_width // 5), sample_states, strict=True):
        size = 8 if sample_state.selected else 4
        y = ypad - size // 2
        fill = WHITE if sample_state.selected else BLACK
        left = x - size // 2
        draw.rounded_rectangle((left, y, left + size, y + size),
                               radius=3, fill=fill, outline=WHITE)
