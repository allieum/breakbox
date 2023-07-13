import adafruit_ssd1306
import board
import busio
import time
from PIL import Image, ImageDraw, ImageFont

import utility

logger = utility.get_logger(__name__)

BORDER = 5
REFRESH_RATE = 0.5
last_refresh = time.time()
last_state = None

oled = None
def init():
    global oled
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        oled = adafruit_ssd1306.SSD1306_I2C(128, 64, i2c, addr=0x3d)
    except:
        logger.info("failed to initialize OLED")

def update(state):
    if oled is None:
        return

    global last_refresh, last_state
    if time.time() - last_refresh < REFRESH_RATE:
        return
    last_refresh = time.time()

    if state == last_state:
        return
    last_state = state
    bpm = state

    # Create blank image for drawing.
    # Make sure to create image with mode '1' for 1-bit color.
    image = Image.new("1", (oled.width, oled.height))

    # Get drawing object to draw on image.
    draw = ImageDraw.Draw(image)

    # Draw a white background
    draw.rectangle((0, 0, oled.width, oled.height), outline=255, fill=255)

    # Draw a smaller inner rectangle
    draw.rectangle(
        (BORDER, BORDER, oled.width - BORDER - 1, oled.height - BORDER - 1),
        outline=0,
        fill=0,
    )

    # Load default font.
    # font = ImageFont.load_default()
    font = ImageFont.truetype("DejaVuSans.ttf", size=16)

    # Draw Some Text
    text = f"{bpm} bpm"
    (font_width, font_height) = font.getsize(text)
    draw.text(
        (oled.width // 2 - font_width // 2, oled.height // 2 - font_height // 2),
        text,
        font=font,
        fill=255,
    )

    # Display image
    oled.image(image)
    oled.show()
