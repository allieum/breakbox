import board
import adafruit_ws2801

odata = board.D5
oclock = board.D6
numleds = 25
bright = 1.0
leds = adafruit_ws2801.WS2801(oclock, odata, numleds, brightness=bright, auto_write=False)


def init():
    leds.fill(0)
    leds[0] = (0, 255, 0)
    leds[1] = (255, 0, 0)
    leds[2] = (0, 0, 255)
    leds.show()
