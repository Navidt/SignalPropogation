from sense_hat import SenseHat
from picamera2 import Picamera2, Preview
import time

sense = SenseHat()
sense.clear()

x, y = 3, 5
sense.set_pixel(x, y, (255,255,255))

while True:
    for e in sense.stick.get_events():
        if e.action == "pressed":
            sense.set_pixel(x, y, (0,0,0))

            if e.direction == "middle":
                break
            elif e.direction == "up":
                y -= 1
            elif e.direction == "right":
                x += 1
            elif e.direction == "down":
                y += 1
            elif e.direction == "left":
                x -= 1

            x %= 8
            y %= 8
            sense.set_pixel(x, y, (255,255,255))

            
