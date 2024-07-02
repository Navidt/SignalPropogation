from sense_hat import SenseHat
from picamera2 import Picamera2, Preview
import time

sense = SenseHat()
sense.clear()

picam2 = Picamera2()
config = picam2.create_preview_configuration()
picam2.configure(config)
picam2.start_preview(Preview.QTGL)
picam2.start()
flag = False

while True:
    for e in sense.stick.get_events():
        if e.action == "pressed" and e.direction == "middle":
            time.sleep(2)
            picam2.capture_file('hi.jpg')
            picam2.stop_preview()
            picam2.stop_()
            sense.show_message(f"hi.jpg",text_colour=(0,255,255), scroll_speed=0.05)
            flag = True
            break
    if flag:
        break
            

            
