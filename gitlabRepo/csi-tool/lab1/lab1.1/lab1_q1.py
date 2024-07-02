from sense_hat import SenseHat
from time import sleep


sense = SenseHat()
sense.clear()

t = sense.get_temperature()
p = sense.get_pressure()
h = sense.get_humidity()

sense.show_message(f"{t:.1f}",text_colour=(0,0,255), scroll_speed=0.05)
sense.show_message(f"{p:.1f}",text_colour=(255,0,255), scroll_speed=0.05)
sense.show_message(f"{h:.1f}",text_colour=(0,255,255), scroll_speed=0.05)

print(t)
print(p)
print(h)

for i in range(0, 10000):
    print(abs(t - sense.get_temperature()))
    if (abs(t - sense.get_temperature()) >= 1):
        sense.set_pixel(3, 3, (0,0,0))
        sleep(0.5)
        sense.set_pixel(3, 3, (255,255,255))
        sleep(0.5)
        sense.set_pixel(3, 3, (0,0,0))
    
    sleep(1)
    