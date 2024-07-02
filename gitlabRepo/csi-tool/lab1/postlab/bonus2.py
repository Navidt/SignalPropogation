from sense_hat import SenseHat
import socket

def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    return s.getsockname()[0]

sense = SenseHat()
sense.clear()

ip = get_ip_address()

sense.show_message(f"{ip}",text_colour=(0,0,255), scroll_speed=0.05)