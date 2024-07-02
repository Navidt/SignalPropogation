import asyncio
import csv

from scapy.all import *
from sense_hat import SenseHat

"""
Run monitor_mode.sh first to set up the network adapter to monitor mode and to
set the interface to the right channel.
To get RSSI values, we need the MAC Address of the connection 
of the device sending the packets.
"""

# Variables to be modified
dev_mac = "e4:5f:01:d4:9d:ce"  # Change to a hidden camera's MAC
iface_n = "wlan1"  # Interface for network adapter (do not modify)
duration = 30  # Number of seconds to sniff for
record_file_name = datetime.now().strftime("%m-%d-%H:%M") + ".csv"

rssi_data = None


def create_record_file():
    """Create and prepare a file for record"""
    header = ["timestamp", "x", "y", "z", "rssi"]
    with open(record_file_name, "w", encoding="UTF8") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def write_to_file(file_name, data):
    """Write data to a file"""
    with open(file_name, "a", encoding="UTF8") as f:
        writer = csv.writer(f)
        writer.writerow(data)


def captured_packet_callback(pkt):
    """Save MAC addresses, time, and RSSI values to CSV file if MAC address of src matches.
    
    Example output CSV line:
    2024/02/11,11:12:13.12345, 1707692954.1, 00-B0-D0-63-C2-26, 00:00:5e:00:53:af, -32.2
    """
    cur_dict = {}
    # Check pkt for dst, src, and RSSI field
    try:
        cur_dict["dest"] = pkt.addr1
        cur_dict["src"] = pkt.addr2
        cur_dict["rssi"] = pkt.dBm_AntSignal
    except AttributeError:
        return  # Ignore packet without RSSI field

    # @TODO: Filter packets with src = the hidden camera's MAC
    if cur_dict["src"] != dev_mac:
        return

    global rssi_data
    rssi_data = cur_dict["rssi"]


async def main_loop():
    """Main loop to record IMU data (in Lab 3)"""
    start = time.time()

    while (time.time() - start) < duration:
        accel = sense.get_accelerometer_raw()  # returns float values representing acceleration intensity in Gs
        data = [time.time() - start, accel['x'], accel['y'], accel['z'], rssi_data]
        write_to_file(record_file_name, data)


if __name__ == "__main__":
    create_record_file()

    sense = SenseHat()
    sense.set_imu_config(True, True, True)  # Config the Gyroscope, Accelerometer, Magnetometer

    start_date_time = datetime.now().strftime("%d/%m/%Y,%H:%M:%S.%f")  # Get current date and time
    print("Start Time: ", start_date_time)

    t = AsyncSniffer(iface=iface_n, prn=captured_packet_callback, store=0)
    t.daemon = True
    t.start()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main_loop())
    loop.close()

    t.stop()
