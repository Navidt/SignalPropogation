# CS598-team11


## What is inside the repo

- *.dat: beamform file (bf file)
- read_b.py: read bf file into csv file
- parse_csi.py: Python version of the MATLAB tool in  [linux-80211n-csitool-supplementary](https://github.com/dhalperi/linux-80211n-csitool-supplementary/tree/master/matlab)
- plot.py: Visualize the CSI data
- data folder: our collected data in csv format
- merge_data.py: merge our CSI value with ground truth moisture value
- analyze.py: our prediction model

## How to execute the code

1. [Follow Installation Instructions to generate beamform file (*.dat)](https://dhalperi.github.io/linux-80211n-csitool/installation.html)
2. Run `read_b.py` to store the data from beamform file into csv file
3. Run `merge_data.py` to match the CSI data with ground truth moistrue value
4. Run `analyze.py` to build a model to predict moisture value based on CSI data

## Documentation about hardware setup

Hardware:

- Intel NUC Kit D54250WYKH: Mini-PC
- Lenmar PowerPort laptop: Laptop Power Pack
- Intel Wi-Fi Wireless Link 5300 NIC: WiFi card
- NETGEAR 5-Port Gigabit Ethernet Unmanaged Switch (GS105)

Software:

- Linux 802.11n CSI Tool