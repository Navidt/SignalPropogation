sudo wpa_cli -i wlan1 terminate
sudo ip link set wlan1 down
sudo iw dev wlan1 set type monitor
sudo ip link set wlan1 up
sudo iw wlan1 set freq 5580 80MHz