import socket

message = "Hi"
ip_addr = "10.193.31.222"

port_num = 5000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
sock.sendto(bytes(message, "utf-8"), (ip_addr, port_num))
