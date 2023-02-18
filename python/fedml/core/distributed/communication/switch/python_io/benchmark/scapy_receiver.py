import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 50000))
while True:
  data = sock.recv(1000)
  print(data)