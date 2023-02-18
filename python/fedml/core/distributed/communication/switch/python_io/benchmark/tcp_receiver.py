import socket

recv_addr = ("0.0.0.0", 30003)
recv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
recv_sock.bind(recv_addr)
recv_sock.listen(1)
sock, addr = recv_sock.accept()

data = sock.recv(10000)

sock.close()