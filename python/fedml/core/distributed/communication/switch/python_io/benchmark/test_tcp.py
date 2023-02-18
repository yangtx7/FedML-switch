import socket
import threading
import time
from packet import Packet, pkt_size
import numpy as np

send_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
send_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
send_sock.getsockopt(socket.IPPROTO_IP, socket.IP_OPTIONS)

recv_addr = ("127.0.0.1", 30003)
recv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
recv_sock.bind(recv_addr)
recv_sock.listen(1)
print('Waiting for connection...')

recv_cnt = 0
send_cnt = 0


def send():
    global send_cnt
    tensor = np.ones((256 * 1024 * 100), dtype=np.int32)
    send_sock.connect(recv_addr)
    send_sock.send(tensor.data)


def recv():
    global recv_cnt
    tensor = np.zeros((256 * 1024 * 100), dtype=np.int32)
    sock, addr = recv_sock.accept()
    cnt = 0
    
    while cnt != 1024 * 1024 * 100:
        bytes = sock.recv_into(tensor.data, 1024 * 1024 * 100)
        cnt += bytes
    sock.close()
    recv_sock.close()


r = threading.Thread(target=recv)
s = threading.Thread(target=send)

s.start()
r.start()
start = time.time()
r.join()
end = time.time()
print("cost %fs" % (end-start))
