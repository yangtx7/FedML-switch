import socket
import threading
import time
from packet import Packet, pkt_size
import numpy as np

send_addr = ("127.0.0.1", 30000)
send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
send_sock.bind(send_addr)
recv_addr = ("127.0.0.1", 30001)
recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
recv_sock.bind(recv_addr)
send_cnt = 0
send_cnt2 = 0
recv_cnt = 0


def send():
    global send_cnt
    tensor = np.ones((2048), dtype=np.int32)
    pkt = Packet()
    rx_pkt = Packet()
    pkt.set_tensor(tensor)
    while True:
        time.sleep(0.0001) # 2ms
        pkt.set_header(0, 1, 1, 1, 1, 0, 0)
        pkt.deparse_buffer()
        send_sock.sendto(pkt.buffer, recv_addr)
        send_cnt += 1
        send_sock.recv_into(rx_pkt.buffer, pkt_size)

def send2():
    global send_cnt2
    tensor = np.ones((2048), dtype=np.int32)
    pkt = Packet()
    rx_pkt = Packet()
    pkt.set_tensor(tensor)
    while True:
        time.sleep(0.0001) # 2ms
        pkt.set_header(0, 1, 1, 1, 1, 0, 0)
        pkt.deparse_buffer()
        send_sock.sendto(pkt.buffer, recv_addr)
        send_cnt2 += 1
        send_sock.recv_into(rx_pkt.buffer, pkt_size)

def recv():
    global recv_cnt
    pkt = Packet()
    tensor = np.zeros((2048), dtype=np.int32)
    while True:
        recv_sock.recv_into(pkt.buffer, pkt_size)
        pkt.parse_buffer()
        recv_cnt += 1
        recv_sock.sendto(pkt.gen_ack_packet(), send_addr)


s = threading.Thread(target=send)
s2 = threading.Thread(target=send)
r = threading.Thread(target=recv)

s2.start()
s.start()
r.start()

time.sleep(1)

print("send_byte %d MB" % ((send_cnt + send_cnt2) * 8192 / 1024 / 1024))
print("recv_byte %d MB" % (recv_cnt * 8192 / 1024 / 1024))
