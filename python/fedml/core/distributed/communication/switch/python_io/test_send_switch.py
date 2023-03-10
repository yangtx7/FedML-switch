from packet import *
import numpy as np
import socket

pkt = Packet()
pkt.set_header(
  flow_control=0,
  data_type=0,
  round_id=100,
  segment_id=0,
  node_id=1,
  aggregate_num=1,
  mcast_grp=2,
  pool_id=0
)
pkt.set_tensor(np.ones((256), dtype=np.int32))
pkt.deparse_header()
pkt.deparse_payload()
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, "veth1".encode())
sock.bind(("11.11.11.1", 50001))
dst_addr = ("11.11.11.9", 50000)
sock.sendto(pkt.buffer, dst_addr)
pkt.node_id = 2
pkt.deparse_header()
sock.sendto(pkt.buffer, dst_addr)
pkt.node_id = 3
pkt.deparse_header()
sock.sendto(pkt.buffer, dst_addr)

# client = sock.recvfrom_into(pkt.buffer, 1000)
# pkt.parse_header()
# print(pkt)