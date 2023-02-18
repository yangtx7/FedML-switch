import socket
import numpy as np
from scapy.all import IPOption, raw

# tensor_id     uint32
# node_id       uint32
# grp_id        uint32

switchfl_header_format = ">"

# https://www.iana.org/assignments/ip-parameters/ip-parameters.xhtml
ipopts = raw(IPOption(
    copy_flag=1, optclass='control', option=30,
    value='213123213123'))

send_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
send_sock.connect(("127.0.0.1", 30003))
send_sock.setsockopt(socket.IPPROTO_IP, socket.IP_OPTIONS, ipopts)
arr = np.ones((365), dtype=np.int32)
send_sock.sendall(arr.data)
send_sock.close()