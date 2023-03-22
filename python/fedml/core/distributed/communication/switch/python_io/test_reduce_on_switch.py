from server import Server
from client import Client
from switch import Switch
import numpy as np
import time
# from multiprocessing import Process
from threading import Thread

round_id = 100
pkt_num = 1000
node_num = 5

start_port = 50000

server_node_id = 100
server_ip_addr = "127.0.0.1"
server_rx_port = start_port
server_tx_port = start_port + 1
server_rpc_addr = "127.0.0.1:%d" % (start_port + 2)

mock_switch_node_id = 101
mock_switch_ip_addr = "127.0.0.1"
mock_switch_port = 30000


client_configs = [
    {
        "node_id": i+1,
        "ip_addr": "127.0.0.1",
        "rx_port": start_port + (i+1)*3,
        "tx_port": start_port + (i+1)*3 + 1,
        "rpc_addr": "127.0.0.1:%d" % (start_port + (i+1) * 3 + 2)
    } for i in range(node_num)
]

data = np.random.rand((256 * pkt_num)).astype(np.float32)

def client_send(index: int):
    global data
    server = Server(
        node_id=server_node_id,
        ip_addr=server_ip_addr,

        # rx_port=server_rx_port,
        rx_port=mock_switch_port,

        tx_port=server_tx_port,
        rpc_addr=server_rpc_addr,
        is_remote_node=True
    )
    config = client_configs[index]
    client = Client(
        node_id=config["node_id"],
        ip_addr=config["ip_addr"],
        rx_port=config["rx_port"],
        tx_port=config["tx_port"],
        rpc_addr=config["rpc_addr"],
        is_remote_node=False,
        # iface="veth1"
    )
    packet_list = [client.create_packet(
        round_id=round_id,
        segment_id=i,
        group_id=1,
        bypass=False,
        data=data[256*i: 256*(i+1)]
    ) for i in range(pkt_num)]
    client.send(
        server=server,
        round_id=round_id,
        packet_list=packet_list,
        has_switch=True
    )


def server_receive():
    server = Server(
        node_id=server_node_id,
        ip_addr=server_ip_addr,
        rx_port=server_rx_port,
        tx_port=server_tx_port,
        rpc_addr=server_rpc_addr,
        is_remote_node=False,
        # iface="veth5"
    )
    switch = Switch(
        node_id=mock_switch_node_id,
        ip_addr=mock_switch_ip_addr,
        rx_port=mock_switch_port,
        tx_port=mock_switch_port,
        rpc_addr=""
    )
    for config in client_configs:
        switch.add_child(Client(
            node_id=config["node_id"],
            ip_addr=config["ip_addr"],
            rx_port=config["rx_port"],
            tx_port=config["tx_port"],
            rpc_addr=config["rpc_addr"],
            is_remote_node=True,
            # iface="veth1"
        ))

    packet_list = server.receive(
        node=switch,
        round_id=round_id,
        total_packet_num=pkt_num
    )
    for i, pkt in enumerate(packet_list):
        expect = data[i*256: (i+1) * 256] * node_num
        diff = (pkt.tensor - expect).max()
        if diff > 1e-6:
            print("diff")
    print("server recv finish")

p1 = Thread(target=server_receive)
p1.start()
time.sleep(0.1)

for i in range(node_num):
    Thread(target=client_send, args=(i,)).start()

p1.join()
