from server import Server
from client import Client
from switch import Switch
import numpy as np
from multiprocessing import Process
from threading import Thread

round_id = 100
pkt_num = 10

server_node_id = 100
server_ip_addr = "127.0.0.1"
server_rx_port = 50000
server_tx_port = 50001
server_rpc_addr = "127.0.0.1:50002"

mock_switch_node_id = 101
mock_switch_ip_addr = "127.0.0.1"
mock_switch_port = 30000

client_node_id = 1
client_ip_addr = "127.0.0.1"
client_rx_port = 50003
client_tx_port = 50004
client_rpc_addr = "127.0.0.1:50005"

server = Server(
    node_id=server_node_id,
    ip_addr=server_ip_addr,
    rx_port=server_rx_port,
    tx_port=server_tx_port,
    rpc_addr=server_rpc_addr,
    is_remote_node=True,
    iface="lo"
)

data = None

def client_recv():
    server = Server(
        node_id=server_node_id,
        ip_addr=server_ip_addr,
        rx_port=server_rx_port,
        tx_port=server_tx_port,
        rpc_addr=server_rpc_addr,
        is_remote_node=True
    )
    client = Client(
        node_id=client_node_id,
        ip_addr=client_ip_addr,
        rx_port=client_rx_port,
        tx_port=client_tx_port,
        rpc_addr=client_rpc_addr,
        is_remote_node=False,
        iface="lo"
    )
    packet_list2 = client.receive(
        node=server,
        round_id=round_id,
        total_packet_num=pkt_num
    )
    print(packet_list2[0].tensor[0:5])
    recv_data = None
    for packet in packet_list2:
        if recv_data is None:
            recv_data = np.array(packet.tensor)
        else:
            recv_data = np.concatenate((recv_data, packet.tensor))
    loss = recv_data - data
    print(loss)


def server_send():
    global data
    server = Server(
        node_id=server_node_id,
        ip_addr=server_ip_addr,
        rx_port=server_rx_port,
        tx_port=server_tx_port,
        rpc_addr=server_rpc_addr,
        is_remote_node=False,
        iface="lo"
    )
    client = Client(
        node_id=client_node_id,
        ip_addr=client_ip_addr,
        rx_port=client_rx_port,
        tx_port=client_tx_port,
        rpc_addr=client_rpc_addr,
        is_remote_node=True
    )
    switch = Switch(
        node_id=mock_switch_node_id,
        ip_addr=mock_switch_ip_addr,
        rx_port=mock_switch_port,
        tx_port=mock_switch_port,
        rpc_addr="",
        is_remote_node=True
    )
    switch.add_child(client)

    data = np.random.rand((256 * pkt_num)).astype(np.float32)
    packet_list = [
        server.create_packet(
            round_id=round_id,
            segment_id=i,
            group_id=1,
            bypass=False,
            data=data[i*256:(i+1)*256],
            multicast=True
        ) for i in range(pkt_num)
    ]
    print(data[0:5])
    server.send(
        node=switch,
        round_id=round_id,
        packet_list=packet_list
    )


t2 = Thread(target=client_recv)
t2.start()
t1 = Thread(target=server_send)
t1.start()

t2.join()


# p1 = Process(target=client_recv)
# p2 = Process(target=server_send)

# p1.start()
# time.sleep(0.5)
# p2.start()

# p1.join()
