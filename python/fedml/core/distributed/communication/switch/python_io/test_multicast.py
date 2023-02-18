from server import Server
from client import Client
from switch import Switch
import numpy as np
import time
from multiprocessing import Process

job_id = 100


def client_recv():
    server = Server(
        node_id=3,
        ip_addr="127.0.0.1",
        rx_port=50000,
        tx_port=50001,
        rpc_addr="127.0.0.1:50001",
        is_remote_node=True)
    client = Client(
        node_id=1,
        ip_addr='127.0.0.1',
        rx_port=50003,
        tx_port=50004,
        rpc_addr="127.0.0.1:50000",
        is_remote_node=False,
        iface="lo")
    
    packet_list = client.receive(
        node=server,
        job_id=job_id,
        total_packet_num=10240
    )
    print(packet_list[0].tensor[0:5])


def server_send():
    server = Server(
        node_id=3,
        ip_addr="127.0.0.1",
        rx_port=50000,
        tx_port=50001,
        rpc_addr="127.0.0.1:50001",
        is_remote_node=False,
        iface="lo")
    client = Client(
        node_id=1,
        ip_addr='127.0.0.1',
        rx_port=50003,
        tx_port=50004,
        rpc_addr="127.0.0.1:50000",
        is_remote_node=True)
    data = np.random.rand((256)).astype(np.float32)
    packet_list = [
        server.create_packet(
            job_id=job_id,
            segment_id=i,
            group_id=1,
            bypass=True,
            data=data
        ) for i in range(10240)
    ]
    print(data[0:5])
    server.send(
        node=client,
        job_id=job_id,
        packet_list=packet_list
    )
    


p1 = Process(target=client_recv)
p2 = Process(target=server_send)

p1.start()
time.sleep(0.5)
p2.start()

p1.join()
