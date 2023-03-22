from packet import *
import time
from node import Node
from typing import List

class Server(Node):

    def __init__(self, ip_addr: str, rx_port: int, tx_port: int, rpc_addr: str, node_id: int, is_remote_node: bool, iface: str = ""):
        super().__init__(ip_addr, rx_port, tx_port, rpc_addr, node_id, is_remote_node, iface, -1)
        self.type = "server"

    def close(self):
        self._close()

    # 下发不检测丢包
    def send(self, node: Node, round_id: int, packet_list: list, meta: dict, group_len_meta: List[tuple]):
        """
        - node: 发送目标
        - round_id: 此次发送的任务号
        - packet_list: list[Packet]
        """
        print("server 开始发送")
        send_start = time.time()
        server_addr = (node.options['ip_addr'], node.options['rx_port'])
        total_packet_num = len(packet_list)
        for i in range(total_packet_num):
            self.tx_sock.sendto(packet_list[i].buffer, server_addr)
            self.tx_sock.recvfrom(1000)

        send_end = time.time()

        resend_time = 0
        if node.type == "switch":
            for client in node.children.values():
                resend_time += self.check_and_retransmit(
                    client, 
                    round_id, 
                    packet_list, 
                    meta,
                    sum(group_len_meta[:client.options["max_group_id"]]) - 1 
                )
                print("check and retransmit finish node=%d" % (client.options["node_id"]))
        else:
            resend_time += self.check_and_retransmit(
                node, 
                round_id, 
                packet_list, 
                meta,
                sum(group_len_meta[:node.options["max_group_id"]]) - 1
            )

        print("server 发送结束 发送耗时 %f 发送速率 %f Mbps 重传耗时 %f" % (
            send_end - send_start,
            element_per_packet * total_packet_num * 4 /
            1024 / 1024 * 8 / (send_end - send_start),
            resend_time))
            
        return

    def get_node_list_by_group_id(self):
        # TODO
        return self.children
