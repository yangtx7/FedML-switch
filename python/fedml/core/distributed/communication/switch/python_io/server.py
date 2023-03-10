from packet import *
import time
from node import Node
from node import Node


class Server(Node):

    def __init__(self, ip_addr: str, rx_port: int, tx_port: int, rpc_addr: str, node_id: int, is_remote_node: bool, iface: str = ""):
        super().__init__(ip_addr, rx_port, tx_port, rpc_addr, node_id, is_remote_node, iface)
        self.type = "server"

    def close(self):
        self._close()

    # 下发不检测丢包
    def send(self, node: Node, round_id: int, packet_list: list):
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
            # TODO: 限速
            if i % 100 == 0:
                time.sleep(0.001)
        send_end = time.time()

        resend_time = 0
        if node.type == "switch":
            for client in node.children.values():
                resend_time += self.check_and_retransmit(
                    client, round_id, packet_list)
        else:
            resend_time += self.check_and_retransmit(
                node, round_id, packet_list)

        print("server 发送结束 发送耗时 %f 发送速率 %f Mbps 重传耗时 %f" % (
            send_end - send_start,
            elemenet_per_packet * total_packet_num * 4 /
            1024 / 1024 * 8 / (send_end - send_start),
            resend_time))
            
        return

    def get_node_list_by_group_id(self):
        # TODO
        return self.children
