from node import Node
import numpy as np
from packet import *
import time

class Client(Node):
    def __init__(self, ip_addr: str, rx_port: int, tx_port: int, rpc_addr: str, node_id: int, is_remote_node: bool, max_group_id: int, iface: str = ""):
        super().__init__(ip_addr, rx_port, tx_port, rpc_addr, node_id, is_remote_node, iface, max_group_id)
        self.type = "client"

    def send(self, server: Node, round_id: int, packet_list: list, meta: dict = {}, has_switch: bool = False) -> int:
        """
        - server: 参数服务器
        - round_id: 此次发送的轮次号，需要与 packet_list 内所有包轮次号相同
        - packet_list: list[Packet]
        - meta: 任意可以 pickle.dumps 的字典，将会发送给对端
        - has_switch: 是否使用 switch 聚合模式发送
        """
        print("client 开始发送")
        server_addr = (server.options['ip_addr'], server.options['rx_port'])

        # 一次性发出发送窗口所有包
        finish_cnt = 0
        window_size = switch_pool_size  # TODO: 对于互联网传输提供更大的 window
        send_window = []
        send_window_time = []
        total_packet_num = len(packet_list)

        send_start = time.time()
        for i in range(min(window_size, total_packet_num)):
            try:
                send_window.append(packet_list[i])
                send_window_time.append(time.time())
                self.tx_sock.sendto(send_window[i].buffer, server_addr)
            except:
                pass
        rtt = 0.01
        rx_pkt = Packet()

        while finish_cnt != total_packet_num:
            self.tx_sock.settimeout(rtt)
            try:
                self.tx_sock.recv_into(rx_pkt.buffer)
                rx_pkt.parse_header()
                if rx_pkt.ack and rx_pkt.round_id == send_window[rx_pkt.pool_id].round_id and rx_pkt.segment_id == send_window[rx_pkt.pool_id].segment_id:
                    finish_cnt += 1
                    next_packet_segment_id = send_window[rx_pkt.pool_id].segment_id + window_size
                    # print("rtt %f" % (time.time() - send_window_time[rx_pkt.pool_id]))
                    # 尝试发出这个窗口下一个包
                    if next_packet_segment_id < total_packet_num:
                        send_window[rx_pkt.pool_id] = packet_list[next_packet_segment_id]
                        send_window_time[rx_pkt.pool_id] = time.time()
                        self.tx_sock.sendto(
                            send_window[rx_pkt.pool_id].buffer, server_addr)
            except:
                # 找出超时的包重发
                now = time.time()
                for i in range(len(send_window)):
                    if now - send_window_time[i] > rtt:
                        send_window[i].flow_control |= retranmission_bitmap
                        send_window[i].deparse_header()
                        send_window_time[i] = now
                        try:
                            self.tx_sock.sendto(
                                send_window[i].buffer, server_addr)
                        except:
                            pass

        send_end = time.time()

        time.sleep(0.05)
        retransmit_time = self.check_and_retransmit(server, round_id, packet_list, meta, len(packet_list) - 1)

        print("client %d 发送结束 发送耗时 %f 发送速率 %f Mbps 重传耗时 %f" % (
            self.options["node_id"],
            send_end - send_start,
            element_per_packet * total_packet_num * 4 / 1024 / 1024 * 8 / (send_end - send_start),
            retransmit_time)
        )
        return