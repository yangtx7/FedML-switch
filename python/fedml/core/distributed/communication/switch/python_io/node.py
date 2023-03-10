import numpy as np
import socket
import threading
from packet import *
from job import Job
import grpc
from io_pb2_grpc import *
from io_pb2 import *
import typing
from grpc_server import GrpcServer
import time


class Node:
    def __init__(self, ip_addr: str, rx_port: int, tx_port: int, rpc_addr: str, node_id: int, is_remote_node: bool, iface: str, group_id: int = 10):
        self.options = {
            "ip_addr": ip_addr,
            "rx_port": rx_port,
            "tx_port": tx_port,
            "rpc_addr": rpc_addr,
            "node_id": node_id,
            "group": group_id,  # 所在的分组
            "speed": 100,  # 100 Mbps
        }
        self.type = "node"
        self.children: dict[int, Node] = {}
        self.iface = iface

        self.rx_jobs: dict[(int, int), Job] = {}
        self.rx_jobs_lock = threading.Lock()

        self.rpc_stub: typing.Optional[SwitchmlIOStub] = None
        self.rpc_server: typing.Optional[GrpcServer] = None
        self.rx_sock: typing.Optional[socket.socket] = None
        self.tx_sock: typing.Optional[socket.socket] = None

        if not is_remote_node:
            self.rx_sock = self._create_udp_socket()
            self.rx_sock.bind(
                (self.options['ip_addr'], self.options['rx_port']))
            self.tx_sock = self._create_udp_socket()
            self.tx_sock.bind(
                (self.options['ip_addr'], self.options['tx_port']))

            print("成功监听数据端口 %s:%d" %
                  (self.options['ip_addr'], self.options['rx_port']))
            self.__receive_thread = threading.Thread(
                target=self.receive_thread,
                daemon=True
            )
            self.__receive_thread.start()
            print("成功启动接收线程 id=%d" % (self.__receive_thread.ident))
            self.rpc_server: GrpcServer = GrpcServer(self)
            print("成功启动 grpc 服务 %s" %
                  (self.options['rpc_addr']))
        else:
            if self.type != "switch":
                self._init_as_remote_node()

    def _create_udp_socket(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET,
                        socket.SO_BINDTODEVICE, self.iface.encode())
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock

    # stop the server
    def _close(self):
        if self.rpc_server is not None:
            print("grpc 服务正在关闭")
            self.rpc_server.stop()
        if self.tx_sock is not None:
            print("读 socket 正在关闭")
            self.tx_sock.close()
        if self.rx_sock is not None:
            print("写 socket 正在关闭")
            self.rx_sock.close()
        if self.__receive_thread is not None:
            # TODO
            pass

    def _init_as_remote_node(self):
        if not self.type == "switch":
            addr = self.options['rpc_addr']
            channel = grpc.insecure_channel(addr, options=[
                ('grpc.max_send_message_length', 100 * 1024 * 1024),
                ('grpc.max_receive_message_length', 100 * 1024 * 1024)
            ])
            self.rpc_stub = SwitchmlIOStub(channel)

    def receive_async(self, node, round_id, total_packet_num, worker_number):
        # type: (Node, int, int, int) -> Job
        key: tuple = (round_id, node.options['node_id'])
        job = Job(key, total_packet_num, worker_number)
        self.rx_jobs[key] = job
        return job

    def receive(self, node, round_id, total_packet_num):
        # type: (Node, int, int) -> list
        """
        - node: 接收来源
        - round_id: 接收的轮次 id
        - total_packet_num: 当前任务接收包数量

        返回收到的 packet list
        """
        print("开始接收")
        if node.type == "switch":
            job = self.receive_async(
                node, round_id, total_packet_num, len(node.children))
            for child_node_id in node.children.keys():
                print("register job for child=%d" % (child_node_id))
                self.rx_jobs[(round_id, child_node_id)] = job
        else:
            job = self.receive_async(node, round_id, total_packet_num, 1)

        job.wait_until_job_finish()

        received = job.bitmap.sum()
        total = job.bitmap.size
        print("receive %d packet, expect %d, loss %f %%" %
              (received, total, 100 * (total - received) / total))
        key: tuple = (round_id, node.options['node_id'])
        del self.rx_jobs[key]
        if node.type == "switch":
            for child_node_id in node.children.keys():
                del self.rx_jobs[(round_id, child_node_id)]
        return job.buffer

    def add_child(self, node):
        # type: (Node) -> None
        self.children[node.options['node_id']] = node

    def remove_child(self, node) -> bool:
        # type: (Node) -> bool
        if self.children.get(node.options["node_id"]) is None:
            return False
        del self.children[node.options["node_id"]]
        return True

    # 向这个节点重传数据
    # 将会触发接收任务结束
    def rpc_retranmission(self, round_id, node_id, data):
        # type: (int, int, dict[int,  str]) -> None
        self.rpc_stub.Retransmission(
            Retransmission.Request(
                round_id=round_id,
                node_id=node_id,
                data=data
            )
        )

    # 获取这个节点的丢包状态
    def rpc_read_missing_slice(self, round_id, node_id):
        # type: (int, int) -> list
        return self.rpc_stub.ReadMissingSlice(
            PacketLoss.Request(
                round_id=round_id,
                node_id=node_id
            )
        )

    def check_and_retransmit(self, node, round_id, packet_list):
        # type: (Node, int, list)->int
        retransmit_start = time.time()
        missing_slice = node.rpc_stub.ReadMissingSlice(PacketLoss.Request(
            round_id=round_id, node_id=self.options['node_id'], max_segment_id=len(packet_list)-1)).missing_packet_list
        payload = []
        for segment_id in missing_slice:
            payload.append(bytes(packet_list[segment_id].buffer))
        node.rpc_stub.Retransmission(Retransmission.Request(
            round_id=round_id, node_id=self.options['node_id'], data=payload))
        retransmit_end = time.time()
        return retransmit_end - retransmit_start

    def receive_thread(self) -> None:
        while True:
            pkt = Packet()
            _, client = self.rx_sock.recvfrom_into(pkt.buffer, pkt_size)
            pkt.parse_header()
            pkt.parse_payload()
            key: tuple = (pkt.round_id, pkt.node_id)
            job = self.rx_jobs.get(key)
            if job is None:
                print("WARNING: receive job not exist! round_id:%d node_id:%d" % (
                    pkt.round_id, pkt.node_id))
                continue
            job.handle_packet(pkt)
            # if pkt.aggregate_num == 1:
            #     # 聚合数不等于 1 通常是 switch 发出，不需要 ack
            #     self.rx_sock.sendto(pkt.gen_ack_packet(), client)
            if self.type == "server":
                self.rx_sock.sendto(pkt.gen_ack_packet(), client)

    def create_packet(self, round_id: int, segment_id: int, group_id: int, bypass: bool, data: np.ndarray, multicast: bool = False):
        """
        - round_id: 轮次 id 可以认为一次 send 是一次轮次
        - segment_id (packet_id): 在当前轮次中包 id
        - node_id: 发送方 node_id
        - group_id: 分组号，用于剪枝
        - bypass: 是否禁用 switch 聚合
        - data: 有效数据，必须是长度为 256 的 float32 一维数组

        创建的包可以直接发送，不推荐手动操作数据
        """
        pkt = Packet()
        flow_control = 0
        if bypass:
            flow_control |= bypass_bitmap
        if multicast:
            flow_control |= multicast_bitmap

        pkt.set_header(
            flow_control=flow_control,
            data_type=DataType.FLOAT32.value,
            round_id=round_id,
            segment_id=segment_id,
            node_id=self.options['node_id'],
            aggregate_num=1,
            mcast_grp=group_id,
            pool_id=segment_id % switch_pool_size
        )
        pkt.deparse_header()
        pkt.set_tensor(data)
        pkt.deparse_payload()
        return pkt
