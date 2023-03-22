from concurrent import futures
import io_pb2_grpc
from google.protobuf.empty_pb2 import Empty
from io_pb2 import Retransmission, PacketLoss
import numpy as np
import grpc
import io_pb2_grpc
import threading
from packet import *


class SwitchmlIOServicer(io_pb2_grpc.SwitchmlIOServicer):
    def __init__(self, node) -> None:
        super().__init__()
        self.node = node

    def Retransmission(self, request: Retransmission.Request, context):
        """
        重传的同时携带元信息，减少 1 RTT 的元信息传输时间
        """
        job = self.node.rx_jobs.get((request.round_id, request.node_id))
        for slice in request.data:
            pkt = Packet()
            pkt.buffer = slice
            pkt.parse_header()
            pkt.parse_payload()
            job.handle_retransmission_packet(pkt)
        job.handle_meta(request.meta)
        job.finish()
        return Empty()

    def ReadMissingSlice(self, request: PacketLoss.Request, context):
        job = self.node.rx_jobs.get((request.round_id, request.node_id))
        if job is None:
            raise "对方的接收 job 已经结束，调用丢包检测时机有误，或者对端提前结束了接收"
        return PacketLoss.Response(missing_packet_list=job.read_missing_slice(request.max_segment_id))


class GrpcServer:
    def __init__(self, node):
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024)
        ])
        io_pb2_grpc.add_SwitchmlIOServicer_to_server(
            SwitchmlIOServicer(node),
            self.server
        )
        self.server.add_insecure_port(node.options["rpc_addr"])
        self.server.start()
        threading.Thread(target=self.server.wait_for_termination).start()

    def stop(self):
        self.server.stop(grace=None)
