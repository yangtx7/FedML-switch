from group import Group
from node import Node
from aggr import Aggr
from packet import Packet, multicast_bitmap, pkt_size, ack_bitmap
import socket


class Switch:
    def __init__(self, addr: str, port: int, node_id: int, debug = False) -> None:
        self.init_aggrs()
        self.nodes = {}  # id => Node
        self.groups = {}  # id => Group
        self.addr = addr
        self.port = port
        self.node_id = node_id
        self.debug = debug

    def init_aggrs(self):
        self.aggrs = {}  # id => Aggr
        for i in range(128):
            self.aggrs[i] = Aggr(i, self)

    def log(self, msg:str):
        if self.debug:
            print(msg)

    def start(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ep = (self.addr, self.port)
        sock.bind(ep)
        while True:
            pkt = Packet()
            sock.recvfrom_into(pkt.buffer, pkt_size)
            pkt.parse_header()
            if pkt.flow_control & ack_bitmap > 0:
                continue
            if pkt.flow_control & multicast_bitmap > 0:
                self.log("receive multicast packet, round=%d, segment=%d" % (pkt.round_id, pkt.segment_id))
                # 广播
                group: Group = self.groups[pkt.mcast_grp]
                for node in group.nodes.values():
                    self.log("multicast to node=%d" % (node.id))
                    sock.sendto(pkt.buffer, (node.ip_addr, node.rx_port))

            else:
                # 聚合
                pkt.parse_payload()
                # handle packet
                aggr: Aggr = self.aggrs[pkt.pool_id]
                act = aggr.aggregate(pkt)
                self.log("receive reduce packet, from node=%d, round=%d, segment=%d, action=%d" % (pkt.node_id, pkt.round_id, pkt.segment_id, act))

                if act == 1:
                    # 发出所有包
                    pkt.node_id = self.node_id # 聚合完成后才能将包的 node_id 重写为 switch node_id
                    group: Group = self.groups[pkt.mcast_grp]
                    ack_pkt = pkt.gen_ack_packet()
                    for node in group.nodes.values():
                        self.log("ack reduce pkt to node=%d" % (node.id))
                        sock.sendto(ack_pkt, (node.ip_addr, node.tx_port))
                    pkt.deparse_header()
                    pkt.deparse_payload()
                    self.log("proxy reduce pkt to ps=%d" % (group.ps.id))
                    sock.sendto(pkt.buffer, (group.ps.ip_addr, group.ps.rx_port))
                if act == 2:
                    # ack
                    node: Node = self.nodes[pkt.node_id]
                    ack_pkt = pkt.gen_ack_packet()
                    sock.sendto(ack_pkt, (node.ip_addr, node.tx_port))
