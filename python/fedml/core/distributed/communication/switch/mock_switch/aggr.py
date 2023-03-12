import numpy as np
from packet import Packet

class Aggr:
    def __init__(self, id: int, switch) -> None:
        self.id = id
        self.switch = switch
        self.reset()

    def reset(self):
        self.data = np.zeros(256, dtype=np.int32)
        self.round_id = -1
        self.segment_id = -1
        self.is_busy = False
        self.aggregate_count = 0
        self.bitmap = 0

    def acquire(self, pkt: Packet):
        self.is_busy = True
        self.bitmap = 0
        self.aggregate_count = 0
        self.round_id = pkt.round_id
        self.segment_id = pkt.segment_id

    def release(self):
        self.is_busy = False
        self.aggregate_count = 0
        self.bitmap = 0

    def aggregate(self, pkt: Packet):
        """
        return 0: drop
        return 1: 发出聚合包
        return 2: ack
        """
        aggregate_finish_bitmap = self.switch.groups[pkt.mcast_grp].aggregate_finish_bitmap
        node_bitmap = self.switch.nodes[pkt.node_id].bitmap
        if pkt.round_id > self.round_id or (pkt.round_id == self.round_id and pkt.segment_id > self.segment_id):
            if self.is_busy:
                # swap
                temp = self.data
                self.data = pkt.tensor
                pkt.set_tensor(temp)
                pkt.round_id, self.round_id = self.round_id, pkt.round_id
                pkt.segment_id, self.segment_id = self.segment_id, pkt.segment_id
                self.bitmap = node_bitmap
                pkt.aggregate_num, self.aggregate_count = self.aggregate_count, pkt.aggregate_num
                return 1
            else:
                self.acquire(pkt)
                self.aggregate_count = pkt.aggregate_num
                self.data = pkt.tensor
                self.bitmap = node_bitmap
                if aggregate_finish_bitmap == node_bitmap:
                    self.release()
                    # 不需要对包里的 aggregate_num 做操作
                    return 1
                else:
                    self.bitmap |= node_bitmap
                    return 0
        if pkt.round_id == self.round_id and pkt.segment_id == self.segment_id:
            if not self.is_busy or self.bitmap & node_bitmap > 0:
                return 2
            self.bitmap |= node_bitmap
            self.aggregate_count += 1

            self.data += pkt.tensor

            if self.bitmap == aggregate_finish_bitmap:
                pkt.aggregate_num = self.aggregate_count
                pkt.tensor = self.data
                self.release()
                return 1
            else:
                return 0
        else:
            return 2
