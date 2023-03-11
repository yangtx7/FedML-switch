from packet import Packet
import numpy as np
import threading
import math

# 接收任务
class Job:
    def __init__(self, key: tuple, total_packet_num: int, worker_number: int = 1):
        """
        - key: tuple(JobId, NodeId)
        - total_packet_num: 总共接收包的个数
        - worker_number: 参与这个 job 的节点个数，当前 Job 需要等待所有节点发送完毕后才会结束
        """
        self.total_packet_num = total_packet_num
        self.buffer = [None for i in range(self.total_packet_num)]
        self.bitmap = np.zeros(shape=(self.total_packet_num), dtype=np.int8)
        # 任务完成时解锁
        self._lock = threading.Lock()
        self._lock.acquire()
        self.remain_worker_number = worker_number
        self.missing_slice_cache = None

    def finish(self):
        self.remain_worker_number -= 1
        print("一个节点完成了发送流程，剩余等待节点为 %d" % (self.remain_worker_number))
        if self.remain_worker_number == 0:
            self._lock.release()

    def wait_until_job_finish(self):
        self._lock.acquire()
        self._lock.release()

    def handle_packet(self, pkt: Packet):
        self.buffer[pkt.segment_id] = pkt
        self.bitmap[pkt.segment_id] = 1

    def handle_retransmission_packet(self, pkt: Packet):
        if self.bitmap[pkt.segment_id] == 0:
            if self.buffer[pkt.segment_id] is None:
                self.buffer[pkt.segment_id] = pkt
            else:
                self.buffer[pkt.segment_id].tensor += pkt.tensor
                self.buffer[pkt.segment_id].aggregate_num += pkt.aggregate_num
    
    def read_missing_slice(self, range_end: int = -1):
        missing_slice = np.where(self.bitmap == 0)[0]
        if range_end == -1:
            return missing_slice
        return missing_slice[missing_slice <= range_end] 