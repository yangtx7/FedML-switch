from packet import Packet
import numpy as np
import threading
import math

JOB_STATE_RUNNING = 0
JOB_STATE_RETRANSMITING = 1

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
        self.state = JOB_STATE_RUNNING
        self.missing_slice_cache = None

    def finish(self):
        self.remain_worker_number -= 1
        if self.remain_worker_number == 0:
            self._lock.release()

    def wait_until_job_finish(self):
        self._lock.acquire()
        self._lock.release()

    def handle_packet(self, pkt: Packet):
        # 进入重传阶段，应该停止接收包
        if self.state != JOB_STATE_RUNNING:
            return
        self.buffer[pkt.segment_id] = pkt
        self.bitmap[pkt.segment_id] = 1

    def handle_retransmission_packet(self, pkt: Packet):
        if self.bitmap[pkt.segment_id] == 0:
            self.buffer[pkt.segment_id] = pkt
            self.bitmap[pkt.segment_id] = 1
        else:
            self.buffer[pkt.segment_id].tensor += pkt.tensor
            self.buffer[pkt.segment_id].aggregate_num += pkt.aggregate_num
            self.bitmap[pkt.segment_id] += pkt.aggregate_num
    
    def read_missing_slice(self, range_end: int = -1):
        self.state = JOB_STATE_RETRANSMITING
        if self.missing_slice_cache is None:
            self.missing_slice_cache = np.where(self.bitmap == 0)[0]
        if range_end == -1:
            return self.missing_slice_cache
        return self.missing_slice_cache[self.missing_slice_cache <= range_end]