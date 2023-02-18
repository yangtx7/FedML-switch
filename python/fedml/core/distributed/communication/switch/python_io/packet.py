from enum import Enum
import struct
import numpy as np
import typing

# int32 和 float32 互转时的系数
scaling_factor = 1e8


class DataType(Enum):
    INT32 = 0
    FLOAT32 = 1


# 包头结构
# flow_control  uint8
# data_type     uint8
# pool_id       uint16
# job_id        uint32
# segment_id    uint32
# node_id       uint16
# aggregate_num uint16
# mcast_grp     uint16

header_format = ">BBHIIHHH"
header_size = struct.calcsize(header_format)

# packer param
# elemenet_per_packet = 2048  # MTU 9000
elemenet_per_packet = 256  # MTU 1100
switch_pool_size = 16
pkt_size = elemenet_per_packet * 4 + header_size


# flow control
# |  0  |  1  |   2    |     3     |       4        | 5 | 6 | 7 |
# | ack | ecn | bypass | multicast | retranmission  |   |   |   |
ack_bitmap = 1 << 7
ecn_bitmap = 1 << 6
bypass_bitmap = 1 << 5
multicast_bitmap = 1 << 4
retranmission_bitmap = 1 << 3


class Packet:
    def __init__(self, buffer: bytearray = None) -> None:
        self.buffer = buffer if buffer is not None else bytearray(pkt_size)

        self.flow_control = 0
        self.data_type = DataType.FLOAT32.value
        self.job_id = 0
        self.segment_id = 0
        self.node_id = 0
        self.aggregate_num = 0
        self.mcast_grp = 0
        self.pool_id = 0

        # flow control
        self.ecn = 0
        self.ack = 0
        self.bypass = 0

        self.tensor: typing.Union[np.ndarray, None] = None

    def set_header(self, flow_control: int, data_type: int, job_id: int, segment_id: int, node_id: int, aggregate_num: int, mcast_grp: int, pool_id: int):
        self.flow_control = flow_control
        self.job_id = job_id
        self.segment_id = segment_id
        self.node_id = node_id
        self.aggregate_num = aggregate_num
        self.mcast_grp = mcast_grp
        self.data_type = data_type
        self.ecn = flow_control & ecn_bitmap
        self.bypass = flow_control & bypass_bitmap
        self.ack = flow_control & ack_bitmap
        self.pool_id = pool_id

    # 必须是 float 数组且 shape: (elemenet_per_packet)
    def set_tensor(self, tensor: np.ndarray):
        self.tensor = tensor

    def parse_header(self):
        header_val = struct.unpack_from(header_format, self.buffer)
        self.set_header(
            flow_control=header_val[0],
            data_type=header_val[1],
            pool_id=header_val[2],
            job_id=header_val[3],
            segment_id=header_val[4],
            node_id=header_val[5],
            aggregate_num=header_val[6],
            mcast_grp=header_val[7],
        )
        return

    def deparse_header(self):
        struct.pack_into(
            header_format,
            self.buffer,
            0,
            self.flow_control,
            self.data_type,
            self.pool_id,
            self.job_id,
            self.segment_id,
            self.node_id,
            self.aggregate_num,
            self.mcast_grp
        )
        return

    def parse_payload(self):
        self.set_tensor(np.frombuffer(
            self.buffer,
            dtype=np.int32,
            offset=header_size
        ))
        if self.data_type == DataType.FLOAT32.value:
            self.tensor = self.tensor.astype(np.float32)
            self.tensor /= scaling_factor
        return

    # 将 tensor 写入 buffer
    def deparse_payload(self):
        if self.data_type == DataType.FLOAT32.value:
            self.buffer[header_size: pkt_size] = (
                self.tensor * scaling_factor).astype(np.int32).tobytes()
        else:
            self.buffer[header_size: pkt_size] = self.tensor.tobytes()

    def gen_ack_packet(self):
        return struct.pack(
            header_format,
            self.flow_control | ack_bitmap | bypass_bitmap,
            self.data_type,
            self.pool_id,
            self.job_id,
            self.segment_id,
            self.node_id,
            self.aggregate_num,
            self.mcast_grp
        )
