from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Null(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class PacketLoss(_message.Message):
    __slots__ = []
    class Request(_message.Message):
        __slots__ = ["job_id", "max_segment_id", "node_id"]
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        MAX_SEGMENT_ID_FIELD_NUMBER: _ClassVar[int]
        NODE_ID_FIELD_NUMBER: _ClassVar[int]
        job_id: int
        max_segment_id: int
        node_id: int
        def __init__(self, job_id: _Optional[int] = ..., node_id: _Optional[int] = ..., max_segment_id: _Optional[int] = ...) -> None: ...
    class Response(_message.Message):
        __slots__ = ["missing_packet_list"]
        MISSING_PACKET_LIST_FIELD_NUMBER: _ClassVar[int]
        missing_packet_list: _containers.RepeatedScalarFieldContainer[int]
        def __init__(self, missing_packet_list: _Optional[_Iterable[int]] = ...) -> None: ...
    def __init__(self) -> None: ...

class Retransmission(_message.Message):
    __slots__ = []
    class Request(_message.Message):
        __slots__ = ["data", "job_id", "node_id"]
        DATA_FIELD_NUMBER: _ClassVar[int]
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        NODE_ID_FIELD_NUMBER: _ClassVar[int]
        data: _containers.RepeatedScalarFieldContainer[bytes]
        job_id: int
        node_id: int
        def __init__(self, job_id: _Optional[int] = ..., node_id: _Optional[int] = ..., data: _Optional[_Iterable[bytes]] = ...) -> None: ...
    def __init__(self) -> None: ...
