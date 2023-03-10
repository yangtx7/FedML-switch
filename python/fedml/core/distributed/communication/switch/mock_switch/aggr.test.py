from switch import Switch
from group import Group
from node import Node
from aggr import Aggr
from packet import Packet, DataType
import numpy as np


ps = Node(id=10, ip_addr="127.0.0.1", rx_port=50000, tx_port=50001, bitmap=0)
node = Node(id=1, ip_addr="127.0.0.1", rx_port=50003, tx_port=50004, bitmap=1)
node2 = Node(id=2, ip_addr="127.0.0.1", rx_port=50003, tx_port=50004, bitmap=2)
group = Group(id=1, ps=ps)
group.addNode(node)
group.addNode(node2)

switch = Switch("127.0.0.1", 30000, 9)
switch.groups[group.id] = group
switch.nodes[node.id] = node
switch.nodes[node2.id] = node2
switch.nodes[ps.id] = ps

pkt = Packet()
pkt.set_header(
    flow_control=0,
    data_type=DataType.FLOAT32.value,
    round_id=1,
    segment_id=0,
    node_id=1,
    aggregate_num=1,
    mcast_grp=1,
    pool_id=1
)
pkt.set_tensor(np.ones((256)))
pkt.deparse_payload()

aggr: Aggr = switch.aggrs[1]
# node = 1 seg = 1
pkt.node_id = 1
pkt.segment_id = 1
pkt.deparse_header()
act = aggr.aggregate(pkt)

# node = 2 seg = 1
pkt.node_id = 2
pkt.segment_id = 1
pkt.deparse_header()
act = aggr.aggregate(pkt)

# node = 1 seg = 2
pkt.node_id = 1
pkt.segment_id = 2
pkt.deparse_header()
act = aggr.aggregate(pkt)

# node = 2 seg = 2
pkt.node_id = 2
pkt.segment_id = 2
pkt.deparse_header()
act = aggr.aggregate(pkt)

# node = 1 seg = 1
pkt.node_id = 1
pkt.segment_id = 2
pkt.deparse_header()
act = aggr.aggregate(pkt)

print(pkt)
