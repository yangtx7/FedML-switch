from switch import Switch
from node import Node
from group import Group

node_num = 2
base_port = 51500

mock_switch = Switch("127.0.0.1", 30000, 100, debug=True)
ps = Node(200, "127.0.0.1", base_port, base_port+1, 0)
mock_switch.nodes[ps.id] = ps
group = Group(1, ps)
mock_switch.groups[group.id] = group

for i in range(node_num):
    node = Node(i+1, "127.0.0.1", base_port+9+i*6, base_port+10+i*6 , 1 << i)
    group.addNode(node)
    mock_switch.nodes[node.id] = node

mock_switch.start()
