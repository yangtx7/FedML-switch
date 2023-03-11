from switch import Switch
from node import Node
from group import Group

node_num = 2
mock_switch = Switch("127.0.0.1", 30000, 100, debug=True)
ps = Node(200, "127.0.0.1", 50000, 50001, 0)
mock_switch.nodes[ps.id] = ps
group = Group(1, ps)
mock_switch.groups[group.id] = group

for i in range(node_num):
    node = Node(i+1, "127.0.0.1", 50009+i*6, 50010+i*6 , 1 << i)
    group.addNode(node)
    mock_switch.nodes[node.id] = node

mock_switch.start()
