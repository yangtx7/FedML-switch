from switch import Switch
from node import Node
from group import Group

if __name__ == "__main__":
    group_num = 2
    node_config = [
        [1],
        [1,2],
    ]
    mock_switch = Switch("127.0.0.1", 30000, 101, debug=True)
    
    # 创建 ps 实例
    ps = Node(100, "127.0.0.1", 50000, 50001, 0)
    mock_switch.nodes[ps.id] = ps

    # 注册组
    groups = []
    for i in range(1, group_num+1):
        g = Group(i, ps)
        mock_switch.groups[g.id] = g

    # 创建 client 实例
    for i, groups in enumerate(node_config):
        node = Node(i+1, "127.0.0.1", 50000+(i+1)*3, 50000+(i+1)*3 + 1, 1 << i)
        for j in groups:
            mock_switch.groups[j].addNode(node)
        mock_switch.nodes[node.id] = node

    mock_switch.start()
