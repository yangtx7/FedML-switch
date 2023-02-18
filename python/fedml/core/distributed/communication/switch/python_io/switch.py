from node import Node
class Switch(Node):
  def __init__(self, ip_addr: str, rx_port: int, tx_port: int, rpc_addr: str, node_id: int, is_remote_node: bool, iface: str = ""):
    super().__init__(ip_addr, rx_port, tx_port, rpc_addr, node_id, is_remote_node, iface)
    self.type = "switch"