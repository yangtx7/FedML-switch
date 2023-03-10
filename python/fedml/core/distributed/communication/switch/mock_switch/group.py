from node import Node
class Group:
    def __init__(self, id: int, ps: Node) -> None:
        self.id = id
        self.nodes = {} # int => Node
        self.ps = ps
        self.aggregate_finish_bitmap = 0
        
    def addNode(self, node: Node) -> None:
        self.nodes[node.id] = node
        self.aggregate_finish_bitmap |= node.bitmap
    
    def delNode(self, node: Node) -> None:
        del self.nodes[node.id]
        self.aggregate_finish_bitmap &= ~node.bitmap