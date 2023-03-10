class Node:
    def __init__(self, id: int, ip_addr: str, rx_port: int, tx_port: int, bitmap: int) -> None:
        self.id = id
        self.ip_addr = ip_addr
        self.rx_port = rx_port
        self.tx_port = tx_port
        self.bitmap = bitmap
