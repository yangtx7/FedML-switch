import os
import pickle
import threading
from concurrent import futures
from typing import List
import copy

import grpc
from ..grpc import grpc_comm_manager_pb2_grpc, grpc_comm_manager_pb2

lock = threading.Lock()

from time import sleep
from .....cross_silo.server.message_define import MyMessage
from ...communication.base_com_manager import BaseCommunicationManager
from ...communication.message import Message
from ...communication.observer import Observer
from ..constants import CommunicationConstants
from fedml.core.mlops.mlops_profiler_event import MLOpsProfilerEvent
import time
from ...communication.grpc.grpc_server import GRPCCOMMServicer

import logging
import csv
import torch
import numpy as np
import queue

from .python_io.server import Server
from .python_io.client import Client
from .python_io.switch import Switch
from .python_io.node import Node

import yaml

class SWITCHCommManager(BaseCommunicationManager):
    def __init__(
        self,
        host,
        port,
        ip_config_path,
        topic="fedml",
        client_id=0,
        client_num=0,
        model=None
    ):
        # host is the ip address of server
        self.host = host
        self.port = str(port)
        self._topic = topic
        self.client_id = client_id
        self.client_num = client_num
        self._observers: List[Observer] = []
        self.rank = client_id
        self.round_number = 1

        with open("./config/SwitchFL_config.yaml", 'r') as stream:
            self.config = yaml.safe_load(stream)

        if self.client_id == 0:
            self.server_init_connection()
        else:
            self.client_init_connection()
       
        self.opts = [
            ("grpc.max_send_message_length", 1000 * 1024 * 1024),
            ("grpc.max_receive_message_length", 1000 * 1024 * 1024),
            ("grpc.enable_http_proxy", 0),
        ]
        self.grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=client_num),
            options=self.opts,
        )
        self.grpc_servicer = GRPCCOMMServicer(host, port, client_num, client_id)
        grpc_comm_manager_pb2_grpc.add_gRPCCommManagerServicer_to_server(
            self.grpc_servicer, self.grpc_server
        )
        logging.info(os.getcwd())
        self.ip_config = self._build_ip_table(ip_config_path)

        # starts a grpc_server on local machine using ip address "0.0.0.0"
        self.grpc_server.add_insecure_port("{}:{}".format("0.0.0.0", port))

        self.grpc_server.start()
        self.is_running = True
        logging.info("grpc server started. Listening on port " + str(port))

    def server_init_connection(self):
        self.node_type = "server"
        self.recv_cnt = 0

        if self.config["EnableSwitch"] == 0:
            self.recv_thread = [None for i in range(self.client_num + 1)]
            self.recv_queue = [queue.Queue() for i in range(self.client_num + 1)]
            self.server = [None]
            self.client = [None]

            for i in range(1, self.client_num+1):
                self.server.append(Server(self.config["ServerIPAddr"], 
                                          self.config["CommLibBasePort"]+0+6*i, 
                                          self.config["CommLibBasePort"]+1+6*i, 
                                          "127.0.0.1:"+str(self.config["CommLibBasePort"]+1+6*i), 
                                          0, False, 
                                          "lo"))
                self.client.append(Client(self.config["ClientIPAddr"][i-1], 
                                          self.config["CommLibBasePort"]+3+6*i, 
                                          self.config["CommLibBasePort"]+4+6*i, 
                                          "127.0.0.1:"+str(self.config["CommLibBasePort"]+0+6*i), 
                                          i, True))
        else:
            # TODO : by default EnableSwitch=2, for EnableSwitch=1 need to be completed
            # some counters used in send/recv logic
            self.switchsend = [0 for i in range(self.config["SwitchNum"])]
            self.switchsend2 = [0 for i in range(self.config["SwitchNum"])]
            self.switchrecv = [0 for i in range(self.config["SwitchNum"])]
            self.switchtot = [0 for i in range(self.config["SwitchNum"])]

            self.switchmsgx = [None for i in range(self.config["SwitchNum"])]
            self.switchroundnumber = [0 for i in range(self.config["SwitchNum"])]

            for i in range(1, self.client_num+1):
                self.switchtot[self.config["NetworkTopo"][i-1]] += 1

            self.switch : List[Switch] = []
            self.server : List[Server] = []

            self.recv_queue = [queue.Queue() for i in range(self.config["SwitchNum"])]
            self.recv_thread = [None for i in range(self.config["SwitchNum"])]

            self.client = [None for i in range(self.client_num + 1)]

            for i in range(self.config["SwitchNum"]):
                self.switch.append(Switch(
                    node_id=100+i, # plus 100 to avoid ID conflict
                    ip_addr="127.0.0.1",
                    rx_port=self.config["SwitchPort"][i],
                    tx_port=self.config["SwitchPort"][i],
                    rpc_addr=""
                ))
                self.server.append(Server(
                    node_id=200+i, # plus 200 to avoid ID conflict
                    ip_addr="127.0.0.1",
                    rx_port=self.config["CommLibBasePort"]+0+6*i,
                    tx_port=self.config["CommLibBasePort"]+1+6*i,
                    rpc_addr="127.0.0.1:"+str(self.config["CommLibBasePort"]+2+6*i),
                    is_remote_node=False,
                    iface="lo"                    
                ))

            for i in range(1, self.client_num + 1):
                self.client[i] = Client(ip_addr=self.config["ClientIPAddr"][i-1], 
                                        rx_port=self.config["CommLibBasePort"]+3+6*i, 
                                        tx_port=self.config["CommLibBasePort"]+4+6*i, 
                                        rpc_addr="127.0.0.1:"+str(self.config["CommLibBasePort"]+5+6*i), 
                                        node_id=i,
                                        is_remote_node=True)

                self.switch[self.config["NetworkTopo"][i-1]].add_child(self.client[i])         

                
    def client_init_connection(self):
        self.node_type = "client"
        self.recv_thread = 0
        self.recv_queue = queue.Queue()
        i = self.client_id

        # Establish CommLib links
        if self.config["EnableSwitch"] == 0:
            self.server = Server(self.config["ServerIPAddr"], 
                                self.config["CommLibBasePort"]+0+6*i, 
                                self.config["CommLibBasePort"]+1+6*i, 
                                "127.0.0.1:"+str(self.config["CommLibBasePort"]+1+6*i), 
                                0, True)
            self.client = Client(self.config["ClientIPAddr"][i-1], 
                                self.config["CommLibBasePort"]+3+6*i, 
                                self.config["CommLibBasePort"]+4+6*i, 
                                "127.0.0.1:"+str(self.config["CommLibBasePort"]+0+6*i), 
                                i, False, 
                                "lo")
            
        if self.config["EnableSwitch"] == 1:
            self.server = Server(self.config["ServerIPAddr"], 
                                self.config["CommLibBasePort"]+0+6*i, 
                                self.config["CommLibBasePort"]+1+6*i, 
                                "127.0.0.1:"+str(self.config["CommLibBasePort"]+1+6*i), 
                                0, True)
            self.client = Client(self.config["ClientIPAddr"][i-1], 
                                self.config["CommLibBasePort"]+0+6*i, 
                                self.config["CommLibBasePort"]+1+6*i, 
                                "127.0.0.1:"+str(self.config["CommLibBasePort"]+0+6*i), 
                                i, False, 
                                self.config["ClientSwitchIFace"][i-1])
            
        if self.config["EnableSwitch"] == 2:
            self.server = Server(self.config["ServerIPAddr"], 
                                self.config["CommLibBasePort"]+0+6*self.config["NetworkTopo"][self.client_id-1], 
                                self.config["CommLibBasePort"]+1+6*self.config["NetworkTopo"][self.client_id-1], 
                                "127.0.0.1:"+str(self.config["CommLibBasePort"]+2+6*self.config["NetworkTopo"][self.client_id-1]), 
                                200+self.config["NetworkTopo"][self.client_id-1], True)
            self.server2 = Server(self.config["ServerIPAddr"], 
                                self.config["SwitchPort"][self.config["NetworkTopo"][self.client_id-1]], 
                                self.config["CommLibBasePort"]+1+6*self.config["NetworkTopo"][self.client_id-1], 
                                "127.0.0.1:"+str(self.config["CommLibBasePort"]+2+6*self.config["NetworkTopo"][self.client_id-1]), 
                                200+self.config["NetworkTopo"][self.client_id-1], True)
            self.client = Client(self.config["ClientIPAddr"][i-1], 
                                self.config["CommLibBasePort"]+3+6*i, 
                                self.config["CommLibBasePort"]+4+6*i, 
                                "127.0.0.1:"+str(self.config["CommLibBasePort"]+5+6*i), 
                                i, False, 
                                "lo")


    def recv_tensor(self, typ: int, pkt_num: int, server: Node, client: Node, msg_q: queue.Queue):
        if typ == 0:
            pkt_list = server.receive(client, self.round_number, pkt_num)
        else:
            pkt_list = client.receive(server, self.round_number, pkt_num)
        flg = 0
        for i in range(pkt_num):
            if flg == 0:
                flg = 1
                msgx = np.array(pkt_list[i].tensor/pkt_list[i].aggregate_num)
            else:
                msgx = np.concatenate((msgx, np.array(pkt_list[i].tensor/pkt_list[i].aggregate_num)))
        msg_q.put(msgx)

    def send_message(self, msg: Message):
        receiver_id = msg.get_receiver_id()
        PORT_BASE = CommunicationConstants.GRPC_BASE_PORT
        # lookup ip of receiver from self.ip_config table
        receiver_ip = self.ip_config[str(receiver_id)]
        channel_url = "{}:{}".format(receiver_ip, str(PORT_BASE + receiver_id))

        msg2 = copy.deepcopy(msg)

        active_commlib = 0

        if self.node_type == "server" and (msg2.type == "1" or msg2.type == "2"):
            if self.config["EnableSwitch"] == 0:
                active_commlib = 1
            else:
                self.switchsend[self.config["NetworkTopo"][receiver_id-1]] += 1
                if self.switchsend[self.config["NetworkTopo"][receiver_id-1]] == self.switchtot[self.config["NetworkTopo"][receiver_id-1]]:
                    self.switchsend[self.config["NetworkTopo"][receiver_id-1]] = 0
                    active_commlib = 1

        if self.node_type == "client" and msg2.type == "3":
            active_commlib = 1

        # TODO : This part can be optimized
        if (self.node_type == "server" and (msg2.type == "1" or msg2.type == "2")) or (self.node_type == "client" and msg2.type == "3"):
            msgx = self.extract_tensor(msg2)


        logging.info("msg2 = {}".format(msg2))
        logging.info("pickle.dumps(msg2) START")
        pickle_dump_start_time = time.time()
        msg_pkl = pickle.dumps(msg2)
        MLOpsProfilerEvent.log_to_wandb({"PickleDumpsTime": time.time() - pickle_dump_start_time})
        logging.info("pickle.dumps(msg2) END")

        channel = grpc.insecure_channel(channel_url, options=self.opts)
        stub = grpc_comm_manager_pb2_grpc.gRPCCommManagerStub(channel)
        request = grpc_comm_manager_pb2.CommRequest()
        logging.info("sending message to {}".format(channel_url))

        request.client_id = self.client_id
        request.message = msg_pkl
        tick = time.time()
        stub.sendMessage(request)

        MLOpsProfilerEvent.log_to_wandb({"Comm/send_delay": time.time() - tick})
        logging.debug("sent successfully")
        channel.close()

        # SEND
        if active_commlib == 1:
            self.send_tensor(msg2, msgx)
            if self.node_type == "server":
                self.wait_for_recv(msg2)
        if self.node_type == "client" and msg2.type == "3":
            print("round number", self.round_number, "->", self.round_number+1)
            self.round_number += 1
        
            


    def extract_tensor(self, msg2: Message):
        flg = 0
        print("!!!SEND!!!")

        for it in msg2.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS]:
            print(it, msg2.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS][it].size())
            if flg == 0:
                flg = 1
                if msg2.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS][it].get_device() == 0:
                    msgx = np.float32(msg2.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS][it].flatten().cpu().numpy())
                    
                else:
                    msgx = np.float32(msg2.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS][it].flatten().numpy())
            else:
                if msg2.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS][it].get_device() == 0:
                    msgx = np.concatenate((msgx, np.float32(msg2.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS][it].flatten().cpu().numpy())))
                else:
                    msgx = np.concatenate((msgx, np.float32(msg2.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS][it].flatten().numpy())))
            msg2.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS][it] = msg2.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS][it].size()

        # padding zeros
        app = 256 - (np.size(msgx) % 256)
        msgx = np.concatenate((msgx, np.zeros(shape=app)))
        msg2.pkt_num = np.size(msgx) // 256

        
        print(">>> LENGTH of parameter: ", np.size(msgx))
        if (self.node_type == "server" and self.config["EnableSwitch"] == 1):
            print(">>> COMM_LIB SEND (switch):", msg2.get_sender_id(), "->", msg2.get_receiver_id())
        else:
            print(">>> COMM_LIB SEND :", msg2.get_sender_id(), "->", msg2.get_receiver_id())    

        return msgx
    
    def send_tensor(self, msg2: Message, msgx):
        receiver_id = msg2.get_receiver_id()
        pkt_list = []
        sleep(0.1)
        
        if self.node_type == "server":
            if self.config["EnableSwitch"] == 0:
                for i in range(msg2.pkt_num):
                    pkt_list.append(self.server[receiver_id].create_packet(self.round_number, i, 1, True, msgx[256*i:256*(i+1)]))
                self.server[receiver_id].send(self.client[receiver_id], self.round_number, pkt_list)
            else:
                for i in range(msg2.pkt_num):
                    pkt_list.append(self.server[self.config["NetworkTopo"][receiver_id-1]].create_packet(self.round_number, i, 1, False, msgx[256*i:256*(i+1)], True))
                self.server[self.config["NetworkTopo"][receiver_id-1]].send(self.switch[self.config["NetworkTopo"][receiver_id-1]], self.round_number, pkt_list)

            

        if self.node_type == "client":   
            # For client it doesn't need multicast, even if it is connected to a switch 

            if self.config["EnableSwitch"] == 0:
                for i in range(msg2.pkt_num):
                    pkt_list.append(self.client.create_packet(self.round_number, i, 1, True, msgx[256*i:256*(i+1)]))
                self.client.send(self.server, self.round_number, pkt_list, False)
            elif self.config["EnableSwitch"] == 1:
                for i in range(msg2.pkt_num):
                    pkt_list.append(self.client.create_packet(self.round_number, i, 1, False, msgx[256*i:256*(i+1)]))
                self.client.send(self.server, self.round_number, pkt_list, True)
            else:
                for i in range(msg2.pkt_num):
                    pkt_list.append(self.client.create_packet(self.round_number, i, 1, False, msgx[256*i:256*(i+1)]))
                self.client.send(self.server2, self.round_number, pkt_list, True) 

    # This function is for server receive tensor from client (or switch)
    def wait_for_recv(self, msg2: Message):
        receiver_id = msg2.get_receiver_id()

        enable_wait_recv = 1

        # if self.config["EnableSwitch"] == 0:
        #     enable_wait_recv = 1
        # else:
        #     if self.switchsend2[self.config["NetworkTopo"][receiver_id-1]] == 0:
        #         enable_wait_recv = 1
        #     self.switchsend2[self.config["NetworkTopo"][receiver_id-1]] += 1
        #     if self.switchsend2[self.config["NetworkTopo"][receiver_id-1]] == self.switchtot[self.config["NetworkTopo"][receiver_id-1]]:
        #         self.switchsend2[self.config["NetworkTopo"][receiver_id-1]] = 0

        if enable_wait_recv == 1:
            print(">>>COMMLIB RECV :", msg2.get_receiver_id(), "->", msg2.get_sender_id())

            if self.config["EnableSwitch"] == 0:
                self.recv_thread[msg2.get_receiver_id()] = threading.Thread(
                target=self.recv_tensor, 
                args=(0,
                    msg2.pkt_num, 
                    self.server[msg2.get_receiver_id()], 
                    self.client[msg2.get_receiver_id()], 
                    self.recv_queue[msg2.get_receiver_id()]))
                self.recv_thread[msg2.get_receiver_id()].start()
            else: 
                self.recv_thread[self.config["NetworkTopo"][msg2.get_receiver_id()-1]] = threading.Thread(
                target=self.recv_tensor, 
                args=(0,
                    msg2.pkt_num, 
                    self.server[self.config["NetworkTopo"][msg2.get_receiver_id()-1]], 
                    self.switch[self.config["NetworkTopo"][msg2.get_receiver_id()-1]], 
                    self.recv_queue[self.config["NetworkTopo"][msg2.get_receiver_id()-1]]))
                self.recv_thread[self.config["NetworkTopo"][msg2.get_receiver_id()-1]].start()

    def add_observer(self, observer: Observer):
        self._observers.append(observer)

    def remove_observer(self, observer: Observer):
        self._observers.remove(observer)

    def handle_receive_message(self):
        self._notify_connection_ready()
        self.message_handling_subroutine()

    def message_handling_subroutine(self):
        start_listening_time = time.time()
        MLOpsProfilerEvent.log_to_wandb({"ListenStart": start_listening_time})
        while self.is_running:
            if self.grpc_servicer.message_q.qsize() > 0:
                lock.acquire()
                busy_time_start_time = time.time()
                msg_pkl = self.grpc_servicer.message_q.get()
                logging.info("unpickle START")
                unpickle_start_time = time.time()
                msg = pickle.loads(msg_pkl)

                active_commlib = 0

                # only client need to start the receiving thread (server has started before)

                if self.node_type == "client" and (msg.type == "1" or msg.type == "2"):
                    print(">>>COMMLIB RECV :", msg.get_sender_id(), "->", msg.get_receiver_id())
                    self.recv_thread = threading.Thread(target=self.recv_tensor, args=(1, msg.pkt_num, self.server, self.client, self.recv_queue))
                    self.recv_thread.start()  
                    
                    active_commlib = 1
                    self.recv_thread.join()
                    msgx = self.recv_queue.get()

                if self.node_type == "server" and msg.type == "3":
                    active_commlib = 1
                    actual_recv = 0 

                    if self.config["EnableSwitch"] == 0:
                        self.recv_thread[msg.get_sender_id()].join()
                        msgx = self.recv_queue[msg.get_sender_id()].get()
                    else:
                        
                        if self.switchrecv[self.config["NetworkTopo"][msg.get_sender_id()-1]] == 0:

                            self.recv_thread[self.config["NetworkTopo"][msg.get_sender_id()-1]].join()

                            # TODO : need to * 2
                            msgx = self.recv_queue[self.config["NetworkTopo"][msg.get_sender_id()-1]].get()

                            if self.config["ParameterUploadStrategy"] == 0:
                                # multiple the parameters by the number of clients that switch has
                                msgx *= self.switchtot[self.config["NetworkTopo"][msg.get_sender_id()-1]]
                            else:
                                self.switchmsgx[self.config["NetworkTopo"][msg.get_sender_id()-1]] = msgx
                                self.switchroundnumber[self.config["NetworkTopo"][msg.get_sender_id()-1]] += 1
                            
                            # TODO : for pruning case, it must mutliply some coefficient...
                            # msgx *= self.switchtot[self.config["NetworkTopo"][msg.get_sender_id()-1]]
                        
                        else:
                            if self.config["ParameterUploadStrategy"] == 0:
                                msgx = np.zeros(msg.pkt_num * 256)
                            else:
                                while (1):
                                    sleep(0.1)
                                    if self.switchroundnumber[self.config["NetworkTopo"][msg.get_sender_id()-1]] == self.round_number:
                                        break
                                msgx = self.switchmsgx[self.config["NetworkTopo"][msg.get_sender_id()-1]]

                            # generate a packet padded by zeros to imitate a normal msgx.
                            # TODO : need to modify
                            
                            

                        self.switchrecv[self.config["NetworkTopo"][msg.get_sender_id()-1]] += 1
                        if self.switchrecv[self.config["NetworkTopo"][msg.get_sender_id()-1]] == self.switchtot[self.config["NetworkTopo"][msg.get_sender_id()-1]]:
                            self.switchrecv[self.config["NetworkTopo"][msg.get_sender_id()-1]] = 0

                    self.recv_cnt += 1
                    print("!!!")
                    print("recv_cnt =", self.recv_cnt)
                    if self.recv_cnt == self.config["ClientNum"]:
                        self.recv_cnt = 0
                        print("round number", self.round_number, "->", self.round_number+1)
                        self.round_number += 1

                if active_commlib == 1:
                    self.embed_tensor(msg, msgx)

                MLOpsProfilerEvent.log_to_wandb({"UnpickleTime": time.time() - unpickle_start_time})
                logging.info("unpickle END")
                msg_type = msg.get_type()
                for observer in self._observers:
                    _message_handler_start_time = time.time()
                    observer.receive_message(msg_type, msg)
                    MLOpsProfilerEvent.log_to_wandb({"MessageHandlerTime": time.time() - _message_handler_start_time})
                MLOpsProfilerEvent.log_to_wandb({"BusyTime": time.time() - busy_time_start_time})


                lock.release()
            time.sleep(0.0001)
        MLOpsProfilerEvent.log_to_wandb({"TotalTime": time.time() - start_listening_time})
        return

    def embed_tensor(self, msg: Message, msgx):
        msgx = np.float64(msgx) # convert to float
        cul = 0
        print("!!!RECV!!!")
        for it in msg.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS]:
            cur = 1
            for i in range(len(msg.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS][it])):
                cur *= msg.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS][it][i]
            tmp = torch.from_numpy(msgx[cul:cul+cur])
            cul += cur
            tmp = torch.reshape(tmp, msg.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS][it])
            msg.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS][it] = tmp
            print(it, msg.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS][it].size())

    def stop_receive_message(self):
        self.grpc_server.stop(None)
        self.is_running = False

    def notify(self, message: Message):
        msg_type = message.get_type()
        for observer in self._observers:
            observer.receive_message(msg_type, message)

    def _notify_connection_ready(self):
        msg_params = Message()
        msg_params.sender_id = self.rank
        msg_params.receiver_id = self.rank
        msg_type = CommunicationConstants.MSG_TYPE_CONNECTION_IS_READY
        for observer in self._observers:
            observer.receive_message(msg_type, msg_params)

    def _build_ip_table(self, path):
        ip_config = dict()
        with open(path, newline="") as csv_file:
            csv_reader = csv.reader(csv_file)
            # skip header line
            next(csv_reader)

            for row in csv_reader:
                receiver_id, receiver_ip = row
                ip_config[receiver_id] = receiver_ip
        return ip_config
