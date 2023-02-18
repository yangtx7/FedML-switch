import os
import pickle
import threading
from concurrent import futures
from typing import List

import grpc
from ..grpc import grpc_comm_manager_pb2_grpc, grpc_comm_manager_pb2

lock = threading.Lock()


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
    ):
        # host is the ip address of server
        self.host = host
        self.port = str(port)
        self._topic = topic
        self.client_id = client_id
        self.client_num = client_num
        self._observers: List[Observer] = []
        self.rank = client_id

        with open("./config/SwitchFL_config.yaml", 'r') as stream:
            self.config = yaml.safe_load(stream)

        if client_id == 0:
            self.node_type = "server"
            self.recv_thread = [0 for i in range(self.client_num + 1)]
            self.recv_queue = [queue.Queue() for i in range(self.client_num + 1)]
            self.server = [None]
            self.client = [None]

            if self.config["EnableSwitch"] == 1:
                self.switchsend = [0 for i in range(self.config["SwitchNum"])]
                self.switchrecv = [0 for i in range(self.config["SwitchNum"])]
                self.switchtot = [0 for i in range(self.config["SwitchNum"])]
                # Calculate the number of clients each switch link with
                for i in range(1, self.client_num+1):
                    self.switchtot[self.config["NetworkTopo"][i-1]] += 1

            # Establish CommLib links
            if self.config["EnableSwitch"] == 0:
                for i in range(1, self.client_num+1):
                    self.server.append(Server(self.config["ServerIPAddr"], 
                                              self.config["CommLibBasePort"]+0+5*i, 
                                              self.config["CommLibBasePort"]+1+5*i, 
                                              "127.0.0.1:"+str(self.config["CommLibBasePort"]+1+5*i), 
                                              0, False, 
                                              "lo"))
                    self.client.append(Client(self.config["ClientIPAddr"][i-1], 
                                              self.config["CommLibBasePort"]+3+5*i, 
                                              self.config["CommLibBasePort"]+4+5*i, 
                                              "127.0.0.1:"+str(self.config["CommLibBasePort"]+0+5*i), 
                                              i, True))
            else:
                for i in range(1, self.client_num + 1):
                    self.server.append(Server(self.config["ServerIPAddr"], 
                                              self.config["CommLibBasePort"]+0+5*i, 
                                              self.config["CommLibBasePort"]+1+5*i, 
                                              "127.0.0.1:"+str(self.config["CommLibBasePort"]+1+5*i), 
                                              0, False, 
                                              self.config["ServerSwitchIFace"][self.config["NetworkTopo"][i-1]]))
                    self.client.append(Client(self.config["ClientIPAddr"][i-1], 
                                              self.config["CommLibBasePort"]+0+5*i, 
                                              self.config["CommLibBasePort"]+1+5*i, 
                                              "127.0.0.1:"+str(self.config["CommLibBasePort"]+0+5*i), 
                                              i, True))
        else:
            self.node_type = "client"

            self.recv_thread = 0
            self.recv_queue = queue.Queue()
            i = self.client_id

            # Establish CommLib links
            if self.config["EnableSwitch"] == 0:
                self.server = Server(self.config["ServerIPAddr"], 
                                    self.config["CommLibBasePort"]+0+5*i, 
                                    self.config["CommLibBasePort"]+1+5*i, 
                                    "127.0.0.1:"+str(self.config["CommLibBasePort"]+1+5*i), 
                                    0, True)
                self.client = Client(self.config["ClientIPAddr"][i-1], 
                                    self.config["CommLibBasePort"]+3+5*i, 
                                    self.config["CommLibBasePort"]+4+5*i, 
                                    "127.0.0.1:"+str(self.config["CommLibBasePort"]+0+5*i), 
                                    i, False, 
                                    "lo")
            else:
                self.server = Server(self.config["ServerIPAddr"], 
                                    self.config["CommLibBasePort"]+0+5*i, 
                                    self.config["CommLibBasePort"]+1+5*i, 
                                    "127.0.0.1:"+str(self.config["CommLibBasePort"]+1+5*i), 
                                    0, True)
                self.client = Client(self.config["ClientIPAddr"][i-1], 
                                    self.config["CommLibBasePort"]+0+5*i, 
                                    self.config["CommLibBasePort"]+1+5*i, 
                                    "127.0.0.1:"+str(self.config["CommLibBasePort"]+0+5*i), 
                                    i, False, 
                                    self.config["ClientSwitchIFace"][i-1])

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

    def recv_tensor(self, typ: int, pkt_num: int, server: Server, client :Client, msg_q: queue.Queue):
        if typ == 0:
            pkt_list = server.receive(client, 123, pkt_num)
        else:
            pkt_list = client.receive(server, 123, pkt_num)
        flg = 0
        for i in range(pkt_num):
            if flg == 0:
                flg = 1
                msgx = np.array(pkt_list[i].tensor)
            else:
                msgx = np.concatenate((msgx, np.array(pkt_list[i].tensor)))
        print(">>> For RECV DEBUG:", msgx[3])
        msg_q.put(msgx)

    def send_message(self, msg: Message):
        receiver_id = msg.get_receiver_id()
        PORT_BASE = CommunicationConstants.GRPC_BASE_PORT
        # lookup ip of receiver from self.ip_config table
        receiver_ip = self.ip_config[str(receiver_id)]
        channel_url = "{}:{}".format(receiver_ip, str(PORT_BASE + receiver_id))

        active_commlib = 0
        if self.node_type == "server" and (msg.type == "1" or msg.type == "2"):
            if self.config["EnableSwitch"] == 0:
                active_commlib = 1
            else:
                self.switchsend[self.config["NetworkTopo"][receiver_id-1]] += 1
                if self.switchsend[self.config["NetworkTopo"][receiver_id-1]] == self.switchtot[self.config["NetworkTopo"][receiver_id-1]]:
                    self.switchsend[self.config["NetworkTopo"][receiver_id-1]] = 0
                    active_commlib = 1

        if self.node_type == "client" and msg.type == "3":
            active_commlib = 1

        if active_commlib == 1:
            flg = 0
            for it in msg.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS]:
                print(it, msg.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS][it].size())
                if flg == 0:
                    flg = 1
                    if msg.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS][it].get_device() == 0:
                        msgx = msg.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS][it].flatten().cpu().numpy()
                    else:
                        msgx = msg.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS][it].flatten().numpy()
                else:
                    if msg.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS][it].get_device() == 0:
                        msgx = np.concatenate((msgx, msg.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS][it].flatten().cpu().numpy()))
                    else:
                        msgx = np.concatenate((msgx, msg.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS][it].flatten().numpy()))
                msg.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS][it] = msg.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS][it].size()

                # padding zeros
                app = 256 - (np.size(msgx) % 256)
                msgx = np.concatenate((msgx, np.zeros(shape=app))).astype(np.float32)
                msg.pkt_num = np.size(msgx) // 256

                print(">>> LENGTH of parameter: ", np.size(msgx))
                if (self.node_type == "server" and self.config["EnableSwitch"] == 1):
                    print(">>> COMM_LIB SEND (switch):", msg.get_sender_id(), "->", msg.get_receiver_id())
                else:
                    print(">>> COMM_LIB SEND :", msg.get_sender_id(), "->", msg.get_receiver_id())

        logging.info("msg = {}".format(msg))
        logging.info("pickle.dumps(msg) START")
        pickle_dump_start_time = time.time()
        msg_pkl = pickle.dumps(msg)
        MLOpsProfilerEvent.log_to_wandb({"PickleDumpsTime": time.time() - pickle_dump_start_time})
        logging.info("pickle.dumps(msg) END")

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
            pkt_list = []
            
            if self.config["EnableSwitch"] == 0:
                if self.node_type == "server":
                    for i in range(msg.pkt_num):
                        pkt_list.append(self.server[receiver_id].create_packet(123, i, 0, True, msgx[256*i:256*(i+1)]))
                    self.server[receiver_id].send(self.client[receiver_id], 123, pkt_list)
                if self.node_type == "client":    
                    for i in range(msg.pkt_num):
                        pkt_list.append(self.client.create_packet(123, i, 0, True, msgx[256*i:256*(i+1)]))
                    self.client.send(self.server, 123, pkt_list, False)

            else:
                if self.node_type == "server":
                    for i in range(msg.pkt_num):
                        pkt_list.append(self.server[receiver_id].create_packet(123, i, 0, False, msgx[256*i:256*(i+1)]))
                    self.server[receiver_id].send(self.client[receiver_id], 123, pkt_list)
                if self.node_type == "client":    
                    for i in range(msg.pkt_num):
                        pkt_list.append(self.client.create_packet(123, i, 0, False, msgx[256*i:256*(i+1)]))
                    self.client.send(self.server, 123, pkt_list, True)


            # restore the original message
            cul = 0
            for it in msg.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS]:
                cur = 1
                for i in range(len(msg.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS][it])):
                    cur *= msg.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS][it][i]
                tmp = torch.from_numpy(msgx[cul:cul+cur])
                cul += cur
                tmp = torch.reshape(tmp, msg.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS][it])
                msg.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS][it] = tmp

        # For server, wait for next RECV
        if self.node_type == "server" and active_commlib == 1:
            if self.config["EnableSwitch"] == 0:
                print(">>>COMMLIB RECV :", msg.get_receiver_id(), "->", msg.get_sender_id())
                self.recv_thread[msg.get_receiver_id()] = threading.Thread(
                target=self.recv_tensor, 
                args=(0,
                      msg.pkt_num, 
                      self.server[msg.get_receiver_id()], 
                      self.client[msg.get_receiver_id()], 
                      self.recv_queue[msg.get_receiver_id()]))

                self.recv_thread[msg.get_receiver_id()].start()
            else:
                print(">>>COMMLIB RECV :", msg.get_receiver_id(), "switch ID", self.config["NetworkTopo"][receiver_id-1], "->", msg.get_sender_id())
                self.recv_thread[self.config["NetworkTopo"][receiver_id-1]] = threading.Thread(
                    target=self.recv_tensor, 
                    args=(0,
                        msg.pkt_num, 
                        self.server[msg.get_receiver_id()], 
                        self.client[msg.get_receiver_id()], 
                        self.recv_queue[self.config["NetworkTopo"][receiver_id-1]]))

                self.recv_thread[self.config["NetworkTopo"][receiver_id-1]].start()



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
                        self.switchrecv[self.config["NetworkTopo"][msg.get_sender_id()-1]] += 1
                        if self.switchrecv[self.config["NetworkTopo"][msg.get_sender_id()-1]] == self.switchtot[self.config["NetworkTopo"][msg.get_sender_id()-1]]:
                            self.switchrecv[self.config["NetworkTopo"][msg.get_sender_id()-1]] = 0

                            self.recv_thread[self.config["NetworkTopo"][msg.get_sender_id()-1]].join()
                            msgx = self.recv_queue[self.config["NetworkTopo"][msg.get_sender_id()-1]].get()

                            # multiple the parameters by the number of clients that switch has
                            msgx *= self.switchtot[self.config["NetworkTopo"][msg.get_sender_id()-1]]
                        
                        else:
                            # generate a packet padded by zeros to imitate a normal msgx.
                            msgx = np.zeros(msg.pkt_num * 256)

                if active_commlib == 1:
                    msgx = msgx.astype(np.float) # convert to float
                    cul = 0
                    for it in msg.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS]:
                        cur = 1
                        for i in range(len(msg.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS][it])):
                            cur *= msg.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS][it][i]
                        tmp = torch.from_numpy(msgx[cul:cul+cur])
                        cul += cur
                        tmp = torch.reshape(tmp, msg.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS][it])
                        msg.msg_params[Message.MSG_ARG_KEY_MODEL_PARAMS][it] = tmp

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
