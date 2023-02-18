A modified version of FedML that supports in-network aggregation, forked from https://github.com/FedML-AI/FedML.

This repository also incorporates the code of https://github.com/Tim-Zhong-2000/switch-fed-ml.

## Configure
To ensure proper functioning of SwitchFL, in addition to modifying the configuration of FedML, you may need to make adjustments to the SwitchFL_config.yaml (located in the SwitchFL/python/examples/cross_silo/switch_example/config) to suit your specific circumstances.

The configuration files of FedML may include: (also located in the same directory)
 - fedml_config.yaml
 - gpu_mapping.yaml
 - grpc_ipconfig.csv
 - server.yaml
 - silo_x.yaml (x represents the No. of clients)

Here is an explained example of SwitchFL_config.yaml:
```yaml
# the ID of server is 0, and the ID of clients start from 1
CommLibBasePort : 51500 # The base port of communication library
EnableSwitch : 0
ClientNum : 2
SwitchNum : 2
ServerIPAddr : "127.0.0.1"
ClientIPAddr : # According to the order of client ID, starting from Client ID 1
- "127.0.0.1"
- "127.0.0.1"
ServerSwitchIFace : # The Ethernet port connecting server and each switch, start from Switch ID 0
- "veth1"
- "veth2"
ClientSwitchIFace : # The Ethernet port connecting each client and its switch, start from Client ID 1
- "veth4"
- "veth5"
NetworkTopo : # Specifying which switch each client is belong to, start from Client ID 1
- 0
- 1
```

## Running Examples
```bash
conda create env -n SwitchFL python=3.8
git clone https://github.com/yangtx7/SwitchFL.git
cd python
pip install -e ./

cd /examples/cross_silo/switch_example
# 在x个不同终端上运行，需要先运行server
source run_server.sh # terminal 1
source run_client.sh 1 0 # terminal 2
source run_client.sh 2 0 # terminal 3
...
source run_client.sh {x-1} 0 # terminal x
```

## Existing Problems
### (issue #1)训练完毕无法自动退出
- 原因：通讯库非remote的server与client类无法正常结束
- 解决方法：手动kill进程（ctrl+c即可）
### (issue #2)client启动时报raise ChildFailedError(torch.distributed.elastic.multiprocessing.errors.ChildFailedError）
- 原因：似乎是FedML自带的问题
- 解决方法：重新运行client（重新运行之前最好ps看一下时候有还行运行的相关进程，如果有则手动kill掉，避免报通讯库中端口已被占用的错误）
### (issue #3, solved)更换模型/数据集后通信库报错
### (issue #4, solved)使用更多client时报错
### (issue #5, solved)训练准确率及loss没有变化
### (issue #6)不支持一部分client使用switch而另一部分不使用

## Update Logs
#### 2023-02-18
- 添加对switch的支持，实现server、client与switch通信的逻辑

#### 2023-02-17
- 支持对网络拓扑的配置
- 更新通讯库到commit #5c5ae7f，解决了server无法处理多个client同时发送的问题

#### 2023-02-16
- 更新通讯库到commit #ac37a59，使用packet list来与通讯库通信
- 添加SwitchFL_config.yaml以及相关指引
- 支持使用多个clients
#### 2023-02-13
- 更新tensor发送逻辑，在grpc发送时发送tensor大小
- 更新通讯库到commit #a0f63d2，解决了训练准确率的问题
- 使用通信库，可以正常进行训练