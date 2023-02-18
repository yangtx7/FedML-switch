import fedml
from fedml import FedMLRunner
import sys, os

if __name__ == "__main__":
    
    sys.path.append(os.getcwd()[:-34]+"/fedml/core/distributed/communication/switch/python_io")
    # init FedML framework
    args = fedml.init()

    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model
    model = fedml.model.create(args, output_dim)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()
