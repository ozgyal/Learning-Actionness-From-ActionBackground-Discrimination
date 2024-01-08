import torch as th

from featextract.s3dg import S3D


class Feature:
    def __init__(self, dataloader, net_weights, use_gpu):
        self.dataloader = dataloader
        self.net_weights = net_weights
        self.use_gpu = use_gpu
        self.net = self._set_network()

    def _set_network(self):
        # Initiate the model
        net = S3D(space_to_depth=True)

        # Load the model weights
        net.load_state_dict(th.load(self.net_weights))

        if self.use_gpu:
            net.cuda()

        return net

    def extract(self):
        # TODO
        print(self.dataloader)
