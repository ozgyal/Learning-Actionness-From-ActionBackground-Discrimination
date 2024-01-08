import torch as th

from featextract.s3dg import S3D


class Feature:
    def __init__(self, dataloader, net_weights_path, use_gpu, word2vec_path):
        self.dataloader = dataloader
        self.net_weights_path = net_weights_path
        self.use_gpu = use_gpu
        self.word2vec_path = word2vec_path
        self.net = self._set_network()

    def _set_network(self):
        # Initiate the model
        net = S3D(space_to_depth=True, word2vec_path=self.word2vec_path)

        # Load the model weights
        net.load_state_dict(th.load(self.net_weights_path))

        if self.use_gpu:
            net.cuda()

        return net

    def extract(self):
        # TODO
        print(self.dataloader)
