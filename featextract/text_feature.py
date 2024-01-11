import torch as th

from featextract.feature import Feature


class TextFeature(Feature):
    def __init__(self, dataloader, net_weights_path, use_gpu, word2vec_path, text_features_path):
        self._task_features_path = text_features_path

        super().__init__(dataloader, net_weights_path, use_gpu, word2vec_path)

    def extract_features(self):
        # TODO
        with th.no_grad():
            for i_batch, data in enumerate(self.dataloader):
                print(data['step_name'])
                print(data['step_ids'])
