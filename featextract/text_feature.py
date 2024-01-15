import torch as th
import os

from featextract.feature import Feature


class TextFeature(Feature):
    def __init__(self, dataloader, net_weights_path, use_gpu, word2vec_path, dict_path, text_features_path):
        self._task_features_path = text_features_path

        super().__init__(dataloader, net_weights_path, use_gpu, word2vec_path, dict_path)

    def extract_features(self):
        with th.no_grad():
            step_name_features = dict()
            for i_batch, data in enumerate(self.dataloader):
                step_name = data['step_name'][0]
                step_word_ids = data['step_word_ids']

                if self.use_gpu:
                    step_feature = self.net.module.text_module(step_word_ids.cuda())
                else:
                    step_feature = self.net.text_module(step_word_ids)

                step_name_features[step_name] = step_feature
                print(step_name + ' done')

        th.save(step_name_features, os.path.join(self._task_features_path, 'step_name_features'))
