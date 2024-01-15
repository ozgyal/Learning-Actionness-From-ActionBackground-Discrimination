"""TextDataset class is based on: https://github.com/antoine77340/MIL-NCE_HowTo100M"""

import torch as th
from torch.utils.data import Dataset
import numpy as np
import re


class TextDataset(Dataset):
    def __init__(self, steps, dict_path, max_words):
        self.steps = steps
        self.max_words = max_words
        token_to_word = np.load(dict_path)
        self.word_to_token = {}
        for i, t in enumerate(token_to_word):
            self.word_to_token[t] = i + 1

    def __len__(self):
        return len(self.steps)

    def _zero_pad_tensor_token(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = th.zeros(size - len(tensor)).long()
            return th.cat((tensor, zero), dim=0)

    def _words_to_token(self, words):
        words = [self.word_to_token[word] for word in words if word in self.word_to_token]
        if words:
            we = self._zero_pad_tensor_token(th.LongTensor(words), self.max_words)
            return we
        else:
            return th.zeros(self.max_words).long()

    def _split_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_ids(self, x):
        return self._words_to_token(self._split_text(x))

    def __getitem__(self, idx):

        step_name = self.steps[idx]
        step_word_ids = self._words_to_ids(self.steps[idx])

        return {'step_name': step_name, 'step_word_ids': step_word_ids}
