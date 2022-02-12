import torch
from collections import Counter
import heapq

class DataSet(torch.utils.data.Dataset):
    def __init__(self, documents, topk=150, window=5):
        words             = [word for document in documents for word in document]
        vocabulary        = dict(Counter(words))
        self.times        = torch.tensor(list(vocabulary.values()), dtype=torch.float)
        self.word_count   = len(vocabulary)
        self.topkwords    = self.get_topk_words(vocabulary, topk)
        self.word2index   = {
            word : index
            for index, word in enumerate(vocabulary.keys())
        }
        self.contexts, self.targets = self.get_batches(documents, self.word2index, window // 2)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.targets[index], self.contexts[index]

    @staticmethod
    def get_topk_words(vocabulary, topk):
        topkwords = [
            (times, word)
            for word, times in vocabulary.items()
        ]
        topkwords = heapq.nlargest(topk, topkwords)
        topkwords = [t[1] for t in topkwords]
        return topkwords

    @staticmethod
    def get_batches(documents, word2index, half_window):
        contexts = []
        targets  = []
        for document in documents:
            for index in range(half_window, len(document) - half_window):
                context = [word2index[word] for word in document[index - half_window: index]]
                context.extend([word2index[word] for word in document[index + 1: index + half_window]])
                contexts.append(context)
                targets.append(word2index[document[index]])
        contexts = torch.tensor(contexts)
        targets  = torch.tensor(targets)
        return contexts, targets
