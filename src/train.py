import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tqdm

from DataLoader import DataSet
from Word2vec import Word2vecModel

torch.manual_seed(114514)
device = "cuda" if torch.cuda.is_available() else "cpu"

def show(word2vec):
    X = [vector.numpy() for vector in word2vec.values()]
    tsne_model = TSNE(perplexity=40, n_components=2, init="pca", n_iter=5000, random_state=23)
    Y = tsne_model.fit_transform(X)
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.scatter(Y[:, 0], Y[:, 1])
    for i, word in enumerate(word2vec.keys()):
        plt.annotate(word, xy=(Y[i, 0], Y[i, 1]))
    _ = plt.savefig('pic.png')

def get_most_similar(word, word2vec, topk=5):
    if word not in word2vec.keys():
        raise ValueError(f"{word} doesn't exist!")
    vocabulary = {
        other : torch.cosine_similarity(word2vec[word], word2vec[other], dim=0)
        for other in word2vec.keys()
    }
    vocabulary.pop(word)
    return DataSet.get_topk_words(vocabulary, topk)

with open('dataset/input.pkl', 'rb') as f:
    documents = pickle.load(f)

dataset = DataSet(
    documents = documents,
    topk      = 150,
    window    = 5
)

model = Word2vecModel(
    word_count  = dataset.word_count,
    weights     = dataset.times,
    skipgram    = True,
    vector_dim  = 150,
    sample_size = 5,
    device      = device
)

dataloader = DataLoader(
    dataset    = dataset,
    batch_size = 32
)

optimizer = torch.optim.Adam(model.paramters, lr=0.001)
epoch = 5
try:
    for i in range(epoch):
        with tqdm.tqdm(total=len(dataloader), ncols=100) as pbar:
            avg_loss = 5
            for targets, contexts in dataloader:
                optimizer.zero_grad()
                targets, contexts = targets.to(device), contexts.to(device)
                loss = model.forward(targets, contexts)
                loss.backward()
                optimizer.step()
                loss.detach_()
                avg_loss += loss
                pbar.set_description(f"Epoch: {i}, loss: {loss.item() / len(targets):.6f}")
                pbar.update(1)
            avg_loss /= len(dataset)
            pbar.set_description(f"Epoch: {i}, loss: {avg_loss.item():.6f}")
finally:
    word2vec = {
        word : model.vectors[dataset.word2index[word]].detach().cpu()
        for word in dataset.topkwords
    }
    np.save('vec.npy', word2vec)
    show(word2vec)
    print(get_most_similar('张无忌', word2vec))