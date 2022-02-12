import torch

class Word2vecModel():
    def __init__(self, word_count, weights, skipgram=True, vector_dim=150, sample_size=5, device='cpu'):
        self.vectors         = torch.rand((word_count, vector_dim), requires_grad=True, device=device)
        self.hiddens         = torch.zeros((word_count, vector_dim), requires_grad=True, device=device)
        self.vector_dim      = vector_dim
        self.weights         = weights
        self.sample_size     = sample_size
        self.device          = device
        self.negative_sample = self.negative_sample_skipgram if skipgram else self.negative_sample_cowb
        self.forward         = self.forward_skipgram if skipgram else self.forward_cowb
        self.paramters       = [self.vectors, self.hiddens]

    def negative_sample_cowb(self, target):
        target_weight = self.weights[target]
        self.weights[target] = 0
        indices = torch.multinomial(self.weights, self.sample_size, replacement=True)
        thetas = self.hiddens[indices]
        thetas = torch.cat((
            thetas,
            self.hiddens[target].unsqueeze(0)
        ), dim=0).transpose(-1, -2)
        self.weights[target] = target_weight
        return thetas

    def negative_sample_skipgram(self, target, context):
        target_weight = self.weights[target]
        self.weights[target] = 0
        indices = torch.multinomial(self.weights, len(context) * self.sample_size, replacement=True)
        thetas = self.hiddens[indices]
        thetas = thetas.view(len(context), self.sample_size, self.vector_dim)
        thetas = torch.cat((
            thetas,
            self.hiddens[target].expand(len(context), 1, self.vector_dim)
        ), dim=1).transpose(-1, -2)
        self.weights[target] = target_weight
        return thetas

    def forward_cowb(self, targets, contexts):
        thetas = torch.stack([
            self.negative_sample(target)
            for target in targets
        ]).to(self.device)
        X = torch.sum(
            self.vectors[contexts],
            axis=1
        ).unsqueeze(1)
        y = X @ thetas
        codes = torch.zeros_like(y).to(self.device)
        codes[..., self.sample_size] = 1
        loss = torch.nn.BCEWithLogitsLoss(reduction='sum')(y, codes)
        return loss

    def forward_skipgram(self, targets, contexts):
        thetas = torch.stack([
            self.negative_sample(target, context)
            for target, context in zip(targets, contexts)
        ]).to(self.device)
        X = self.vectors[contexts].unsqueeze(2)
        y = (X @ thetas).squeeze()
        codes = torch.zeros_like(y).to(self.device)
        codes[..., self.sample_size] = 1
        loss = torch.nn.BCEWithLogitsLoss(reduction='sum')(y, codes)
        return loss
