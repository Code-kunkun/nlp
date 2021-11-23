import argparse
from torch import nn
import torch
from torch.nn.functional import cross_entropy,softmax
from utils import Dataset,process_w2v_data
from visual import show_w2v_word_embedding
from torch.utils.data import Dataset, DataLoader
from dataloader_skip import Skip_data

corpus = [
    # numbers
    "5 2 4 8 6 2 3 6 4",
    "4 8 5 6 9 5 5 6",
    "1 1 5 2 3 3 8",
    "3 6 9 6 8 7 4 6 3",
    "8 9 9 6 1 4 3 4",
    "1 0 2 0 2 1 3 3 3 3 3",
    "9 3 3 0 1 4 7 8",
    "9 9 8 5 6 7 1 2 3 0 1 0",

    # alphabets, expecting that 9 is close to letters
    "a t g q e h 9 u f",
    "e q y u o i p s",
    "q o 9 p l k j o k k o p",
    "h g y i u t t a e q",
    "i k d q r e 9 e a d",
    "o p d g 9 s a f g a",
    "i u y g h k l a s w",
    "o l u y a o g f s",
    "o p i u y g d a s j d l",
    "u k i l o 9 l j s",
    "y g i s h k j l f r f",
    "i o h n 9 9 d 9 f a 9",
]

class SkipGram(nn.Module):

    def __init__(self, v_dim, emb_dim):
        
        super().__init__()
        self.v_dim = v_dim
        self.emb_dim = emb_dim

        self.embeddings = nn.Embedding(self.v_dim, self.emb_dim)
        self.embeddings.weight.data.normal_(0, 0.1)
        self.hidden_out = nn.Linear(emb_dim, v_dim)

        self.opt = torch.optim.Adam(self.parameters(),lr=0.01)


    def forward(self, x):
        o = self.embeddings(x)
        return o 

    def loss(self, x, y):
        embedded = self(x)
        pred = self.hidden_out(embedded)
        y = y.long()
        return cross_entropy(pred, y)

    def step(self, x, y):
        self.opt.zero_grad()
        loss = self.loss(x, y)
        loss.backward()
        self.opt.step()
        return loss


def train(model, data):
    if torch.cuda.is_available():
        print('GPU train available')
        device = torch.device('cuda')
        model = model.cuda()

    else:
        device = torch.device('cpu')
        model = model.cpu()

    for t in range(2500):
        bx, by = data.sample(8)
        bx, by = torch.from_numpy(bx).to(device), torch.from_numpy(by).to(device)
        print('bx shape', bx)
        print('by shape', by)
        loss = model.step(bx, by)
        if t % 200 == 0:
            print(f'step:{t} | loss:{loss}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="batch_size")
    parser.add_argument("--epoches", type=int, default=10, help="epoches")
    args = parser.parse_args()
    train_data = Skip_data()
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    model = SkipGram(train_data.vocab, 2)
    device = torch.device('cuda')
    if torch.cuda.is_available():
        print('GPU train available')
        model = model.cuda()
    epoches = 1000
    for epoch in range(args.epoches):
        for batch_id, batch in enumerate(train_loader):
            x = batch[0].to(device)
            y = batch[1].to(device)
            loss = model.step(x, y)
            if batch_id % 30 == 0:
                print(f'batch_id:{batch_id} | loss:{loss}')
        print(f'epoch:{epoch} | loss:{loss}')

    show_w2v_word_embedding(model,train_data,"./skip_gram.png")
