from torch.utils.data import Dataset, DataLoader
from skip_gram import corpus
from utils import process_w2v_data

class Skip_data(Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.x, self.y, self.v2i, self.i2v = process_w2v_data(corpus, skip_window=2, method='skip_gram')
        self.vocab = len(self.v2i)
        self.num_word = self.vocab

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        return x,y

    def __len__(self):
        return len(self.x)


# train_loader = DataLoader(train_data,batch_size=8,shuffle=False)
# for i, batch in enumerate(train_loader):
#     print(batch)


    