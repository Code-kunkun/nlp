import numpy as np 
import itertools

class Dataset:
    def __init__(self, x, y, v2i, i2v):
        self.x, self.y = x, y 
        self.v2i = v2i
        self.i2v = i2v 
        self.vocab = v2i.keys()

    def sample(self, n):
        b_idx = np.random.randint(0, len(self.x), n)
        bx, by = self.x[b_idx], self.y[b_idx]
        return bx, by

    @property
    def num_word(self):
        return len(self.v2i)


def process_w2v_data(corpus, skip_window=2, method='skip_gram'):
    all_words = [sentence.split() for sentence in corpus]
    all_words = np.array(list(itertools.chain(*all_words)))
    # 按token出现频率降序排序
    vocab, counts = np.unique(all_words, return_counts=True)
    vocab = vocab[np.argsort(counts)[::-1]]
    print(vocab)
    v2i = {v:i for i, v in enumerate(vocab)}
    i2v = {i:v for v, i in v2i.items()}

    # pair data
    pairs = []
    js = [i for i in range(-skip_window, skip_window + 1) if i != 0]

    for c in corpus:
        words = c.split()
        w_idx = [v2i[w] for w in words]
        if method == "skip_gram":
            for i in range(len(w_idx)):
                for j in js:
                    if i + j < 0 or i + j >= len(w_idx):
                        continue
                    else:
                        pairs.append((w_idx[i], w_idx[i+j])) # (center, context)
        elif method == "cbow":
            for i in range(skip_window, len(w_idx)-skip_window):
                context = []
                for j in js:
                    context.append(w_idx[i+j])
                pairs.append(context + [w_idx[i]]) # (contexts, center)
        else:
            raise ValueError

    pairs = np.array(pairs)
    if method == "skip_gram":
        x, y = pairs[:, 0], pairs[:, 1]
    elif method == 'cbow':
        x, y = pairs[:, :-1], pairs[:, -1]

    else:
        raise ValueError
    return x, y, v2i, i2v
    #return Dataset(x, y, v2i, i2v)

