import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

if __name__ == '__main__':
    a = torch.tensor(
        [
            [1, 3, 4, 5],
            [2, 0, 0, 0],
            [1, 0, 0, 0]
        ]
    )
    b = pack_padded_sequence(a, [4, 1, 1], batch_first=True)
    c = torch.nn.Embedding(10, 20)
    print(c(b))
