import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class ICLDataset(Dataset):
    def __init__(self, X, y):

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), \
               torch.tensor(self.y[idx], dtype=torch.long)


def collate_fn(batch):

    sequences, labels = zip(*batch)

    lengths = [seq.shape[0] for seq in sequences]
    max_len = max(lengths)

    padded = []

    for seq in sequences:
        pad_len = max_len - seq.shape[0]

        if pad_len > 0:
            pad = torch.zeros((pad_len, seq.shape[1]))
            seq = torch.cat([seq, pad], dim=0)

        padded.append(seq)

    padded_X = torch.stack(padded)
    y = torch.stack(labels)

    return padded_X, y


def create_dataloader(X, y, batch_size=32):
    dataset = ICLDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    return loader


from dataset import build_dataset
X, y = build_dataset(20, 30, "BPSK", 10, 3)
loader = create_dataloader(X, y, batch_size=8)

for batch_X, batch_y in loader:
    print(batch_X.shape)  # expect (8, seq_len, 2)
    print(batch_y.shape)  # expect (8,)
    break
