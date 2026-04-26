import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class ICLDataset(Dataset):
    def __init__(self, y_seqs, x_seqs, labels):

        self.y_seqs = y_seqs
        self.x_seqs = x_seqs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.y_seqs[idx], dtype=torch.float32), \
               torch.tensor(self.x_seqs[idx], dtype=torch.float32), \
               torch.tensor(self.labels[idx], dtype=torch.long)
    


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


def create_dataloader(y_seqs, x_seqs, labels, batch_size=32):
    dataset = ICLDataset(y_seqs, x_seqs, labels)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return loader


# from dataset import build_dataset
# y, x, labels = build_dataset(20, 30, "BPSK", 10, 3)
# loader = create_dataloader(y, x, labels, batch_size=8)

# for batch_x, batch_y, labels in loader:
#     print(batch_x.shape)  # expect (8, seq_len, 2)
#     print(batch_y.shape)  # expect (8, seq_len, 2)
#     print(labels.shape)  # expect (8,)
#     break
