import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import torch


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, seq_length, features_indices, target_index):
        self.sequences, self.targets, self.scaler = self.preprocess_data(
            trajectories, seq_length, features_indices, target_index)

    def preprocess_data(self, trajectories, seq_length, features_indices, target_index):
        sequences, targets = [], []
        scaler = StandardScaler()
        all_data = np.vstack(
            [df.iloc[:, features_indices].values for df in trajectories])
        scaler.fit(all_data)
        for df in trajectories:
            X = scaler.transform(df.iloc[:, features_indices].values)
            y = df.iloc[:, target_index].values * 10**-1
            X_seq, y_seq = self.create_sequences(X, y, seq_length)
            sequences.extend(X_seq)
            targets.extend(y_seq)
        return np.array(sequences), np.array(targets), scaler

    def create_sequences(self, X, y, seq_length):
        Xs, ys = [], []
        for i in range(len(X) - seq_length):
            Xs.append(X[i:(i + seq_length)])
            ys.append(y[i + seq_length])
        return np.array(Xs), np.array(ys)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)
