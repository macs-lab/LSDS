import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Define constants for file paths
DATA_DIR = 'LSDS/datasets/'
TRAIN_MODEL_PATH = 'LSDS/slip_severity/Lstm_Mar11.pth' # Set your training model path here
SCALER_PATH = os.path.join(DATA_DIR, 'scaler.save') # Set your scalar path
TEST_DATA_DIR = os.path.join(DATA_DIR, 'test/') # Set your test data dir path

# LSTM Network
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # Take the last sequence output


# Dataset Class
class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, seq_length, features_indices, target_index):
        self.sequences, self.targets, self.scaler = self.preprocess_data(trajectories, seq_length, features_indices, target_index)

    def preprocess_data(self, trajectories, seq_length, features_indices, target_index):
        sequences, targets = [], []
        scaler = StandardScaler()
        all_data = np.vstack([df.iloc[:, features_indices].values for df in trajectories])
        scaler.fit(all_data)
        for df in trajectories:
            X = scaler.transform(df.iloc[:, features_indices].values)
            y = df.iloc[:, target_index].values * 10**-1  # Scale targets
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


# Utility Functions
def load_data_from_folders(main_folders, data_dir):
    trajectories = []
    for main_folder in main_folders:
        folder_path = os.path.join(data_dir, main_folder)
        object_folders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
        for object_folder in object_folders:
            for file in os.listdir(object_folder):
                if file.endswith('.csv'):
                    trajectories.append(pd.read_csv(os.path.join(object_folder, file)))
    return trajectories


def load_trajectory_for_eval(data_dir):
    trajectories, file_paths = [], []
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                file_path = os.path.join(folder_path, file)
                trajectories.append(pd.read_csv(file_path))
                file_paths.append(file_path)
    return trajectories, file_paths


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


# Evaluation Functions
def evaluate_Test_trajectories(model, trajectories, file_paths, seq_length, features_indices, target_index, window_size):
    model.eval()
    for df, path in zip(trajectories, file_paths):
        dataset = TrajectoryDataset([df], seq_length, features_indices, target_index)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        predictions, actuals = [], []

        with torch.no_grad():
            for sequences, targets in loader:
                outputs = model(sequences)
                predictions.extend(outputs.squeeze().cpu().numpy().flatten().tolist())
                actuals.extend(targets.squeeze().cpu().numpy().flatten().tolist())

        smoothed_predictions = moving_average(predictions, window_size)
        aligned_actuals = actuals[len(actuals) - len(smoothed_predictions):]

        mae = mean_absolute_error(aligned_actuals, smoothed_predictions)
        rmse = np.sqrt(mean_squared_error(aligned_actuals, smoothed_predictions))
        r2 = r2_score(aligned_actuals, smoothed_predictions)

        plt.figure(figsize=(10, 4))
        plt.plot(aligned_actuals, label='Ground Truth')
        plt.plot(smoothed_predictions, label='Predicted (Smoothed)')
        plt.title(f'Evaluation for: {os.path.basename(path)}\nMAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

        print(f"Metrics for {os.path.basename(path)}: MAE = {mae:.2f}, RMSE = {rmse:.2f}, R² = {r2:.2f}")


def evaluate_model(model, test_loader, train_losses=None):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for sequences, targets in test_loader:
            outputs = model(sequences)
            predictions.extend(outputs.squeeze().cpu().numpy())
            actuals.extend(targets.squeeze().cpu().numpy())

    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mse)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')
    plt.xlabel('Ground Truth (cm/s)')
    plt.ylabel('Predicted Slip Severity (cm/s)')
    plt.title('Actual vs. Predicted')

    if train_losses:
        plt.subplot(1, 2, 2)
        plt.plot(train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()

    plt.tight_layout()
    plt.show()

    print(f'Test MSE: {mse}, Test MAE: {mae}, Test RMSE: {rmse}')


# Main Function
def main(train=False, eval=False):
    if train:
        main_folders = ['SlipScore_new', 'SlipScore_Mar10']
        trajectories = load_data_from_folders(main_folders, DATA_DIR)
        dataset = TrajectoryDataset(trajectories, seq_length=5, features_indices=[23, 8, 9, 10, 11, 12, 13, 16, 17], target_index=24)
        joblib.dump(dataset.scaler, SCALER_PATH)

        train_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        model = LSTMNet(input_size=9, hidden_size=30, num_layers=3, output_size=1, dropout=0.2)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_losses = []
        for epoch in range(1000):
            model.train()
            total_loss = 0
            for sequences, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(sequences)
                loss = criterion(outputs.squeeze(), targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            train_losses.append(total_loss / len(train_loader))
            print(f'Epoch {epoch + 1}/1000, Loss: {train_losses[-1]}')

        evaluate_model(model, train_loader, train_losses)
        torch.save(model.state_dict(), TRAIN_MODEL_PATH)

    if eval:
        model = LSTMNet(input_size=9, hidden_size=30, num_layers=3, output_size=1, dropout=0.2)
        model.load_state_dict(torch.load(TRAIN_MODEL_PATH))
        trajectories, file_paths = load_trajectory_for_eval(TEST_DATA_DIR)
        evaluate_Test_trajectories(model, trajectories, file_paths, seq_length=3, features_indices=[23, 8, 9, 10, 11, 12, 13, 16, 17], target_index=24, window_size=5)


if __name__ == '__main__':
    main(train=True, eval=False)
