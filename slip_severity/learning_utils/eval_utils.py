import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from slip_severity.scripts.trajectory_dataset import TrajectoryDataset
import os


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def evaluate_Test_trajectories(model, trajectories, file_paths, seq_length, features_indices, target_index, window_size):
    model.eval()
    for df, path in zip(trajectories, file_paths):
        dataset = TrajectoryDataset(
            [df], seq_length, features_indices, target_index)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        predictions, actuals = [], []

        with torch.no_grad():
            for sequences, targets in loader:
                outputs = model(sequences)
                predictions.extend(outputs.squeeze().cpu().numpy())
                actuals.extend(targets.squeeze().cpu().numpy())

        smoothed_predictions = moving_average(predictions, window_size)
        aligned_actuals = actuals[len(actuals) - len(smoothed_predictions):]

        mae = mean_absolute_error(aligned_actuals, smoothed_predictions)
        rmse = np.sqrt(mean_squared_error(
            aligned_actuals, smoothed_predictions))
        r2 = r2_score(aligned_actuals, smoothed_predictions)

        plt.figure(figsize=(10, 4))
        plt.plot(aligned_actuals, label='Ground Truth')
        plt.plot(smoothed_predictions, label='Predicted (Smoothed)')
        plt.title(
            f'Evaluation for: {os.path.basename(path)}\nMAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
