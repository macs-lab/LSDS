import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from slip_severity.model_utils.nn_models import LSTMNet
from slip_severity.scripts.trajectory_dataset import TrajectoryDataset
from slip_severity.learning_utils.train_utils import train_epoch
from slip_severity.learning_utils.eval_utils import evaluate_Test_trajectories
import os
import pandas as pd

DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../slip_severity/datasets/'))
MODEL_PATH = os.path.abspath(os.path.join(
    os.path.abspath(__file__), '../../learned_models/lstm_test.pth'))


def train():
    main_folders = ['SlipScore_new', 'SlipScore_Mar10']
    trajectories = load_data_from_folders(main_folders, DATA_DIR)
    dataset = TrajectoryDataset(trajectories, seq_length=5, features_indices=[
                                23, 8, 9, 10, 11, 12, 13, 16, 17], target_index=24)

    train_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model = LSTMNet(input_size=9, hidden_size=30,
                    num_layers=3, output_size=1, dropout=0.2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        loss = train_epoch(model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch + 1}: Loss = {loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)


def load_data_from_folders(main_folders, data_dir):
    trajectories = []
    for main_folder in main_folders:
        folder_path = os.path.join(data_dir, main_folder)
        object_folders = [f.path for f in os.scandir(
            folder_path) if f.is_dir()]
        for object_folder in object_folders:
            for file in os.listdir(object_folder):
                if file.endswith('.csv'):
                    trajectories.append(pd.read_csv(
                        os.path.join(object_folder, file)))
    return trajectories
