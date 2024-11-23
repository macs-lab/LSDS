import os
import logging
import numpy as np
import pandas as pd
import joblib
from typing import List, Tuple

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


def load_data(
    no_slip_paths: List[str], slip_path: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and concatenate data from no-slip and slip directories.

    Args:
        no_slip_paths (List[str]): List of directory paths for no-slip data.
        slip_path (str): Directory path for slip data.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features DataFrame and target Series.
    """
    feature_columns = [
        "ent",
        "delta_entropy",
        "vx",
        "vy",
        "vnet",
        "div",
        "delta_div",
        "curl",
        "delta_curl",
        "theta",
        "delta_theta",
        "area",
        "delta_area",
    ]

    data_list = []
    target_list = []

    # Load no-slip data
    logging.info("Loading no-slip data...")
    for path in no_slip_paths:
        try:
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                df = pd.read_csv(file_path, delimiter=",", header=None)
                # Assuming columns are indexed from 0
                features = df.iloc[1:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
                data_list.append(features)
                target_list.extend([0] * len(features))
        except Exception as e:
            logging.error(f"Error loading no-slip file {file_path}: {e}")

    # Load slip data
    logging.info("Loading slip data...")
    try:
        for filename in os.listdir(slip_path):
            file_path = os.path.join(slip_path, filename)
            df = pd.read_csv(file_path, delimiter=",", header=None)
            features = df.iloc[1:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
            data_list.append(features)
            target_list.extend([1] * len(features))
    except Exception as e:
        logging.error(f"Error loading slip file {file_path}: {e}")

    # Concatenate all data
    if data_list:
        X = pd.concat(data_list, ignore_index=True)
        X.columns = feature_columns
        Y = pd.Series(target_list, name="slipFlag")
        logging.info(f"Total samples loaded: {len(Y)}")
        return X, Y
    else:
        logging.error("No data loaded. Please check the file paths.")
        raise ValueError("No data loaded.")


def train_models(
    X_train: pd.DataFrame, Y_train: pd.Series
) -> Tuple[RandomForestClassifier, GradientBoostingClassifier]:
    """
    Train Random Forest and Gradient Boosting classifiers.

    Args:
        X_train (pd.DataFrame): Training features.
        Y_train (pd.Series): Training targets.

    Returns:
        Tuple[RandomForestClassifier, GradientBoostingClassifier]: Trained models.
    """
    # Initialize models with specific hyperparameters
    rf = RandomForestClassifier(
        bootstrap=True,
        max_depth=20,
        max_features=3,
        min_samples_leaf=5,
        min_samples_split=10,
        n_estimators=40,
        random_state=42,
    )
    gb = GradientBoostingClassifier(
        max_depth=9,
        min_samples_split=450,
        n_estimators=115,
        random_state=42,
    )

    # Train models
    logging.info("Training Random Forest classifier...")
    rf.fit(X_train, Y_train)
    logging.info("Random Forest training completed.")

    logging.info("Training Gradient Boosting classifier...")
    gb.fit(X_train, Y_train)
    logging.info("Gradient Boosting training completed.")

    return rf, gb


def evaluate_models(
    rf: RandomForestClassifier,
    gb: GradientBoostingClassifier,
    X_test: pd.DataFrame,
    Y_test: pd.Series,
) -> None:
    """
    Evaluate the trained models and print metrics.

    Args:
        rf (RandomForestClassifier): Trained Random Forest model.
        gb (GradientBoostingClassifier): Trained Gradient Boosting model.
        X_test (pd.DataFrame): Testing features.
        Y_test (pd.Series): Testing targets.
    """
    # Predictions
    y_pred_rf = rf.predict(X_test)
    y_pred_gb = gb.predict(X_test)

    # Evaluation Metrics
    metrics_dict = {
        "Random Forest": y_pred_rf,
        "Gradient Boosting": y_pred_gb,
    }

    for model_name, y_pred in metrics_dict.items():
        accuracy = accuracy_score(Y_test, y_pred) * 100
        precision = precision_score(Y_test, y_pred, zero_division=0) * 100
        recall = recall_score(Y_test, y_pred, zero_division=0) * 100
        f1 = f1_score(Y_test, y_pred, zero_division=0) * 100

        logging.info(f"--- {model_name} Metrics ---")
        logging.info(f"Accuracy: {accuracy:.2f}%")
        logging.info(f"Precision: {precision:.2f}%")
        logging.info(f"Recall: {recall:.2f}%")
        logging.info(f"F1 Score: {f1:.2f}%")
        logging.debug(f"Classification Report for {model_name}:\n{classification_report(Y_test, y_pred)}")

    # Optionally, more detailed reports or visualizations can be added here


def save_models(
    rf: RandomForestClassifier, gb: GradientBoostingClassifier, directory: str
) -> None:
    """
    Save the trained models to disk.

    Args:
        rf (RandomForestClassifier): Trained Random Forest model.
        gb (GradientBoostingClassifier): Trained Gradient Boosting model.
        directory (str): Directory path to save the models.
    """
    os.makedirs(directory, exist_ok=True)

    rf_path = os.path.join(directory, "rf_w_grasp.sav")
    gb_path = os.path.join(directory, "gb_w_grasp.sav")

    try:
        joblib.dump(rf, rf_path)
        logging.info(f"Random Forest model saved to {rf_path}")
    except Exception as e:
        logging.error(f"Error saving Random Forest model: {e}")

    try:
        joblib.dump(gb, gb_path)
        logging.info(f"Gradient Boosting model saved to {gb_path}")
    except Exception as e:
        logging.error(f"Error saving Gradient Boosting model: {e}")


def main():
    """
    Main function to execute the data loading, training, evaluation, and saving processes.
    """
    # Configuration
    config = {
        "no_slip_paths": [
            "./slip_detection/datasets/Classifier_train/NoSlip",
            "./slip_detection/datasets/Classifier_train/Grasp",
        ],
        "slip_path": "./slip_detection/datasets/Classifier_train/Slip",
        "test_size": 0.25,
        "random_state": 10,
        "model_save_dir": "./slip_detection/trained_models/",
    }

    # Load and preprocess data
    try:
        X, Y = load_data(config["no_slip_paths"], config["slip_path"])
    except ValueError as ve:
        logging.critical(f"Data loading failed: {ve}")
        return

    # Feature and target separation (already done in load_data)
    # Split the dataset
    logging.info("Splitting data into training and testing sets...")
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=config["test_size"], random_state=config["random_state"]
    )
    logging.info(f"Training samples: {len(Y_train)}, Testing samples: {len(Y_test)}")

    # Train models
    rf_model, gb_model = train_models(X_train, Y_train)

    # Evaluate models
    evaluate_models(rf_model, gb_model, X_test, Y_test)

    # Save models
    save_models(rf_model, gb_model, config["model_save_dir"])

    logging.info("Script execution completed successfully.")


if __name__ == "__main__":
    main()
