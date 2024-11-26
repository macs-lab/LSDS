# [LSDS](https://arxiv.org/abs/2411.07442) Codebase (Click on the link to go to the research paper)

This repository contains minimal code for implementing and trying out Learned Slip-Detection-Severity networks and algorithms. The project is organized to enable easy training, evaluation, and further customization of the codebase.

---

## Slip Severity Training

## Prerequisites

1. **Install Dependencies**: Ensure you have Python 3.8 or above and install the required Python packages (can be installed in base env but conda env recommended for better project management)

```bash
# Step 0: Download Miniconda (or) Ananconda for your linux distribution (Google the steps, pretty straightforward)

# Step 1: Create a new conda environment
conda create -n lsds python=3.8 -y

# Step 2: Activate the environment
conda activate lsds

# Step 3: Use pip to install packages from requirements.txt
pip install -r requirements.txt

# Step 4: Verify the installations (optional)
conda list
```

2. **Dataset Structure**: Ensure your dataset is placed in the `datasets/` directory. For example:
   ```
   datasets/
   ├── test_1/
   ├── test_2/
   ├── test_3/
   ```

3. **Environment Variables**: Verify dataset paths in the scripts are correct.

In case of Module not found error, go to the LSDS directory and run this in the terminal. This should fix the error.
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```
---

## Running Training

To train the LSTM model for LSDS:

1. Run the training script:
   ```bash
   python slip_severity/scripts/main.py
   ```

2. By default, training is enabled in the `main.py` script. Ensure you run the script this way. You can set different flags according to your use case.
   ```python
   python slip_severity/scripts/main.py --train
   ```

---

<!-- ## Running Evaluation

To evaluate the trained model on test datasets:

1. Ensure the model weights are saved in `slip_severity/learned_models/

2. Run the evaluation script:
   ```bash
   python slip_severity/scripts/main.py
   ```

Enable evaluation mode in `main.py` by setting:
```python
train_flag = False
eval_flag = True
```

Note that eval has not been implemented, however one may implement it the way one wants in scripts

--- -->

## Outputs

- **Training**:
  - Model weights will be saved to `slip_severity/learned_models/<your_model_name>.pth`.
  - Training logs will be printed to the console.

<!-- - **Evaluation**:
  - Metrics such as MAE, RMSE, and R² are printed for each test trajectory.
  - Plots for predicted vs. ground truth values are displayed. -->

---

## Slip Detection Training

Same prerequisites as Slip Severity Training

1. **Dataset Structure**: Place your datasets in the `datasets/` directory under `slip_detection/`. The expected directory structure is:

   ```
   slip_detection/
   ├── datasets/
   │   ├── NoSlip/
   │   │   ├── file1.csv
   │   │   ├── file2.csv
   │   ├── Slip/
   │   │   ├── file3.csv
   │   │   ├── file4.csv
   │   ├── Grasp/      # If applicable
   │   │   ├── file5.csv
   ```

2. **Environment Variables**: Verify dataset paths in the scripts are correct.

---

### Running Training

To train the Slip Detection model:

1. Run the training script:

   ```bash
   python scripts/train.py
   ```

   By default, the training script will process the datasets, train the models, and save them to the `trained_models/` directory.

---

### Outputs

- **Training**:
  - Trained models will be saved in the `trained_models/` directory, e.g., `trained_models/gb_w_grasp.sav`.
  - Training metrics and accuracy scores will be printed to the console.

---

## Troubleshooting

1. **Module Not Found**: Ensure you're running the scripts from the project root:
   ```bash
   python slip_severity/scripts/main.py
   ```

2. **FileNotFoundError**: Verify your dataset paths and structure match the expected format.

3. **Custom Dataset**: Update dataset paths and feature indices in training script

