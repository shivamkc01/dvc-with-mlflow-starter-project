
# DVC with MLflow for Machine Learning

This project demonstrates the integration of **DVC (Data Version Control)** with **MLflow** for managing machine learning workflows. The code trains an ElasticNet model to predict wine quality based on various features, logging relevant parameters and metrics using MLflow.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Setting Up DVC](#setting-up-dvc)
- [Tracking Datasets with DVC](#tracking-datasets-with-dvc)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)

## Features

- Data versioning with DVC
- Experiment tracking and model logging with MLflow
- ElasticNet regression for predicting wine quality
- Logging of metrics and parameters
- Input example and model signature inference

## Requirements

- Python 3.6 or higher
- Pip
- DVC
- MLflow
- scikit-learn
- pandas
- numpy

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. **Create a Virtual Environment and Activate It**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the Required Packages**:
   Create a `requirements.txt` file with the following content:
   ```
   dvc
   mlflow
   scikit-learn
   pandas
   numpy
   ```

   Then install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Setting Up DVC

1. **Initialize DVC**:
   Inside your project directory, run:
   ```bash
   dvc init
   ```

   This command creates a `.dvc` directory in your project to store DVC metadata.

2. **Add a Remote Storage**:
   You need a remote storage location to store your dataset versions. This can be a local directory, S3 bucket, or any other supported storage type. For local storage, use:
   ```bash
   dvc remote add -d dvc-remote /path/to/your/remote/storage
   ```

   To verify your remote:
   ```bash
   dvc remote -v
   ```

## Tracking Datasets with DVC

1. **Add Your Dataset to DVC**:
   To track the dataset (for example, `data/winequality-red.csv`), run:
   ```bash
   dvc add data/winequality-red.csv
   ```

   This command creates a `.dvc` file (e.g., `data/winequality-red.csv.dvc`) that tracks changes to the dataset.

2. **Commit the Changes**:
   Since DVC modifies the Git repository, you need to commit the changes:
   ```bash
   git add data/winequality-red.csv.dvc .gitignore
   git commit -m "Add wine quality dataset to DVC"
   ```

3. **Push the Dataset to Remote Storage**:
   After adding the dataset, push it to the remote storage:
   ```bash
   dvc push
   ```

   This command uploads the dataset to the specified remote location, allowing you to track different versions of your dataset.

4. **Pull the Dataset from DVC**:
   When you or someone else needs to work on this project later, run:
   ```bash
   dvc pull
   ```

   This command retrieves the dataset from remote storage based on the latest version in the DVC tracking.

## Usage

To run the training script, use the following command, providing optional hyperparameters for the ElasticNet model:

```bash
python src/train.py [alpha] [l1_ratio]
```

- `alpha`: Regularization strength (default: `0.5`)
- `l1_ratio`: The ElasticNet mixing parameter (default: `0.5`)

Logs will be saved in the `logs` directory, and MLflow will track the experiment.

## Project Structure

```
your-repo/
├── src/
│   └── train.py          # Main training script
├── data/
│   └── winequality-red.csv  # Dataset
├── logs/                 # Log files
├── outputs/              # Artifacts such as feature and target files
├── .dvc/                 # DVC metadata
├── .git/                 # Git metadata
├── requirements.txt      # Python package dependencies
└── README.md             # Project documentation
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to contribute to this project! For any issues or questions, please open an issue on GitHub.

