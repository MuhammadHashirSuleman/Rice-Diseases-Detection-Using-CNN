# Rice Diseases Detection Using CNN

## Overview
This project implements a Convolutional Neural Network (CNN) for detecting rice plant diseases. It identifies three common diseases: Bacterial Leaf Blight, Brown Spot, and Leaf Smut using image classification.

The project includes data preprocessing, model training, evaluation, and a Streamlit web app for predictions.

## Features
- Data augmentation and preprocessing
- CNN model training with TensorFlow/Keras
- Model evaluation with metrics and visualizations
- Web interface for disease detection

## Setup
1. Create Anaconda environment: `conda create -n rice_disease_env python=3.9`
2. Activate: `conda activate rice_disease_env`
3. Install dependencies: `pip install -r requirements.txt`
4. Place datasets in `data/raw/mendeley/` and `data/raw/other/`.

## Usage
- Run preprocessing: `python src/preprocess.py`
- Train model: `python src/train_model.py`
- Evaluate model: `python src/evaluate_model.py`
- Launch app: `streamlit run src/app.py`
- Detect disease: `python src/detect_disease.py`

## Results
- Trained model saved in `models/`
- Evaluation metrics and plots in `results/`

## Project Structure
- `data/`: Raw, processed, and augmented datasets
- `models/`: Saved models
- `notebooks/`: Jupyter notebooks for exploration
- `results/`: Evaluation results
- `src/`: Source code for scripts and app

## .gitignore
A `.gitignore` file has been added to exclude unnecessary files like virtual environments, caches, and large datasets.