# Simpsons Character Classifier

This repository contains a TensorFlow CNN model to classify Simpsons characters from images.

## Installation

Install the required Python packages:

```
pip install -r requirements.txt
```

## Dataset

Download the dataset from Kaggle:

[The Simpsons Characters Dataset](https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset/data)

Before placing the data, **create the `Dataset` folder** in your project directory if it doesn't exist:

```bash
mkdir Dataset
```

Then, place the extracted dataset as follows:

- Training images in `Dataset/train/` (one subfolder per character)
- Test images in `Dataset/test/test/`

## Folder structure for test set

The notebook will automatically organize test images into subfolders based on filenames.

## Usage

> **Note:** This project is designed to be run inside a Jupyter Notebook.  
> To start, open your terminal or command prompt, navigate to the project folder, and run:
>
> ```bash
> jupyter notebook
> ```
>
> This will open the notebook interface in your browser. Then, open the notebook file and run the code cells.

### Train model

Run the training script (or notebook cell):

```python
train_and_evaluate()
```

### Predict

Run the predict (or notebook cell):

```python
load_trained_model_and_predict(image_path)
```
