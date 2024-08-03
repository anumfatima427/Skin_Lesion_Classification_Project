# Skin Lesion Classification using DenseNet121

This repository contains the code and trained models for the classification of skin lesions, based on the filtered ISIC 2019 dataset.

## Dataset

The dataset used for this project is the filtered ISIC 2019 dataset. You can download it from the following link:
[ISIC 2019 Dataset](https://drive.google.com/drive/folders/1TeIbfKU5SiThZA-qFCpfzZq6Lk1Uib3l?usp=sharing)

## Trained Models

Pretrained models are available in the [`trained_models`](https://drive.google.com/drive/folders/1qdtUhE-9hO-yIn8K85vJDgR25n7BJEkb?usp=sharing) folder. The following models are included:
1. DenseNet121
2. VGG16
3. InceptionV3
4. ResNet50

## Code Files

This repository includes the following Jupyter notebooks:

### 1. Multiclass-Classification-Using-DenseNet121.ipynb
- **Description**: Contains the code for data preprocessing, model training, and model evaluation.
- **Usage**: To train the model from scratch and evaluate its performance on the filtered ISIC 2019 dataset.

### 2. Test_Model.ipynb
- **Description**: Allows users to test the pretrained models on a provided test dataset.
- **Usage**: To test the performance of the pretrained models using a separate test set.

### 3. UMAP_Projection.ipynb
- **Description**: Visualizes 2D projections of the model's features using UMAP.
- **Usage**: To understand the feature space and visualize the clustering of different classes.

### 4. Train_Test_Val_Split.ipynb
- **Description**: Splits the dataset into training, validation, and test sets.
- **Usage**: To prepare the dataset for training and evaluation.

## Model Performance

| Model         | Accuracy | Precision | Recall  | F1-Score | AUC    |
|---------------|----------|-----------|---------|----------|--------|
| DenseNet121   | 0.75     | 0.75      | 0.75    | 0.75     | 0.93   |
| VGG16         | 0.32     | 0.66      | 0.32    | 0.31     | 0.85   |
| InceptionV3   | 0.64     | 0.69      | 0.64    | 0.65     | 0.90   |
| ResNet50      | 0.51     | 0.26      | 0.51    | 0.34     | 0.67   |

## Usage

### Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/skin-lesion-classification.git
    cd skin-lesion-classification
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset and place it in the appropriate folder.

### Running the Notebooks

1. **Train the Model**:
    - Open `Multiclass-Classification-Using-DenseNet121.ipynb` and run all cells to preprocess the data, train the model, and evaluate its performance.

2. **Test the Model**:
    - Open `Test_Model.ipynb` and run all cells to load a pretrained model and evaluate it on the test set.

3. **Visualize with UMAP**:
    - Open `UMAP_Projection.ipynb` and run all cells to generate 2D projections of the model's feature space.

4. **Prepare the Dataset**:
    - Open `Train_Test_Val_Split.ipynb` and run all cells to split the dataset into training, validation, and test sets.

### Test, Train, and Validation Set

You can download the test, train, and validation set from the following link:
[Train, Test, Val Set](https://drive.google.com/drive/folders/1vM8rlLS-N9xQFQcSvaOiPvkd21rNItGY?usp=sharing)

## Acknowledgements

- ISIC 2019 Challenge for providing the dataset.
- The developers of DenseNet for their contributions to deep learning research.
- Bhattiprolu, S. (2023). python_for_microscopists. [GitHub](https://github.com/bnsreenu/python_for_microscopists/tree/master)
