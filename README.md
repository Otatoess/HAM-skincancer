
# Skin Cancer Classification with InceptionV3

This repository contains a deep learning model for skin cancer classification using the InceptionV3 architecture. The model was trained on the HAM10000 dataset and is designed with computational efficiency in mind. It was originally developed as a high school project and is optimized to run on a CPU.

## Project Overview

The project involves classifying skin lesions into seven different categories using the HAM10000 dataset. The dataset is severely/highly imbalanced and therefore balanced using data augmentation techniques and class weighting to handle class imbalances effectively.

### Model Performance (val)

- **F1 Score:** 0.8992
- **Precision:** 0.9234
- **Recall:** 0.8932

#### Per-Class Metrics:
- **akiec:**
  - F1 Score: 0.9710
  - Precision: 0.9710
  - Recall: 0.9710
- **bcc:**
  - F1 Score: 0.9787
  - Precision: 0.9684
  - Recall: 0.9892
- **bkl:**
  - F1 Score: 0.8673
  - Precision: 0.7906
  - Recall: 0.9605
- **df:**
  - F1 Score: 0.9032
  - Precision: 0.8235
  - Recall: 1.0000
- **mel:**
  - F1 Score: 0.7526
  - Precision: 0.6152
  - Recall: 0.8543
- **vasc:**
  - F1 Score: 0.9545
  - Precision: 0.9130
  - Recall: 1.0000

## Requirements

- **Python:** 3.6+
- **TensorFlow:** 2.10 (Compatible with higher versions, but not recommended for versions too high)
- **OpenCV:** For image preprocessing
- **Pandas, NumPy, scikit-learn:** For data handling and manipulation

You can use the provided YAML file to create a compatible environment. 

## Installation

1. Navigate to the project directory:

   ```bash
   cd Project
   ```

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages or simply create the env without any     
   hassle using the provided YAML file:

   ```bash
   conda env create -f mlenv.yaml
   conda activate mlearn
   ```

   If using `pip`, manually install the required packages listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

1. Ensure the following directory structure:

   ```
   x/x/x/Project
       HAM10000_metadata.csv
       HAM10000_images_part_1/
       HAM10000_images_part_2/
   ```

2. Place your images and metadata CSV in the appropriate directories.

## Running the Code

1. Run the script:

   ```bash
   python models/inception_v3_xxxx.py
   ```

   This will execute the model training pipeline, including data loading, augmentation, training, and evaluation.

## Notes

- The model was trained on a CPU with the following specifications:
  - **Intel(R) Core(TM) i5-6500 CPU @ 3.20GHz**
  - **Base speed:** 3.19 GHz
  - **Cores:** 4
  - **Logical processors:** 4
  - **Virtualization:** Enabled

- The project was originally developed as a high school project. Due to limited computational resources, the code is optimized for CPU usage.

- There were some issues with Jupyter; comments have been added to clarify parts of the code.


## Acknowledgements

- [HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- [OpenCV Documentation](https://docs.opencv.org/)
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)
- [TensorFlow](https://www.tensorflow.org/)
- [Plotly](https://plotly.com/)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
