import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix
import cv2
import os

def preprocess_image(img, target_size=(128, 128)):
    if isinstance(img, str):  # If it's a file path
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    return img

def load_data(csv_path, image_folder1, image_folder2):
    df = pd.read_csv(csv_path)
    df['path'] = df.apply(lambda row: os.path.join(image_folder1, f"{row['image_id']}.jpg") 
                          if os.path.exists(os.path.join(image_folder1, f"{row['image_id']}.jpg"))
                          else os.path.join(image_folder2, f"{row['image_id']}.jpg"), axis=1)
    return df

def calculate_metrics(model, generator, class_names):
    # Predict on the validation set
    y_pred = model.predict(generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = generator.classes

    # Calculate overall metrics
    f1 = f1_score(y_true, y_pred_classes, average='weighted')
    precision = precision_score(y_true, y_pred_classes, average='weighted')
    recall = recall_score(y_true, y_pred_classes, average='weighted')

    print("Overall Metrics:")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Calculate and print metrics for each class
    class_report = classification_report(y_true, y_pred_classes, target_names=class_names, output_dict=True)
    
    print("\nPer-class Metrics:")
    for class_name in class_names:
        print(f"{class_name}:")
        print(f"  F1 Score: {class_report[class_name]['f1-score']:.4f}")
        print(f"  Precision: {class_report[class_name]['precision']:.4f}")
        print(f"  Recall: {class_report[class_name]['recall']:.4f}")

    # Calculate and plot confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plot_confusion_matrix(cm, class_names)

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'")

def main():
    # Load the trained model
    model = load_model('l.h5')

    # Load and prepare the validation data
    val_df = load_data('HAM10000_metadata.csv', 'HAM10000_images_part_1', 'HAM10000_images_part_2')
    
    # You might want to use a subset of data for validation, e.g., 20% of the data
    val_df = val_df.sample(frac=0.2, random_state=42)

    # Prepare the data generator
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_image)
    
    val_generator = val_datagen.flow_from_dataframe(
        val_df,
        x_col='path',
        y_col='dx',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    # Get class names
    class_names = list(val_generator.class_indices.keys())

    # Calculate and print metrics, including confusion matrix
    calculate_metrics(model, val_generator, class_names)

if __name__ == "__main__":
    main()