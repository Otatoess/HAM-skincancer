import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import cv2
import tensorflow as tf

# Limit TensorFlow to only use CPU
tf.config.set_visible_devices([], 'GPU')

# Load and organize data
def load_data(csv_path, image_folder1, image_folder2):
    df = pd.read_csv(csv_path)
    df['path'] = df.apply(lambda row: os.path.join(image_folder1, f"{row['image_id']}.jpg") 
                          if os.path.exists(os.path.join(image_folder1, f"{row['image_id']}.jpg"))
                          else os.path.join(image_folder2, f"{row['image_id']}.jpg"), axis=1)
    return df

# Image preprocessing
def preprocess_image(img, target_size=(128, 128)):
    if isinstance(img, str):  # If it's a file path
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    return img

# Create InceptionV3 model
def create_inceptionv3_model(num_classes):
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Fine-tune the last few layers
    for layer in base_model.layers[-30:]:
        layer.trainable = True
    
    return model

def get_class_weights(y):
    # Define the importance of each class
    class_importance = {
        'mel': 2.5,  # Melanoma - highest importance
        'bcc': 2.0,  # Basal cell carcinoma - high importance
        'akiec': 2.0,  # Actinic keratosis - high importance
        'bkl': 1.5,  # Benign keratosis-like lesions - medium importance
        'df': 1.2,  # Dermatofibroma - slightly higher than base
        'vasc': 1.2,  # Vascular lesions - slightly higher than base
        'nv': 1.0   # Melanocytic nevi (benign) - base importance
    }
    
    class_weights = {}
    total_samples = len(y)
    
    for idx, cls in enumerate(np.unique(y)):
        class_weights[idx] = (1 / (np.sum(y == cls) / total_samples)) * class_importance[cls]
    
    return class_weights

# Main pipeline
def main():
    # Load data
    df = load_data('HAM10000_metadata.csv', 'HAM10000_images_part_1', 'HAM10000_images_part_2')
    
    # Balance dataset
    total_samples = 20000
    num_classes = df['dx'].nunique()
    samples_per_class = total_samples // num_classes
    
    balanced_df = df.groupby('dx').apply(lambda x: x.sample(n=samples_per_class, replace=True)).reset_index(drop=True)
    
    # Prepare labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(balanced_df['dx'])
    
    # Calculate class weights
    class_weights = get_class_weights(balanced_df['dx'])
    
    # Split data
    train_df, val_df = train_test_split(balanced_df, test_size=0.2, stratify=balanced_df['dx'], random_state=42)
    
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=preprocess_image
    )
    
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_image)
    
    # Create generators
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='path',
        y_col='dx',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical'
    )
    
    val_generator = val_datagen.flow_from_dataframe(
        val_df,
        x_col='path',
        y_col='dx',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    # Create model
    model = create_inceptionv3_model(num_classes)
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    
    # Train model with class weights
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=40,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=[early_stopping, reduce_lr],
        workers=2,
        use_multiprocessing=False,
        class_weight=class_weights
    )
    
    # Evaluate model
    val_loss, val_accuracy = model.evaluate(val_generator)
    print(f"Validation accuracy: {val_accuracy:.4f}")
    
    # Save the final best model
    model.save('final_best_inceptionv3_model.h5')

if __name__ == "__main__":
    main()
