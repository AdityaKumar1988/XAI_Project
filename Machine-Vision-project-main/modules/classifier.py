import os
import numpy as np
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentClassifier:
    """Handles training and prediction for RF and CNN models."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rf_path = os.path.join("models", "rf_model.pkl")
        self.cnn_path = os.path.join("models", "cnn_model.h5")
        self.cnn_model = None
        os.makedirs("models", exist_ok=True)

    # --- PART A: Random Forest ---
    def train_rf(self, X_train, y_train):
        self.logger.info("Training Random Forest model.")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        joblib.dump(rf, self.rf_path)
        self.logger.info(f"RF model saved to {self.rf_path}")
        return rf

    def predict_rf(self, feature_vector):
        if not os.path.exists(self.rf_path):
            raise FileNotFoundError(f"Model not found at {self.rf_path}. Train first.")
        rf = joblib.load(self.rf_path)
        # Reshape for single prediction
        feat = feature_vector.reshape(1, -1)
        pred = rf.predict(feat)[0]
        prob = rf.predict_proba(feat)[0][1] if pred == 1 else rf.predict_proba(feat)[0][0]
        return pred, prob

    # --- PART B: CNN ---
    def build_cnn(self, input_shape=(224, 224, 3)):
        self.logger.info("Building CNN architecture using MobileNetV2 Transfer Learning.")
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False  # Freeze base layers initially

        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        # Using a slightly lower learning rate for stability
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_cnn(self, X_train_images, y_train):
        self.logger.info("Training CNN model with Data Augmentation and official preprocessing.")
        
        # 1. Official Preprocessing (MobileNetV2 expects -1 to 1)
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        X_train_preprocessed = preprocess_input(X_train_images * 255.0)

        # 2. Data Augmentation
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,
            fill_mode='nearest',
            validation_split=0.2
        )

        model = self.build_cnn()
        
        callbacks = [
            EarlyStopping(patience=8, restore_best_weights=True, monitor='val_loss'),
            ModelCheckpoint(self.cnn_path, save_best_only=True, monitor='val_loss')
        ]
        
        # Train with augmented data
        train_generator = datagen.flow(X_train_preprocessed, y_train, batch_size=32, subset='training')
        val_generator = datagen.flow(X_train_preprocessed, y_train, batch_size=32, subset='validation')

        history = model.fit(
            train_generator,
            epochs=50,
            validation_data=val_generator,
            callbacks=callbacks
        )
        
        # Optional: Fine-tuning step (unfreeze top layers)
        self.logger.info("Unfreezing base model for fine-tuning...")
        model.layers[0].trainable = True
        for layer in model.layers[0].layers[:-20]:
            layer.trainable = False
            
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
        
        model.fit(
            train_generator,
            epochs=20,
            validation_data=val_generator,
            callbacks=callbacks
        )
        
        self.logger.info(f"CNN model saved to {self.cnn_path}")
        return history

    def predict_cnn(self, image):
        if self.cnn_model is None:
            if not os.path.exists(self.cnn_path):
                raise FileNotFoundError(f"Model not found at {self.cnn_path}. Train first.")
            self.cnn_model = load_model(self.cnn_path)
        
        # Preprocess for CNN (ensure RGB)
        import cv2
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        img = cv2.resize(image, (224, 224))
        
        # Use official preprocessing
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img.astype(float))
        
        pred_prob = self.cnn_model.predict(img)[0][0]
        label = 1 if pred_prob > 0.5 else 0
        prob = pred_prob if label == 1 else 1 - pred_prob
        
        return label, prob

if __name__ == "__main__":
    clf = DocumentClassifier()
    print("Classifier module ready.")