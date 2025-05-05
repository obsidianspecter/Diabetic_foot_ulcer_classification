import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import ResNet50
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Constants
IMG_SIZE = (224, 224)  # ResNet50 default input size
BATCH_SIZE = 8
EPOCHS = 100
NUM_CLASSES = 3  # normal, moderate, severe
LEARNING_RATE = 0.0001  # Reduced learning rate for transfer learning

def focal_loss(gamma=2., alpha=.25):
    """Create a focal loss function for handling class imbalance."""
    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Calculate focal loss
        ce = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1 - y_pred, gamma) * y_true
        fl = alpha * weight * ce
        
        # Reduce mean over all samples and classes
        return tf.reduce_mean(tf.reduce_sum(fl, axis=1))
    return focal_loss_fixed

def create_advanced_augmentation():
    """Create an advanced data augmentation pipeline."""
    return tf.keras.Sequential([
        # Geometric transformations
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.3),
        layers.RandomZoom(0.3),
        layers.RandomTranslation(0.2, 0.2),
        
        # Color and intensity transformations
        layers.RandomBrightness(0.4),
        layers.RandomContrast(0.4),
        
        # Adding random noise
        layers.GaussianNoise(0.1),
        
        # Randomly dropping pixels (simulating occlusions)
        layers.Dropout(0.1),
    ])

def mixup(x1, x2, y1, y2, alpha=0.2):
    """Perform mixup augmentation on the batch."""
    batch_size = tf.shape(x1)[0]
    
    # Generate random mixup weights
    weights = tf.random.beta(alpha, alpha, batch_size)
    x1_weights = tf.reshape(weights, (batch_size, 1, 1, 1))
    x2_weights = tf.reshape(weights, (batch_size, 1, 1, 1))
    y_weights = tf.reshape(weights, (batch_size, 1))
    
    # Mix the images and labels
    mixed_x1 = x1 * x1_weights + x1 * (1 - x1_weights)
    mixed_x2 = x2 * x2_weights + x2 * (1 - x2_weights)
    mixed_y = y1 * y_weights + y2 * (1 - y_weights)
    
    return mixed_x1, mixed_x2, mixed_y

def load_and_preprocess_image(image_path, is_thermal=False):
    """Load and preprocess image."""
    if is_thermal:
        # Load TIFF image using PIL
        img = Image.open(image_path)
        img = np.array(img)
        
        # Normalize thermal image
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = (img * 255).astype(np.uint8)
        
        # Convert to RGB and repeat channels for thermal
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        # Load RGB image using OpenCV
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0  # Normalize
    return img

def create_dataset(data_dir):
    """Create dataset from directory structure."""
    rgb_images = []
    thermal_images = []
    labels = []
    
    categories = ['normal', 'moderate', 'severe']
    
    for idx, category in enumerate(categories):
        category_path = os.path.join(data_dir, category)
        
        # Get RGB images
        rgb_path = os.path.join(category_path, 'rgb')
        if not os.path.exists(rgb_path):
            continue
            
        for img_name in os.listdir(rgb_path):
            if img_name.endswith('.png'):
                # Load RGB image
                rgb_img_path = os.path.join(rgb_path, img_name)
                rgb_img = load_and_preprocess_image(rgb_img_path, is_thermal=False)
                
                # Load corresponding thermal image
                thermal_img_name = img_name.replace('.png', '.tiff')
                thermal_img_path = os.path.join(category_path, 'thermal', thermal_img_name)
                
                if os.path.exists(thermal_img_path):
                    thermal_img = load_and_preprocess_image(thermal_img_path, is_thermal=True)
                    
                    rgb_images.append(rgb_img)
                    thermal_images.append(thermal_img)
                    labels.append(idx)
    
    return np.array(rgb_images), np.array(thermal_images), np.array(labels)

def create_custom_branch(input_shape, name_prefix):
    """Create a custom CNN branch with explicit layer naming."""
    inputs = Input(shape=input_shape, name=f'{name_prefix}_input')
    
    # Advanced data augmentation only for training
    augmented = create_advanced_augmentation()(inputs)
    
    # First block with residual connection
    x = layers.Conv2D(64, (3, 3), padding='same', name=f'{name_prefix}_conv1_1')(augmented)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn1_1')(x)
    x = layers.Activation('relu', name=f'{name_prefix}_act1_1')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', name=f'{name_prefix}_conv1_2')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn1_2')(x)
    shortcut = layers.Conv2D(64, (1, 1), name=f'{name_prefix}_shortcut1')(augmented)
    x = layers.Add(name=f'{name_prefix}_add1')([x, shortcut])
    x = layers.Activation('relu', name=f'{name_prefix}_act1_2')(x)
    x = layers.MaxPooling2D((2, 2), name=f'{name_prefix}_pool1')(x)
    x = layers.Dropout(0.3, name=f'{name_prefix}_drop1')(x)
    
    # Second block with residual connection
    prev = x
    x = layers.Conv2D(128, (3, 3), padding='same', name=f'{name_prefix}_conv2_1')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn2_1')(x)
    x = layers.Activation('relu', name=f'{name_prefix}_act2_1')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', name=f'{name_prefix}_conv2_2')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn2_2')(x)
    shortcut = layers.Conv2D(128, (1, 1), name=f'{name_prefix}_shortcut2')(prev)
    x = layers.Add(name=f'{name_prefix}_add2')([x, shortcut])
    x = layers.Activation('relu', name=f'{name_prefix}_act2_2')(x)
    x = layers.MaxPooling2D((2, 2), name=f'{name_prefix}_pool2')(x)
    x = layers.Dropout(0.3, name=f'{name_prefix}_drop2')(x)
    
    # Third block with residual connection
    prev = x
    x = layers.Conv2D(256, (3, 3), padding='same', name=f'{name_prefix}_conv3_1')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn3_1')(x)
    x = layers.Activation('relu', name=f'{name_prefix}_act3_1')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', name=f'{name_prefix}_conv3_2')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn3_2')(x)
    shortcut = layers.Conv2D(256, (1, 1), name=f'{name_prefix}_shortcut3')(prev)
    x = layers.Add(name=f'{name_prefix}_add3')([x, shortcut])
    x = layers.Activation('relu', name=f'{name_prefix}_act3_2')(x)
    x = layers.MaxPooling2D((2, 2), name=f'{name_prefix}_pool3')(x)
    x = layers.Dropout(0.3, name=f'{name_prefix}_drop3')(x)
    
    # Global pooling and dense layers with L2 regularization
    x = layers.GlobalAveragePooling2D(name=f'{name_prefix}_gap')(x)
    x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.01), name=f'{name_prefix}_dense1')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn_dense1')(x)
    x = layers.Activation('relu', name=f'{name_prefix}_act_dense1')(x)
    x = layers.Dropout(0.5, name=f'{name_prefix}_drop_dense1')(x)
    
    return inputs, x

def create_dual_input_model():
    """Create a dual-input model using custom CNN branches."""
    # Create the two branches
    rgb_input, rgb_features = create_custom_branch((*IMG_SIZE, 3), 'rgb')
    thermal_input, thermal_features = create_custom_branch((*IMG_SIZE, 3), 'thermal')
    
    # Combine features
    combined = layers.concatenate([rgb_features, thermal_features], name='concat_features')
    
    # Classification layers with L2 regularization
    x = layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='relu', name='dense_1')(combined)
    x = layers.Dropout(0.5, name='dropout_1')(x)
    x = layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='relu', name='dense_2')(x)
    x = layers.Dropout(0.5, name='dropout_2')(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)
    
    # Create model
    model = Model(inputs=[rgb_input, thermal_input], outputs=outputs, name='foot_ulcer_classifier')
    return model

def plot_training_history(history):
    """Plot training history."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    rgb_images, thermal_images, labels = create_dataset('local_database_Processed')
    
    print(f"Dataset size: {len(rgb_images)} samples")
    print(f"Class distribution: {np.bincount(labels)}")
    
    # Calculate class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weight_dict = dict(enumerate(class_weights))
    print("\nClass weights:", class_weight_dict)
    
    # Convert labels to categorical
    labels_cat = tf.keras.utils.to_categorical(labels, NUM_CLASSES)
    
    # Split the data
    X_rgb_train, X_rgb_test, X_thermal_train, X_thermal_test, y_train, y_test = train_test_split(
        rgb_images, thermal_images, labels_cat, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create and compile model
    print("\nCreating and compiling model...")
    model = create_dual_input_model()
    
    # Use focal loss for better handling of class imbalance
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy']
    )
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,  # Increased patience
        restore_best_weights=True
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=10,  # Increased patience
        min_lr=0.00001
    )
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        [X_rgb_train, X_thermal_train],
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        class_weight=class_weight_dict,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Save training history
    np.save('training_history.npy', history.history)
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(
        [X_rgb_test, X_thermal_test],
        y_test,
        verbose=0
    )
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    # Generate predictions
    y_pred = model.predict([X_rgb_test, X_thermal_test])
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test_classes, y_pred_classes, 
                              target_names=['Normal', 'Moderate', 'Severe']))
    
    # Plot training history
    plot_training_history(history)
    
    # Save model
    model.save('foot_ulcer_classifier.keras')
    print("\nModel saved as 'foot_ulcer_classifier.keras'")

if __name__ == "__main__":
    main() 