import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import tensorflow as tf
import os

def plot_detailed_training_history(history_dict):
    """Create detailed plots of training metrics."""
    # Set up the style
    plt.style.use('seaborn-v0_8')
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Accuracy Plot
    plt.subplot(3, 2, 1)
    plt.plot(history_dict['accuracy'], label='Training', linewidth=2)
    plt.plot(history_dict['val_accuracy'], label='Validation', linewidth=2)
    plt.title('Model Accuracy Over Time', fontsize=12, pad=15)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 2. Loss Plot
    plt.subplot(3, 2, 2)
    plt.plot(history_dict['loss'], label='Training', linewidth=2)
    plt.plot(history_dict['val_loss'], label='Validation', linewidth=2)
    plt.title('Model Loss Over Time', fontsize=12, pad=15)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 3. Learning Rate Plot
    plt.subplot(3, 2, 3)
    plt.plot(history_dict['learning_rate'], label='Learning Rate', linewidth=2, color='green')
    plt.title('Learning Rate Decay', fontsize=12, pad=15)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    
    # 4. Training vs Validation Performance
    plt.subplot(3, 2, 4)
    train_acc = history_dict['accuracy'][-1]
    val_acc = history_dict['val_accuracy'][-1]
    plt.bar(['Training', 'Validation'], [train_acc, val_acc])
    plt.title('Final Training vs Validation Accuracy', fontsize=12, pad=15)
    plt.ylabel('Accuracy')
    for i, v in enumerate([train_acc, val_acc]):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    plt.grid(True)
    
    # 5. Convergence Plot (Loss vs Accuracy)
    plt.subplot(3, 2, (5, 6))
    plt.scatter(history_dict['loss'], history_dict['accuracy'], 
               label='Training', alpha=0.5, s=100)
    plt.scatter(history_dict['val_loss'], history_dict['val_accuracy'], 
               label='Validation', alpha=0.5, s=100)
    plt.title('Model Convergence (Loss vs Accuracy)', fontsize=12, pad=15)
    plt.xlabel('Loss')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('detailed_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_class_distribution(labels):
    """Plot the class distribution in the dataset."""
    plt.figure(figsize=(10, 6))
    class_names = ['Normal', 'Moderate', 'Severe']
    class_counts = np.bincount(labels)
    
    # Create bar plot
    sns.barplot(x=class_names, y=class_counts)
    plt.title('Class Distribution in Dataset', fontsize=12, pad=15)
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    
    # Add value labels on top of each bar
    for i, count in enumerate(class_counts):
        plt.text(i, count, str(count), ha='center', va='bottom')
    
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load the saved model
    print("Loading model...")
    model = load_model('foot_ulcer_classifier.keras')
    
    # Load the training history
    if os.path.exists('training_history.npy'):
        print("Loading training history...")
        history_dict = np.load('training_history.npy', allow_pickle=True).item()
        
        # Create detailed plots
        print("Generating detailed training history plots...")
        plot_detailed_training_history(history_dict)
        
        print("Plots have been saved as 'detailed_training_history.png'")
    else:
        print("No training history found. Please make sure 'training_history.npy' exists.")

if __name__ == "__main__":
    main() 