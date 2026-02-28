import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import os
from recover_image_data import recover_image_data

def evaluate_and_visualize(overall_accuracy, y_pred, y_true, correct_indices, wrong_indices, X_test_raw, 
                           class_names=None, num_display_images=5,
                           title_prefix="", save_path_prefix=""):
    print("\n--- Evaluation Metrics ---")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")

    num_classes = len(np.unique(y_true))
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    print("\nPer-class Metrics:")
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    
    for class_id in range(num_classes):
        label = class_names[class_id]
        if label in report:
            metrics = report[label]
            if isinstance(metrics, dict):
                print(f"  Class {label}:")
                print(f"    Precision: {metrics['precision']:.4f}")
                print(f"    Recall:    {metrics['recall']:.4f}")
                print(f"    F1-score:  {metrics['f1-score']:.4f}")

    # 3. Macro-averaged F1-score
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"\nMacro-averaged F1-score: {macro_f1:.4f}")

    # 4. Weighted-average F1-score
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f"Weighted-average F1-score: {weighted_f1:.4f}")

    # 5. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    
    cm_title = f"Confusion Matrix\n{title_prefix}" if title_prefix else "Confusion Matrix"
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    if save_path_prefix:
        os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)
        plt.savefig(f"{save_path_prefix}_confusion_matrix.png", dpi=300)
        print(f"Confusion Matrix saved to: {save_path_prefix}_confusion_matrix.png")
    
    plt.show()

    print(f"\n--- Displaying {num_display_images} Correctly Predicted Images ---")
    if len(correct_indices) > 0:
        plt.figure(figsize=(15, 3))
        chosen_indices = np.random.choice(correct_indices, min(len(correct_indices), num_display_images), replace=False)
        
        for i, idx in enumerate(chosen_indices):
            plt.subplot(1, num_display_images, i + 1)
            decoded_image = recover_image_data(X_test_raw[idx])
            plt.imshow(decoded_image, cmap='gray')
            plt.title(f"True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}")
            plt.axis('off')
        plt.tight_layout()
        
        if save_path_prefix:
            plt.savefig(f"{save_path_prefix}_correct_predictions.png", dpi=300)
            print(f"Correct predictions images saved to: {save_path_prefix}_correct_predictions.png")
        
        plt.show()
    else:
        print("No correctly predicted images to display.")

    print(f"\n--- Displaying {num_display_images} Incorrectly Predicted Images ---")
    if len(wrong_indices) > 0:
        plt.figure(figsize=(15, 3))
        chosen_indices = np.random.choice(wrong_indices, min(len(wrong_indices), num_display_images), replace=False)

        for i, idx in enumerate(chosen_indices):
            plt.subplot(1, num_display_images, i + 1)
            decoded_image = recover_image_data(X_test_raw[idx])
            plt.imshow(decoded_image, cmap='gray')
            plt.title(f"True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}", color='red')
            plt.axis('off')
        plt.tight_layout()
        
        if save_path_prefix:
            plt.savefig(f"{save_path_prefix}_incorrect_predictions.png", dpi=300)
            print(f"Incorrect predictions images saved to: {save_path_prefix}_incorrect_predictions.png")
        
        plt.show()
    else:
        print("No incorrectly predicted images to display.")

