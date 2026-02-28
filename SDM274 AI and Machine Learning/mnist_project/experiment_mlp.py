import numpy as np
import matplotlib.pyplot as plt
import time
import os
from metrics import evaluate_and_visualize
from mlp_classifier import MultiLayerPerceptron

output_image_dir = "images/mlp_experiments"
os.makedirs(output_image_dir, exist_ok=True)

train_set = np.load('input/train_set.npy')
train_label = np.load('input/train_label.npy')
test_set = np.load('input/test_set.npy')
test_label = np.load('input/test_label.npy')

epoch_list = [100, 500, 1000]
hidden_layer_configs = [
    [train_set.shape[1], 64, 10],
    [train_set.shape[1], 128, 10],
    [train_set.shape[1], 128, 64, 10]
]

best_accuracy = -1
best_model = None
best_loss = None
best_config = None
best_epoch = None

for n_epoch in epoch_list:
    for layer_dims in hidden_layer_configs:
        print(f"\nTraining MLP with layers {layer_dims}, epochs={n_epoch}")
        model = MultiLayerPerceptron(layer_dims=layer_dims, reg_lambda=0.001, normalization=True)
        
        start_time = time.time()
        losses = model.fit(train_set, train_label, epochs=n_epoch, lr=0.001, batch_size=128)
        elapsed_time = time.time() - start_time
        
        accuracy, _, _, _, _ = model.accuracy(test_set, test_label)
        print(f"Test Accuracy: {accuracy:.2f}%, Training Time: {elapsed_time:.2f}s")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_loss = losses
            best_config = layer_dims
            best_epoch = n_epoch

print(f"\nBest model: layers={best_config}, epochs={best_epoch}, accuracy={best_accuracy:.2f}%")

plt.figure(figsize=(10,6))
plt.plot(range(len(best_loss)), best_loss, color='red', label=f'Best Model Loss')
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.title("Training Loss Curve of Best Model")
plt.legend()
plt.grid(True)
plt.savefig(f"{output_image_dir}/Training_Loss_Curve_of_Best_Model.png", dpi=300)
plt.show()

CLASS_NAMES = [str(i) for i in range(10)]
overall_acc, y_pred, y_true, correct_indices, wrong_indices = best_model.accuracy(test_set, test_label)
evaluation_title_prefix = f"MLP_BestModel_{best_epoch}"
evaluation_save_path_prefix = os.path.join(output_image_dir, f"MLP_BestModel_{best_epoch}")

evaluate_and_visualize(
    overall_acc,
    y_pred,
    y_true,
    correct_indices,
    wrong_indices,
    test_set,
    class_names=CLASS_NAMES,
    num_display_images=8,
    title_prefix=evaluation_title_prefix,
    save_path_prefix=evaluation_save_path_prefix
)
