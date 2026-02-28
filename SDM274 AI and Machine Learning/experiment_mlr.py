from logistic_regression import MulticlassLogisticRegression as MLR
import numpy as np
import matplotlib.pyplot as plt
import time
from metrics import evaluate_and_visualize
import os

output_image_dir = "images/mlr_experiments"
os.makedirs(output_image_dir, exist_ok=True)

train_set = np.load('input/train_set.npy')
train_label = np.load('input/train_label.npy')
test_set = np.load('input/test_set.npy')
test_label = np.load('input/test_label.npy')

epoch_list = np.arange(0, 1001, 100)
accuracy_list = []
time_list = []
losses_list = []
models_list = []

for n_epoch in epoch_list:
    print(f"Training with epochs = {n_epoch}")
    Model = MLR(reg_lambda=0.001)
    
    start_time = time.time()
    losses, w = Model.fit(train_set, train_label, epochs=n_epoch, lr=0.001, batch_size=128, normalization=True)
    elapsed_time = time.time() - start_time
    
    overall_acc, _, _, _, _ = Model.accuracy(test_set, test_label)
    print(f"Accuracy: {overall_acc:.4f}%, Training Time: {elapsed_time:.2f} s\n")
    
    accuracy_list.append(overall_acc)
    time_list.append(elapsed_time)
    losses_list.append(losses)
    models_list.append(Model)

fig, ax1 = plt.subplots(figsize=(10,6))

color1 = 'tab:blue'
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy (%)', color=color1)
ax1.plot(epoch_list, accuracy_list, 'o-', color=color1, label='Accuracy')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True)

ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Training Time (s)', color=color2)
ax2.plot(epoch_list, time_list, 's--', color=color2, label='Training Time')
ax2.tick_params(axis='y', labelcolor=color2)

fig.tight_layout()
fig.legend(loc='upper left', bbox_to_anchor=(0.15,0.85))
plt.title("Accuracy and Training Time vs Epochs")
plt.savefig(f"{output_image_dir}/Accuracy_and_Training_Time_vs_Epochs.png", dpi=300)
plt.show()

best_idx = np.argmax(accuracy_list)
best_epoch = epoch_list[best_idx]
best_model = models_list[best_idx]
best_loss = losses_list[best_idx]
print(f"\nBest epoch: {best_epoch}, Test Accuracy: {accuracy_list[best_idx]:.2f}%")

plt.figure(figsize=(10,6))
for i, losses in enumerate(losses_list):
    plt.plot(range(len(losses)), losses, alpha=0.3, color='blue')
plt.plot(range(len(best_loss)), best_loss, color='red', label=f'Best Epoch = {best_epoch}')
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.title("Training Loss Curves for Different Epoch Settings")
plt.savefig(f"{output_image_dir}/Training_Loss_Curves_for_Different_Epoch_Settings.png", dpi=300)
plt.legend()
plt.grid(True)
plt.show()

CLASS_NAMES = [str(i) for i in range(10)]
overall_acc, y_pred, y_true, correct_indices, wrong_indices = best_model.accuracy(test_set, test_label)
evaluation_title_prefix = f"MLR_MNIST_BestEpoch_{best_epoch}"
evaluation_save_path_prefix = os.path.join(output_image_dir, f"MLR_MNIST_BestEpoch_{best_epoch}")

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
