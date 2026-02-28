from decision_tree import DecisionTree
import numpy as np
import matplotlib.pyplot as plt
import time
from metrics import evaluate_and_visualize
import os

output_image_dir = "images/dt_experiments"
os.makedirs(output_image_dir, exist_ok=True)

train_set = np.load('input/train_set.npy')
train_label = np.load('input/train_label.npy')
test_set = np.load('input/test_set.npy')
test_label = np.load('input/test_label.npy')

max_depth_list = [2]
accuracy_list = []
time_list = []
models_list = []

for max_depth in max_depth_list:
    print(f"Training Decision Tree with max_depth = {max_depth}")
    Model = DecisionTree(max_depth=max_depth)
    
    start_time = time.time()
    Model.fit(train_set, train_label)
    elapsed_time = time.time() - start_time
    
    overall_acc, _, _, _, _ = Model.accuracy(test_set, test_label)
    print(f"Accuracy: {overall_acc:.2f}%, Training Time: {elapsed_time:.2f} s\n")
    
    accuracy_list.append(overall_acc)
    time_list.append(elapsed_time)
    models_list.append(Model)

fig, ax1 = plt.subplots(figsize=(10,6))

color1 = 'tab:blue'
ax1.set_xlabel('Max Depth')
ax1.set_ylabel('Accuracy (%)', color=color1)
ax1.plot([str(d) for d in max_depth_list], accuracy_list, 'o-', color=color1, label='Accuracy')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True)

ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Training Time (s)', color=color2)
ax2.plot([str(d) for d in max_depth_list], time_list, 's--', color=color2, label='Training Time')
ax2.tick_params(axis='y', labelcolor=color2)

fig.tight_layout()
plt.subplots_adjust(top=0.88)
plt.title("Decision Tree Accuracy and Training Time vs Max Depth", fontweight="bold")
plt.savefig(f"{output_image_dir}/Decision_Tree_Accuracy_and_Training_Time_vs_Max_Depth.png", dpi=300)
plt.show()

best_idx = np.argmax(accuracy_list)
best_depth = max_depth_list[best_idx]
best_model = models_list[best_idx]
print(f"\nBest max_depth: {best_depth}, Test Accuracy: {accuracy_list[best_idx]:.2f}%")

CLASS_NAMES = [str(i) for i in range(10)]
overall_acc, y_pred, y_true, correct_indices, wrong_indices = best_model.accuracy(test_set, test_label)
evaluation_title_prefix = f"DecisionTree_BestDepth_{best_depth}"
evaluation_save_path_prefix = os.path.join(output_image_dir, f"DecisionTree_BestDepth_{best_depth}")

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
