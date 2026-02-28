from knn_classifier import KnnClassifier
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from metrics import evaluate_and_visualize

output_image_dir = "images/knn_experiments"
os.makedirs(output_image_dir, exist_ok=True)

print("Loading data...")
full_train_set = np.load('input/train_set.npy')
full_train_label = np.load('input/train_label.npy')
full_test_set = np.load('input/test_set.npy')
full_test_label = np.load('input/test_label.npy')
print("Data loaded.")

N_TEST_FIXED = 200 
sub_test_set = full_test_set[:N_TEST_FIXED, :]
sub_test_label = full_test_label[:N_TEST_FIXED]

NUM_CLASSES = 10 
CLASS_NAMES = [str(i) for i in range(NUM_CLASSES)] 

K_START, K_END = 1, 10 
K_VALUES = np.arange(K_START, K_END) 

N_TRAIN_EXP1 = 2500
sub_train_set_exp1 = full_train_set[:N_TRAIN_EXP1, :]
sub_train_label_exp1 = full_train_label[:N_TRAIN_EXP1]

TRAIN_SIZES_EXP2 = [200, 2000, 5000, 10000] 

def plot_results_curves(x_values, data_dict, title, xlabel, ylabel, filename, is_accuracy=True, N_TRAIN=None, N_TEST=None):
    plt.figure(figsize=(12, 8))
    plot_configs = {
        'BruteForce_NoNorm': {'linestyle': '-', 'marker': 'o', 'color': 'red', 'label': 'BruteForce without Norm'},
        'BruteForce_WithNorm': {'linestyle': '-', 'marker': 's', 'color': 'purple', 'label': 'BruteForce with Norm'},
        'KdTree_NoNorm': {'linestyle': '--', 'marker': '^', 'color': 'blue', 'label': 'KdTree without Norm'},
        'KdTree_WithNorm': {'linestyle': '--', 'marker': 'D', 'color': 'green', 'label': 'KdTree with Norm'}
    }
    active_configs = {k: v for k, v in plot_configs.items() if k in data_dict}
    for config_name, params in active_configs.items():
        if config_name in data_dict and data_dict[config_name]:
            valid_x = [x_values[i] for i, val in enumerate(data_dict[config_name]) if not np.isnan(val)]
            valid_y = [val for val in data_dict[config_name] if not np.isnan(val)]
            if valid_x and valid_y:
                plt.plot(valid_x, valid_y, 
                         linestyle=params['linestyle'], marker=params['marker'], 
                         color=params['color'], 
                         linewidth=2.5, markersize=6, alpha=0.9, label=params['label'])
    plt.legend(fontsize=12)
    plt.xlabel(xlabel, fontsize=13)
    if is_accuracy:
        plt.ylabel(ylabel + " (%)", fontsize=13)
        plt.ylim(0, 100)
    else:
        plt.ylabel(ylabel + " (seconds)", fontsize=13)
        plt.yscale('log')
    if isinstance(x_values, (np.ndarray, list)):
        plt.xticks(x_values)
    full_title = title
    if N_TRAIN is not None and N_TEST is not None:
        full_title += f"\n (Train:{N_TRAIN} Test:{N_TEST})"
    elif N_TRAIN is not None:
         full_title += f"\n (Train:{N_TRAIN})"
    elif N_TEST is not None:
         full_title += f"\n (Test:{N_TEST})"
    plt.title(full_title, fontsize=14, fontweight="bold", pad=15)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_image_dir, filename), dpi=300)
    plt.show()

def plot_bar_chart(labels, data_values, title, ylabel, filename, N_TRAIN=None, N_TEST=None):
    plt.figure(figsize=(10, 6))
    colors = ['red', 'purple', 'blue', 'green']
    color_map = {
        'BruteForce_NoNorm': 'red',
        'BruteForce_WithNorm': 'purple',
        'KdTree_NoNorm': 'blue',
        'KdTree_WithNorm': 'green'
    }
    bar_colors = [color_map.get(label, colors[i % len(colors)]) for i, label in enumerate(labels)]
    plt.bar(labels, data_values, color=bar_colors)
    plt.ylabel(ylabel, fontsize=13)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    full_title = title
    if N_TRAIN is not None and N_TEST is not None:
        full_title += f"\n (Train:{N_TRAIN} Test:{N_TEST})"
    elif N_TRAIN is not None:
         full_title += f"\n (Train:{N_TRAIN})"
    elif N_TEST is not None:
         full_title += f"\n (Test:{N_TEST})"
    plt.title(full_title, fontsize=14, fontweight="bold", pad=15)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_image_dir, filename), dpi=300)
    plt.show()

# =========================================================================================
# --- Experiment 1: K-value impact on Accuracy and Time (all 4 configurations) ---
# =========================================================================================
print("\n--- Running Experiment 1: K-value impact on Accuracy and Time ---")
print(f"  Fixed training size: {N_TRAIN_EXP1}, Fixed testing size: {N_TEST_FIXED}")

results_exp1 = {
    'KdTree_NoNorm': {'accuracies': [], 'times': [], 'KnnInstance': None},
    'KdTree_WithNorm': {'accuracies': [], 'times': [], 'KnnInstance': None},
    'BruteForce_NoNorm': {'accuracies': [], 'times': [], 'KnnInstance': None},
    'BruteForce_WithNorm': {'accuracies': [], 'times': [], 'KnnInstance': None}
}

print("\n  Initializing KnnClassifier instances for Experiment 1...")
results_exp1['KdTree_NoNorm']['KnnInstance'] = KnnClassifier(sub_train_set_exp1, sub_train_label_exp1, use_kdtree=True, normalization=False)
results_exp1['KdTree_WithNorm']['KnnInstance'] = KnnClassifier(sub_train_set_exp1, sub_train_label_exp1, use_kdtree=True, normalization=True)
results_exp1['BruteForce_NoNorm']['KnnInstance'] = KnnClassifier(sub_train_set_exp1, sub_train_label_exp1, use_kdtree=False, normalization=False)
results_exp1['BruteForce_WithNorm']['KnnInstance'] = KnnClassifier(sub_train_set_exp1, sub_train_label_exp1, use_kdtree=False, normalization=True)
print("  KnnClassifier instances initialized.")

total_start_time_exp1 = time.time()
for current_k in K_VALUES:
    print(f"\n  Evaluating k = {current_k} for all configurations...")
    for config_name, config_data in results_exp1.items():
        knn_instance = config_data['KnnInstance']
        start_k_time = time.time()
        overall_acc, _, _, _, _ = knn_instance.accuracy(sub_test_set, sub_test_label, current_k)
        end_k_time = time.time()
        time_taken = end_k_time - start_k_time
        config_data['accuracies'].append(overall_acc)
        config_data['times'].append(time_taken)
        print(f"    {config_name}: Accuracy: {overall_acc:.2f}% (Time taken: {time_taken:.4f} seconds)")

total_end_time_exp1 = time.time()
print(f"\n--- Experiment 1 completed in {total_end_time_exp1 - total_start_time_exp1:.2f} seconds. ---\n")

# Plot accuracy curves
plot_results_curves(
    K_VALUES, 
    {k: v['accuracies'] for k, v in results_exp1.items()}, 
    f"KNN Classification Accuracy vs K-value",
    "K Value", "Accuracy", 
    "Exp1_K_value_accuracy_curves.png",
    is_accuracy=True,
    N_TRAIN=N_TRAIN_EXP1,
    N_TEST=N_TEST_FIXED
)

# --- Find best K only for BruteForce_WithNorm ---
best_k_idx = np.argmax(results_exp1['BruteForce_WithNorm']['accuracies'])
best_k_value = K_VALUES[best_k_idx]
best_k_acc = results_exp1['BruteForce_WithNorm']['accuracies'][best_k_idx]
print(f"\n--- Best K for BruteForce_WithNorm is {best_k_value} with accuracy {best_k_acc:.2f}% ---")

# --- Plot time comparison for best K ---
print(f"\n--- Plotting time comparison for fixed K={best_k_value} ---")
time_at_best_k = {}
labels_for_bar_chart = []
data_for_bar_chart = []
idx_best_k = np.where(K_VALUES == best_k_value)[0][0]

for config_name, config_data in results_exp1.items():
    if config_data['times']:
        time_at_best_k[config_name] = config_data['times'][idx_best_k]
        labels_for_bar_chart.append(config_name)
        data_for_bar_chart.append(time_at_best_k[config_name])

plot_bar_chart(
    labels_for_bar_chart, 
    data_for_bar_chart, 
    f"Prediction Time Comparison at Optimal K={best_k_value}",
    "Time (seconds)",
    f"Exp1_Optimal_K_{best_k_value}_Time_Comparison.png",
    N_TRAIN=N_TRAIN_EXP1,
    N_TEST=N_TEST_FIXED
)

# =========================================================================================
# --- Experiment 2: Training size impact (Fixed K = best_k_value, BruteForce_WithNorm) ---
# =========================================================================================
print(f"\n--- Running Experiment 2: Training size impact (Fixed K={best_k_value}) ---")
print(f"  Using BruteForce_WithNorm, Fixed testing size: {N_TEST_FIXED}")

results_exp2_accuracies = {'BruteForce_WithNorm': []}
results_exp2_times = {'BruteForce_WithNorm': []}

total_start_time_exp2 = time.time()
for n_train_size in TRAIN_SIZES_EXP2:
    print(f"\n  Processing training size: {n_train_size}")
    current_train_set = full_train_set[:n_train_size, :]
    current_train_label = full_train_label[:n_train_size]

    print(f"    Initializing KnnClassifier (BruteForce_WithNorm) with training size {n_train_size}...")
    knn_instance_exp2 = KnnClassifier(current_train_set, current_train_label, use_kdtree=False, normalization=True)
    
    print(f"    Evaluating BruteForce_WithNorm with training size {n_train_size}...")
    start_time = time.time()
    overall_acc, _, _, _, _ = knn_instance_exp2.accuracy(sub_test_set, sub_test_label, best_k_value)
    end_time = time.time()
    time_taken = end_time - start_time
    
    results_exp2_accuracies['BruteForce_WithNorm'].append(overall_acc)
    results_exp2_times['BruteForce_WithNorm'].append(time_taken)
    print(f"      Accuracy: {overall_acc:.2f}%, Time: {time_taken:.4f}s")

total_end_time_exp2 = time.time()
print(f"\n--- Experiment 2 completed in {total_end_time_exp2 - total_start_time_exp2:.2f} seconds. ---\n")

# Plot accuracy vs training size
plot_results_curves(
    TRAIN_SIZES_EXP2, 
    results_exp2_accuracies,
    f"KNN Classification Accuracy vs Training Size (Fixed K={best_k_value}, BruteForce WithNorm)",
    "Training Size", "Accuracy", 
    f"Exp2_Training_Size_Accuracy_K{best_k_value}_BruteForceWithNorm.png",
    is_accuracy=True,
    N_TEST=N_TEST_FIXED
)

# Plot time vs training size
plot_results_curves(
    TRAIN_SIZES_EXP2, 
    results_exp2_times,
    f"KNN Prediction Time vs Training Size (Fixed K={best_k_value}, BruteForce WithNorm)",
    "Training Size", "Time", 
    f"Exp2_Training_Size_Time_K{best_k_value}_BruteForceWithNorm.png",
    is_accuracy=False,
    N_TEST=N_TEST_FIXED
)

# =========================================================================================
# --- Final: Detailed visualization for BruteForce_WithNorm at optimal K ---
# =========================================================================================
print("\n--- Final: Detailed visualization for BruteForce_WithNorm at its optimal K ---")

final_knn_instance = KnnClassifier(
    full_train_set[:N_TRAIN_EXP1, :], 
    full_train_label[:N_TRAIN_EXP1], 
    use_kdtree=False, 
    normalization=True
)

overall_acc, y_pred, y_true, correct_indices, wrong_indices = \
    final_knn_instance.accuracy(sub_test_set, sub_test_label, best_k_value)

evaluation_title_prefix = f"BruteForce_WithNorm, k={best_k_value}, Train:{N_TRAIN_EXP1}, Test:{N_TEST_FIXED}"
evaluation_save_path_prefix = os.path.join(output_image_dir, f"BruteForce_WithNorm_k{best_k_value}_Detailed")

evaluate_and_visualize(overall_acc, 
                       y_pred, 
                       y_true, 
                       correct_indices, 
                       wrong_indices, 
                       sub_test_set, 
                       class_names=CLASS_NAMES, 
                       num_display_images=8,
                       title_prefix=evaluation_title_prefix, 
                       save_path_prefix=evaluation_save_path_prefix)

print("\n--- All experiments and visualizations completed. ---")
