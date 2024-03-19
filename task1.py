def accuracy(TP, TN, FP, FN):
    return (TP + TN) / (TP + TN + FP + FN)

def precision(TP, FP):
    return TP / (TP + FP) if (TP + FP) > 0 else 0

def recall(TP, FN):
    return TP / (TP + FN) if (TP + FN) > 0 else 0

def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# 示例数据
TP = 90
TN = 80
FP = 10
FN = 10

# 计算指标
accuracy_value = accuracy(TP, TN, FP, FN)
precision_value = precision(TP, FP)
recall_value = recall(TP, FN)
f1_value = f1_score(precision_value, recall_value)

print(f"Accuracy: {accuracy_value:.2f}")
print(f"Precision: {precision_value:.2f}")
print(f"Recall: {recall_value:.2f}")
print(f"F1-score: {f1_value:.2f}")

