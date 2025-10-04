import matplotlib.pyplot as plt

# ==== Your Data ====
epochs = list(range(10))
train_loss = [0.958497, 0.541017, 0.351906, 0.291022, 0.206437,
              0.177716, 0.174075, 0.156122, 0.14361, 0.13655]
eval_exact = [78.33167, 79.69425, 80.45862, 80.62479, 80.15952,
              80.82419, 81.58857, 80.79096, 82.25324, 82.81821]

# ==== Plot Training Loss ====
plt.figure(figsize=(7,5))
plt.plot(epochs, train_loss, marker='o', linestyle='-', color='blue', label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Learning Curve - Training Loss (Chinese-PERT-Large QA)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

# ==== Plot Validation EM ====
plt.figure(figsize=(7,5))
plt.plot(epochs, eval_exact, marker='o', linestyle='-', color='green', label="Validation EM")
plt.xlabel("Epoch")
plt.ylabel("Exact Match (%)")
plt.title("Learning Curve - Validation Exact Match (Chinese-PERT-Large QA)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()
