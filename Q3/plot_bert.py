import matplotlib.pyplot as plt

# ==== BERT Data ====
epochs = list(range(5))
train_loss = [1.515916, 0.610577, 0.301693, 0.142632, 0.062609]
eval_exact = [70.72117, 73.41309, 75.10801, 74.77567, 75.90562]

# ==== Plot Training Loss ====
plt.figure(figsize=(7,5))
plt.plot(epochs, train_loss, marker='o', linestyle='-', color='blue', label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Learning Curve - Training Loss (BERT-base-Chinese QA)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

# ==== Plot Validation EM ====
plt.figure(figsize=(7,5))
plt.plot(epochs, eval_exact, marker='o', linestyle='-', color='green', label="Validation EM")
plt.xlabel("Epoch")
plt.ylabel("Exact Match (%)")
plt.title("Learning Curve - Validation Exact Match (BERT-base-Chinese QA)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()
