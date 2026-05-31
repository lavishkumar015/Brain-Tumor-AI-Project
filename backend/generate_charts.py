import matplotlib.pyplot as plt
import os

# Create charts folder
os.makedirs(
    "static/charts",
    exist_ok=True
)

# =========================================
# ACCURACY GRAPH
# =========================================

accuracy = [70, 78, 85, 90, 94, 96]

epochs = [1, 2, 3, 4, 5, 6]

plt.figure(figsize=(6,4))

plt.plot(
    epochs,
    accuracy,
    marker='o'
)

plt.title(
    "Model Accuracy"
)

plt.xlabel(
    "Epochs"
)

plt.ylabel(
    "Accuracy %"
)

plt.grid(True)

plt.savefig(
    "static/charts/accuracy.png"
)

plt.close()

# =========================================
# LOSS GRAPH
# =========================================

loss = [1.8, 1.2, 0.8, 0.5, 0.3, 0.1]

plt.figure(figsize=(6,4))

plt.plot(
    epochs,
    loss,
    marker='o'
)

plt.title(
    "Model Loss"
)

plt.xlabel(
    "Epochs"
)

plt.ylabel(
    "Loss"
)

plt.grid(True)

plt.savefig(
    "static/charts/loss.png"
)

plt.close()

print(
    "Charts generated successfully!"
)