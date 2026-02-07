from sayancpe586fa25 import deepl
from sayancpe586fa25 import animation
import matplotlib.pyplot as plt
from datetime import datetime
import torch
# Parameters
d = 200
n = 40000
epochs = 5000
eta = 0.01

# Run training
W1, W2, W3, W4, loss_vector, weight_history = deepl.binary_classification(
    d=d,
    n=n,
    epochs=epochs,
    eta=eta
)

# Create timestamp YYYYMMDDhhmmss
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

# Filename with timestamp
filename = f"crossentropyloss_{timestamp}.pdf"

# Plot loss vs epochs
plt.figure()
plt.plot(loss_vector)
plt.xlabel("Epochs")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training Loss vs Epochs")
plt.savefig(filename)
plt.close()

animation.animate_large_heatmap(
    torch.stack(weight_history["W1"]),
    dt=0.04,
    file_name="W1_evolution",
    title_str="W1 Weight Evolution"
)

animation.animate_large_heatmap(
    torch.stack(weight_history["W2"]),
    dt=0.04,
    file_name="W2_evolution",
    title_str="W2 Weight Evolution"
)

animation.animate_large_heatmap(
    torch.stack(weight_history["W3"]),
    dt=0.04,
    file_name="W3_evolution",
    title_str="W3 Weight Evolution"
)
animation.animate_large_heatmap(
    torch.stack(weight_history["W4"]),
    dt=0.04,
    file_name="W4_evolution",
    title_str="W4 Weight Evolution"
)
print("Training and Animation Completed")
