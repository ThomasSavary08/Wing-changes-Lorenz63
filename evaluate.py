# Libraries
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load('./trainedModel.pt').to(device)

# Load validation data
x_val, y_val = np.load('./Data/x_test.npy'), np.load('./Data/y_test.npy')
x_val = torch.tensor(x_val, dtype = torch.float32, device = device)

# Use the model to make prediction
logits = model(x_val).cpu()
probs = torch.nn.functional.softmax(logits, dim = -1)
y_hat = torch.argmax(probs, dim = -1).numpy()

# Compute accuracy
acc = np.sum(y_hat == y_val) / len(y_val) * 100
print("Accuracy: {:.1f}".format(acc))

# Plot confusion matrix
conf_matrix = confusion_matrix(y_val, y_hat)
plt.figure()
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = np.unique(y_val), yticklabels = np.unique(y_val))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Load and plot trajectory
traj = np.load('./CLVs/testTraj_1.npy')
plt.figure()
plt.plot(traj[199:199 + 1000,0], color = "blue", label = r'$x_{1}$')
plt.plot(traj[199:199 + 1000,1], color = "black", label = r'$x_{2}$')
plt.plot(5*y_hat[:801], color = "red", label = "Predictions")
plt.xlabel(r'$t$')
plt.legend()
plt.show()
