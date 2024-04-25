# Libraries
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Parameters
seq_length = 200

# Load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load('./trainedModel.pt').to(device)

# Load validation data
x_val, y_val = np.load('./Data/x_test.npy'), np.load('./Data/y_test.npy')
x_val = torch.tensor(x_val, dtype = torch.float32, device = device)

# Use the model to make prediction
logits = model(x_val)
probs = torch.nn.functional.softmax(logits, dim = -1).cpu()
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

# Load trajectory
traj, CLV_ = np.load('./CLVs/testTraj_1.npy'), np.load('./CLVs/testCLV_1.npy')
n, dim, _ = CLV_.shape

# Compute angles
angles = np.zeros((n, dim))
for j in range(n):
    angles[j,0] = np.arccos(np.dot(CLV_[j,:,0], CLV_[j,:,1]))
    angles[j,1] = np.arccos(np.dot(CLV_[j,:,0], CLV_[j,:,2]))
    angles[j,2] = np.arccos(np.dot(CLV_[j,:,1], CLV_[j,:,2]))

# Compute predictions
predictions = []
new_traj = []
for j in range(seq_length, n):
    x = angles[(j-seq_length):j, :]
    x = torch.tensor(x, dtype = torch.float32, device = device).unsqueeze(0)
    probs = torch.nn.functional.softmax(model(x), dim = -1).cpu()
    predictions.append(torch.argmax(probs, dim = -1).numpy()[0])
    new_traj.append(traj[j,:])
predictions = np.asarray(predictions)
new_traj = np.asarray(new_traj)

# Plot predictions
plt.figure()
plt.plot(new_traj[:,0], color = "blue", label = r'$x_{1}(t)$')
plt.plot(new_traj[:,1], color = "black", label = r'$x_{2}(t)$')
plt.plot(5*predictions, color = "red", label = "Predictions")
plt.xlabel(r'$t$')
plt.title("Prediction of the trained model")
plt.legend()
plt.show()