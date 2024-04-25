# Libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
seq_length = 200

# Load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load('./trainedModel.pt').to(device)

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

# Animation
x1 = new_traj[300:, 0]
x2 = new_traj[300:, 1]
y = 5 * predictions[300:]

fig, ax = plt.subplots()
ax.set_xlim(0, len(x1)) 
ax.set_ylim(min(np.min(x1), np.min(x2), np.min(y)), max(np.max(x1), np.max(x2), np.max(y)))

line1, = ax.plot([], [], color = "blue", label = r'$x_{1}(t)$')
line2, = ax.plot([], [], color = "black", label = r'$x_{2}(t)$')
line3, = ax.plot([], [], color = "red", label = "Predictions")

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    return line1, line2, line3

def update(frame):
    line1.set_data(np.arange(frame), x1[:frame])
    line2.set_data(np.arange(frame), x2[:frame])
    line3.set_data(np.arange(frame), y[:frame])
    return line1, line2, line3

ani = FuncAnimation(fig, update, frames = len(x1), init_func = init, blit = True, interval = 50)

plt.legend()
plt.xlabel(r'$t$')
plt.title("Predictions of wing changes using trained model with CLVs")
plt.grid(True)
plt.show()