# Libraries
import numpy as np

# Parameters
seq_length = 200

# Find wing changes in the trajectory
def which_wing(x, y):
    if ((x > 0) and (y > 0)):
        return 1
    elif ((x < 0) and (y < 0)):
        return -1
    else:
        return 0

# Preprocess training data
x_train, labels_train = [], []
for i in range(10):

    # Load data 
    CLV_path = './../CLVs/trainCLV_' + str(i+1) + '.npy'
    traj_path = './../CLVs/trainTraj_' + str(i+1) + '.npy'
    raw_CLV, raw_traj = np.load(CLV_path), np.load(traj_path)

    # Compute angles between CLV
    n, dim, _ = raw_CLV.shape
    angles = np.zeros((n, dim))
    print("Computing angles between CLVs for initial condition n°{:d} (train)".format(i+1))
    for j in range(n):
        angles[j,0] = np.arccos(np.dot(raw_CLV[j,:,0], raw_CLV[j,:,1]))
        angles[j,1] = np.arccos(np.dot(raw_CLV[j,:,0], raw_CLV[j,:,2]))
        angles[j,2] = np.arccos(np.dot(raw_CLV[j,:,1], raw_CLV[j,:,2]))
    print("OK.")
    print("")

    # Find wing changes in trajectory
    print("Finding wing changes for initial condition n°{:d} (train)".format(i+1))
    ind_start = seq_length - 1
    while which_wing(raw_traj[ind_start, 0], raw_traj[ind_start, 1]) == 0:
        ind_start += 1
    current_wing = which_wing(raw_traj[ind_start, 0], raw_traj[ind_start, 1])
    if current_wing != which_wing(raw_traj[ind_start - 1, 0], raw_traj[ind_start - 1, 1]):
        changes = [1]
    else:
        changes = [0]
    for j in range(ind_start + 1, n):
        computed_wing = which_wing(raw_traj[j,0], raw_traj[j,1])
        if ((computed_wing == current_wing) or (computed_wing == 0)):
            changes.append(0)
        else:
            current_wing = computed_wing
            changes.append(1)
    print("OK.")
    print("")

    # Find last transition and compute the number of time steps before the next transition 
    print("Find the number of time steps before for initial condition n°{:d} (train)".format(i+1))
    ind_end = changes[::-1].index(1)
    ind_end = len(changes) - ind_end - 1
    before_changes = [0]
    nb = 0
    for j in range(ind_end - 1, -1, -1):
        if changes[j] == 1:
            nb = 0
        else:
            nb +=  1
        before_changes.append(nb)
    before_changes = before_changes[::-1]
    changes = changes[:(ind_end + 1)]
    print("OK.")
    print("")

    # Convert the number of time steps to labels (classification)
    print("Compute labels for classification for initial condition n°{:d} (train)".format(i+1))
    labels = []
    for j in range(len(before_changes)):
        if before_changes[j] < 50:
            labels.append(0)
        elif ((before_changes[j] >= 50) and (before_changes[j] < 150)):
            labels.append(1)
        else:
            labels.append(2)
    print('OK.')
    print("")

    # Add data
    print("Add data from initial condition n°{:d} (train)".format(i+1))
    for j in range(ind_start, ind_start + len(labels)):
        x_train.append(angles[(j - seq_length + 1):(j + 1),:])
        labels_train.append(labels[j - ind_start])
    print("OK.")
    print("")
    print("#########################################################################")
    
# Preprocess test data
x_test, labels_test = [], []
for i in range(3):

    # Load data 
    CLV_path = './../CLVs/testCLV_' + str(i+1) + '.npy'
    traj_path = './../CLVs/testTraj_' + str(i+1) + '.npy'
    raw_CLV, raw_traj = np.load(CLV_path), np.load(traj_path)

    # Compute angles between CLV
    n, dim, _ = raw_CLV.shape
    angles = np.zeros((n, dim))
    print("Computing angles between CLVs for initial condition n°{:d} (test)".format(i+11))
    for j in range(n):
        angles[j,0] = np.arccos(np.dot(raw_CLV[j,:,0], raw_CLV[j,:,1]))
        angles[j,1] = np.arccos(np.dot(raw_CLV[j,:,0], raw_CLV[j,:,2]))
        angles[j,2] = np.arccos(np.dot(raw_CLV[j,:,1], raw_CLV[j,:,2]))
    print("OK.")
    print("")

    # Find wing changes in trajectory
    print("Finding wing changes for initial condition n°{:d} (test)".format(i+11))
    ind_start = seq_length - 1
    while which_wing(raw_traj[ind_start, 0], raw_traj[ind_start, 1]) == 0:
        ind_start += 1
    current_wing = which_wing(raw_traj[ind_start, 0], raw_traj[ind_start, 1])
    if current_wing != which_wing(raw_traj[ind_start - 1, 0], raw_traj[ind_start - 1, 1]):
        changes = [1]
    else:
        changes = [0]
    for j in range(ind_start + 1, n):
        computed_wing = which_wing(raw_traj[j,0], raw_traj[j,1])
        if ((computed_wing == current_wing) or (computed_wing == 0)):
            changes.append(0)
        else:
            current_wing = computed_wing
            changes.append(1)
    print("OK.")
    print("")

    # Find last transition and compute the number of time steps before the next transition 
    print("Find the number of time steps before for initial condition n°{:d} (test)".format(i+11))
    ind_end = changes[::-1].index(1)
    ind_end = len(changes) - ind_end - 1
    before_changes = [0]
    nb = 0
    for j in range(ind_end - 1, -1, -1):
        if changes[j] == 1:
            nb = 0
        else:
            nb +=  1
        before_changes.append(nb)
    before_changes = before_changes[::-1]
    changes = changes[:(ind_end + 1)]
    print("OK.")
    print("")

    # Convert the number of time steps to labels (classification)
    print("Compute labels for classification for initial condition n°{:d} (test)".format(i+11))
    labels = []
    for j in range(len(before_changes)):
        if before_changes[j] < 50:
            labels.append(0)
        elif ((before_changes[j] >= 50) and (before_changes[j] < 150)):
            labels.append(1)
        else:
            labels.append(2)
    print('OK.')
    print("")

    # Add data
    print("Add data from initial condition n°{:d} (test)".format(i+11))
    for j in range(ind_start, ind_start + len(labels)):
        x_test.append(angles[(j - seq_length + 1):(j + 1),:])
        labels_test.append(labels[j - ind_start])
    print("OK.")
    print("")
    print("#########################################################################")

# Save datasets
print("Save datasets")
x_train, y_train = np.asarray(x_train), np.asarray(labels_train)
x_test, y_test = np.asarray(x_test), np.asarray(labels_test)
np.save('./x_train.npy', x_train)
np.save('./y_train.npy', y_train)
np.save('./x_test.npy', x_test)
np.save('./y_test.npy', y_test)
print("OK.")
print("")
print("#########################################################################")