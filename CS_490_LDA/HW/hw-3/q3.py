# file q3.py
# you can download statistical_header.py from
#     https://www.cs.purdue.edu/homes/ribeirob/courses/Fall2020/hw/hw3/statistical_header.py
from matplotlib import use

use('Agg')

import matplotlib.pyplot as plt

from statistical_header import *
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data_utils

class Net(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        """
        In the constructor we instantiate three nn.Linear modules and assign them as
        member variables.

        first parameter: input dimension
        second parameter: dimension of the output
        """
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # First fully connected layer,  out = out^T W_1 + b_1
        out = self.fc1(x)
        # Activation function (sigmoid, relu, tanh, ...)
        out = torch.sigmoid(out)
        # Second fully connected layer out = out^T W_2 + b_2
        # out = self.fc2(out)
        # # Activation function (sigmoid, relu, tanh, ...)
        # out = torch.relu(out)
        # Third fully connected layer out = out^T W_3 + b_3
        out = self.fc3(out)
        # Final activation function (log of the sigmoid)
        out = F.log_softmax(out, dim=1)
        return out

train_file = "Bank_Data_Train.csv"
validation_file = "Bank_Data_Validation.csv"

train_sizes = [50, 100,300,600,1000,1500,2000, 2500, 3000]
AUC_scores_validation = []
AUC_scores_train = []

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

for train_size in train_sizes:

# read the data in
    print(f'Training data size {train_size}')

    data_train, target_train = data_processing(train_file)
    data_validation, target_validation = data_processing(validation_file)


    data_train = data_train[0:train_size]
    target_train = target_train[0:train_size]

    # replace categorical strings with one-hot encoding and add a small amount of Gaussian noise so it follows Gaussian model assumption

    data_train, data_validation = add_categorical(train=data_train,validation=data_validation,feature_str='FICO Range')

    data_train, data_validation = add_categorical(train=data_train,validation=data_validation,feature_str='Loan Purpose')

    # Create the neural network classifier object in GPU or CPU memory (depending on device)
    model = Net(input_size=data_train.shape[1],hidden_size=100000).to(device=device)

    # Loss and optimizer
    # nn.NLLLoss() is the negative log likelihood. It takes the log_softmax as input.
    #  ... if using nn.CrossEntropyLoss(), it will compute softmax internally.
    criterion = nn.NLLLoss()
    # Optimizer is the standard stochastic gradient descent
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Train the model going over the dataset 100 times
    # Convert data from pandas DataFrame to pytorch Tensor data
    input_batch = torch.from_numpy(np.array(data_train).astype('float32')).to(device=device)
    labels_batch = torch.from_numpy(np.array(target_train).astype('int64')).to(device=device)

    train_tensor = data_utils.TensorDataset(input_batch, labels_batch)
    train_loader = data_utils.DataLoader(dataset = train_tensor, batch_size = 50, shuffle = True)

    # Loop training over epochs.
    #     Note that 1000000//train_size keeps the number of total gradient descent steps about the same accross all dataset sizes
    for epoch in range(1000000//train_size):
        for inputs, labels in train_loader:
            # Need to zero the gradient for this batch, since Pytorch gradients accumulate
            optimizer.zero_grad()

            # Get model outputs
            outputs = model(inputs)
            # Compute loss
            loss = criterion(outputs, labels)

            # Backpropagation and optimize
            loss.backward()
            optimizer.step()

    # Set to evaluation model
    model.eval()
    # Make sure we will not compute gradients when dealing with the model
    with torch.no_grad():
        # Predicted probabilities of label +1
        #  Predicted probabilities for training
        inputs = torch.from_numpy(np.array(data_train).astype('float32')).to(device=device)
        # Tells pytorch not to compute gradients
        # Need to exponentiate since we used the log-softmax as output of our model
        #   Detach makes sure pytorch knows the output of the model does not require gradients
        y_pred_train = torch.exp(model(inputs))[:,1]
        # Make sure results are on CPU and described as numpy arrays
        y_pred_train = y_pred_train.cpu().numpy()

        #  Predicted probabilities for validation
        inputs = torch.from_numpy(np.array(data_validation).astype('float32')).to(device=device)
        # Get the probability of class 1 (second column of output,
        #      probability of class zero needs first column [:,0] in place of second column [:,1]
        y_pred_val = torch.exp(model(inputs))[:,1]
        # Make sure results are on CPU and described as numpy arrays
        y_pred_val = y_pred_val.cpu().numpy()

        ### ROC Curve ###
        # Find false and true positives of each target for various tresholds
        fpr, tpr, thresholds =roc_curve(target_train, y_pred_train)
        # Area under the curve
        roc_auc = auc(fpr, tpr)
        AUC_scores_train.append(roc_auc)

        # Find false and true positives of each target for various tresholds
        fpr, tpr, thresholds =roc_curve(target_validation, y_pred_val)
        # Area under the curve
        roc_auc = auc(fpr, tpr)
        AUC_scores_validation.append(roc_auc)

    # Make sure we free the model's memory (helps CPU and GPU garbage collection)
    del model

print(f'AUC_scores_validation={AUC_scores_validation}')
print(f'AUC_scores_train={AUC_scores_train}')
print("Done.")

print("===[Q1:END]===")

plt.rcParams["figure.figsize"] = (8,6)
plt.rcParams.update({'font.size': 20})
fig, ax = pl.subplots()
plt.plot(train_sizes, AUC_scores_train, label='AUC over training',color="blue")
plt.plot(train_sizes, AUC_scores_validation, label='AUC over validation',color="red")
ax.set_ylim([0.5,1.0])
plt.legend(loc='best')
plt.xlabel('Training Data Size')
plt.ylabel('AUC Score')
plt.title('Deep Learning')
plt.savefig("LearningCurves.png")