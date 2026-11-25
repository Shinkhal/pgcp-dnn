
# We will use pytorch to build a simple binary classification model. The model will learn to classify data points into two classes based on their features.
# pass = 1, fail = 0 based on the features of the data points. The model will be trained on a dataset of students' scores in two subjects and their corresponding pass/fail labels.



import torch
import torch.nn as nn



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

print("PyTorch version:", torch.__version__)

hours_studied = torch.tensor([[1.0],[1.5], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0]], dtype=torch.float32, device=device)

passed_exam = torch.tensor([[0], [0], [0], [1], [1], [1], [1], [1],[0], [1]], dtype=torch.float32, device=device)

# Step 2 : single neuron : z = w.x + b

