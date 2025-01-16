import torch
from ChessCNN import ChessCNN
# Initialize the model


model = ChessCNN()
model.load_state_dict(torch.load('chess_model.pth'))
model.eval()

# Access the state_dict
state_dict = model.state_dict()

print(state_dict)
print("State dict keys:")
for key in state_dict.keys():
    print(key) 


# Loop through the state_dict to print parameter names, shapes, and values
print("Model Parameters:")
for name, param in state_dict.items():
    print(f"{name}: {param.shape}")
    print(f"Values (first 5): {param.view(-1)[:5]}")  # Print the first 5 values


# Check available keys and use the correct key for accessing weights
if 'conv1.weight' in state_dict:
    conv1_weights = state_dict['conv1.weight']
    print(f"Shape of conv1 weights: {conv1_weights.shape}")
    print(f"First 5 weights: {conv1_weights.view(-1)[:5]}")
else:
    print("Key 'conv1.weight' not found in state_dict")

# Access biases of the first fully connected layer
if 'fc1.bias' in state_dict:
    fc1_bias = state_dict['fc1.bias']
    print(f"Shape of fc1 bias: {fc1_bias.shape}")
    print(f"First 5 biases: {fc1_bias[:5]}")
else:
    print("Key 'fc1.bias' not found in state_dict")

# Write the state_dict information to a text file
with open("stateDict.txt", "w") as f:
    f.write("State dict keys:\n")
    for key in state_dict.keys():
        f.write(f"{key}\n")
    
    f.write("\nModel Parameters:\n")
    for name, param in state_dict.items():
        f.write(f"{name}: {param.shape}\n")
        f.write(f"Values : {param.view}\n")
    
    if 'conv1.weight' in state_dict:
        f.write(f"\nShape of conv1 weights: {conv1_weights.shape}\n")
        f.write(f"weights: {conv1_weights.view}\n")
    
    if 'fc1.bias' in state_dict:
        f.write(f"\nShape of fc1 bias: {fc1_bias.shape}\n")
        f.write(f"biases: {fc1_bias}\n")