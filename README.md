# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Include the Problem Statement and Dataset.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Write your own steps

### STEP 2: 



### STEP 3: 



### STEP 4: 



### STEP 5: 



### STEP 6: 





## PROGRAM

### Name:

### Register Number:

```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # write your code here
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128*3*3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # write your code here
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the Model
## Step 3: Train the Model
def train_model(model, train_loader, num_epochs=3):
      # write your code here
    for epoch in range(num_epochs):
        model.train()
        running_loss=0.0
        for images,labels in train_loader:
            optimizer.zero_grad()
            outputs=model(images)
            loss=criterion(outputs,labels)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        print('Name: Malar Mariam S')
        print('Register Number: 212223230118')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

### OUTPUT

## Training Loss per Epoch
<img width="321" height="199" alt="image" src="https://github.com/user-attachments/assets/0346500b-98fc-45a8-865a-b5a9bfeff94f" />


## Confusion Matrix
<img width="661" height="532" alt="image" src="https://github.com/user-attachments/assets/44b8e8a0-902a-4dd8-9243-9dc55ab5dfca" />



## Classification Report
<img width="396" height="268" alt="image" src="https://github.com/user-attachments/assets/3568f300-f473-43f4-8790-adce689ec882" />


## RESULT
<img width="549" height="596" alt="image" src="https://github.com/user-attachments/assets/1deda2f8-916c-49a6-b8da-495ae79e3954" />

