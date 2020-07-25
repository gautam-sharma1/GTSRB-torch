import gtsrb_dataset as dataset
import torchvision
import torchvision.transforms as transforms
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from neuralnet import Net
import matplotlib.pyplot as plt
import numpy as np


# Create Transforms
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.3403, 0.3121, 0.3214),
                         (0.2724, 0.2608, 0.2669))
])

#Function to display an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
# Create Datasets
trainset = dataset.GTSRB(
    root_dir='german-traffic-sign', train=True,  transform=transform)
testset = dataset.GTSRB(
    root_dir='german-traffic-sign', train=False,  transform=transform)

# Load Datasets
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2)

#Instantiate the neural network
net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
loss_plot = []
for epoch in range(10):  # loop over the dataset multiple times

    print("running.....")
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs) 
        _, predicted = torch.max(outputs.data, 1) #outputs.shape = [128,43]
        loss = criterion(outputs, labels)
        loss_plot.append(loss)
        # if(i%100==0):
        #     print("loss ", loss.item())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f accuracy: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200,100 * correct / total ))
            
            running_loss = 0.0


print('Finished Training')
plt.plot(range(len(loss_plot)),loss_plot, 'r+')
plt.title("Loss")
plt.show()
PATH = '/Users/gautamsharma/Desktop/RL/pytorch/german-traffic-sign/best_model.pt'
torch.save(net.state_dict(), PATH)
print('model saved')

print('Testing......')

# dataiter = iter(testloader)
# images, labels = dataiter.next()
# model = torch.load(PATH)
# print images
# imshow(torchvision.utils.make_grid(images[:16]))
# print('GroundTruth: ', ' '.join('%5s' % labels[j] for j in range(16)))
"""
Testing
"""

model = Net()
model.load_state_dict(torch.load(PATH))
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

