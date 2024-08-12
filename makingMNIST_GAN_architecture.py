#imports
import struct
from array import array
from os.path import join
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#set device to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device is ", device)
torch.manual_seed(123)
np.random.seed(123)

#model parameters
batch_size = 300 #60000 should be divisible by this number
num_epochs = 450


class MnistDataloader1(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())

        images = []
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols], dtype=np.float32)
            img = img.reshape(28, 28)
            images.append(img)

        return images, labels
    
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        
        # Convert to PyTorch tensors
        x_train_tensor = torch.tensor(x_train) / 255.0
        y_train_tensor = torch.tensor(y_train, dtype=torch.float) / 10
        x_test_tensor = torch.tensor(x_test) / 255.0
        y_test_tensor = torch.tensor(y_test, dtype=torch.float) / 10
        
        return (x_train_tensor, y_train_tensor), (x_test_tensor, y_test_tensor)
    

#TODO make the first dataloader object do9 this as well by changing load_data to __getitem__ and returning [index] of each one
class MnistDataLoaderTrain(object):
    def __init__(self, x_train_tensor, y_train_tensor):        
        self.x_train_tensor = x_train_tensor
        self.y_train_tensor = y_train_tensor

    def __len__(self):
        return x_train_tensor.shape[0]

    def __getitem__(self, index):
        #print(self.y_train_tensor[index])
        return (self.x_train_tensor[index], self.y_train_tensor[index])

    
# Set file paths based on added MNIST Datasets
input_path = r'C:\Users\Matt\Documents\Pytorch_ML\Generative\GAN\Data\archive'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

# Load MNIST dataset
mnist_dataloader = MnistDataloader1(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train_tensor, y_train_tensor), (x_test_tensor, y_test_tensor) = mnist_dataloader.load_data()     #for purposes of displaying images in the beginning

dataLoaderTrain = MnistDataLoaderTrain(x_train_tensor, y_train_tensor)

#TODO: consolidate train and test dataloaders into one? for both this and random numbers?
mnistDataSetTrain = DataLoader(dataLoaderTrain, batch_size = batch_size, shuffle = False)




# Helper function to show a list of images with their relating titles
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(14, 14))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image.cpu().detach().numpy(), cmap=plt.cm.gray)
        if title_text != '':
            plt.title(title_text, fontsize=15)
        index += 1
    plt.show(block = False)   # Show the image
    plt.pause(3.5)
    plt.close("all")  # Close the figure after showing

# Show some random training and test images
images_2_show = []
titles_2_show = []

for i in range(0, 10):
    r = random.randint(0, 59999)
    images_2_show.append(x_train_tensor[r].numpy())
    titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train_tensor[r].item()))

for i in range(0, 5):
    r = random.randint(0, 9999)
    images_2_show.append(x_test_tensor[r].numpy())
    titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test_tensor[r].item()))

#show_images(images_2_show, titles_2_show)




#_______________________________________________________ model time ________________________________________________________________________________________

img_size = x_train_tensor[0].size(0)
print("Image size is", img_size)
print("Entire tensor size is ",x_train_tensor.size())

class Generator(nn.Module):
    def __init__(self, input_length, batch_size):
        super(Generator, self).__init__()
        
        self.batch_size = batch_size
        self.input_length = input_length
        
        # Encoding layers (downsampling)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)

        # Decoding layers (upsampling)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, dilation = 2, output_padding = 1)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(64)

    def forward(self, x):
        # Reshape the input to add the channel dimension (batch_size, 1, input_length, input_length)
        x = x.view(-1, 1, self.input_length, self.input_length)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        #print(f"Level 4 x shape {x.shape}")
        # Decoding path with transposed convolutional layers, batch normalization, and ReLU activation
        x = F.relu(self.bn4(self.deconv1(x)))
        
        #print(f"Level 5 x shape {x.shape}")
        x = F.relu(self.bn5(self.deconv2(x)))

        
        #print(f"Level 6 x shape {x.shape}")

        # Final output layer with tanh activation
        x = torch.tanh(self.deconv3(x))
        
        print(f"level 7 x shape {x.shape}")
        
        # Remove the channel dimension to return to the original input shape

        x = x.view(batch_size, self.input_length, self.input_length)
        
        return x
    
class Discriminator(nn.Module):
    def __init__(self, input_length, batch_size):
        super(Discriminator, self).__init__()
        
        self.batch_size = batch_size
        self.input_length = input_length
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(batch_size, 64, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64**2, 256)
        self.fc2 = nn.Linear(256, batch_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Reshape the input to match convolutional input size
        x = x.view(-1, self.input_length, self.input_length)

        # Convolutional layers with ReLU activation
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        
        # Flatten the output from the convolutional layers
        x = torch.flatten(x)
        
        # Fully connected layers with ReLU and Dropout
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        
        # Output layer with sigmoid activation
        x = torch.sigmoid(self.fc2(x))
        
        return x

        
def Train(epochs, verbose, showImages):
    generator = Generator(img_size, batch_size).to(device)
    discriminator = Discriminator(img_size, batch_size).to(device)
    generator_opt = torch.optim.Adam(generator.parameters(), lr=0.001)
    discriminator_opt = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    loss = nn.BCELoss()

    for i in range(epochs):
        
        correct_predictions = 0
        total_predictions = 0
        
        generator_opt.zero_grad()
        noise = torch.from_numpy(np.random.rand(img_size, img_size)).to(device)

        for x_train_tensor, y_train_tensor in mnistDataSetTrain:
            noise = torch.rand((batch_size, img_size, img_size)).to(device)
            data = generator(noise).to(device)

            true_data = x_train_tensor.to(device)
            true_labels = y_train_tensor.to(device)
            
            gen_discrim_out = discriminator(data).to(device)

            
            # Calculate the generator loss
            generator_loss = loss(gen_discrim_out, true_labels)
            generator_loss.backward()
            generator_opt.step()

            discriminator_opt.zero_grad()
            true_discriminator_out = discriminator(true_data).to(device)
            true_discriminator_loss = loss(true_discriminator_out, true_labels)

            generator_discriminator_out = discriminator(data.detach()).to(device)
            generator_discriminator_loss = loss(generator_discriminator_out, torch.zeros(batch_size).to(device))
            discriminator_loss = (generator_discriminator_loss + true_discriminator_loss) / 2
            discriminator_loss.backward()
            discriminator_opt.step()

            # Calculate accuracy
            predicted_labels = torch.round(gen_discrim_out)  # Rounding to get binary predictions
            correct_predictions += torch.sum(predicted_labels == true_labels).item()
            total_predictions += batch_size

        if verbose == True:
            accuracy = correct_predictions / total_predictions
            print(f'Epoch [{i+1}/{epochs}], Generator Loss: {generator_loss.item():.4f}, Discriminator Loss: {discriminator_loss.item():.4f}')
        
        if showImages == True:
            title = f"Epoch [{i+1}/{epochs}] generated image"
                #just doing 12 random images for now
            show_images(data[0:10], title)

    if saveModel == True:
        return noise, generator, discriminator, saveModel


saveModel = True

if(saveModel == True):
    noise, generator, discriminator,_ = Train(num_epochs, verbose=True, showImages=saveModel)
else:
    Train(num_epochs, verbose=True, showImages=saveModel)
