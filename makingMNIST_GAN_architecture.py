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
batch_size = 32
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
    
class MnistDataLoaderTrain(object):
    def __init__(self, x_train_tensor, y_train_tensor):        
        self.x_train_tensor = x_train_tensor
        self.y_train_tensor = y_train_tensor

    def __len__(self):
        return x_train_tensor.shape[0]

    def __getitem__(self, index):
        return (self.x_train_tensor[index], self.y_train_tensor[index])

class MnistDataLoaderTest(object):
    def __init__(self, x_test_tensor, y_test_tensor):        
        self.x_train_tensor = x_test_tensor
        self.y_train_tensor = y_test_tensor

    def __len__(self):
        return x_test_tensor.shape[0]

    def __getitem__(self, index):
        return (self.x_test_tensor[index], self.y_test_tensor[index])
    
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
#dataLoaderTest = DataLoader(x_test_tensor, y_train_tensor)


mnistDataSetTrain = DataLoader(dataLoaderTrain, batch_size = batch_size, shuffle = True)




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
    def __init__(self, input_length,batch_size):
        super(Generator,self).__init__()
        
        self.batch_size = batch_size
        
        self.input_length = input_length
        self.Fc1 = nn.Linear(input_length**2*batch_size, 510)
        self.Fc2 = nn.Linear(510, input_length**2*batch_size)
        #reshape output to be 28 x 28
    def forward(self,x):
        input_length = self.input_length
        batch_size = self.batch_size

        #x = torch.flatten(x).unsqueeze(-1)
        x = torch.flatten(x)
        x = F.relu(self.Fc1(x))
        x = F.sigmoid(self.Fc2(x))
        x =  torch.reshape(x, (batch_size, input_length, input_length))

        outputs = x
        return outputs
    
    
class Discriminator(nn.Module):
        def __init__(self, input_length, batch_size):
            super(Discriminator,self).__init__()
            
            self.batch_size = batch_size
            self.input_length = input_length
            self.Fc1 = nn.Linear(input_length**2 *batch_size, 256)
            self.Fc2 = nn.Linear(256, batch_size)

        def forward(self,x):
            input_length = self.input_length
            batch_size = self.batch_size

            x = torch.flatten(x)
            x = F.relu(self.Fc1(x))
            x = F.sigmoid(self.Fc2(x))

            outputs = x
            return outputs
        
        
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
            show_images(data, title)

    if saveModel == True:
        return noise, generator, discriminator, saveModel


saveModel = True

if(saveModel == True):
    noise, generator, discriminator,_ = Train(num_epochs, verbose=True, showImages=saveModel)
else:
    Train(num_epochs, verbose=True, showImages=saveModel)
