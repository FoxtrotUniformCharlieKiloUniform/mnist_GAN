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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device is ", device)
torch.manual_seed(123)



class MnistDataloader(object):
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
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        x_test_tensor = torch.tensor(x_test) / 255.0
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        return (x_train_tensor, y_train_tensor), (x_test_tensor, y_test_tensor)

# Set file paths based on added MNIST Datasets
input_path = r'C:\Users\Matt\Documents\Pytorch_ML\Generative\GAN\Data\archive'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

# Helper function to show a list of images with their relating titles
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(30, 20))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if title_text != '':
            plt.title(title_text, fontsize=15)
        index += 1
    plt.show()  # Ensure that the plot is shown
    

# Load MNIST dataset
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train_tensor, y_train_tensor), (x_test_tensor, y_test_tensor) = mnist_dataloader.load_data()

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

show_images(images_2_show, titles_2_show)


#_______________________________________________________ model time ________________________________________________________________________________________
img_size = x_train_tensor[0].size(0)
print("Image size is", img_size)
print("Entire tensor size is ",x_train_tensor.size())

class Generator(nn.Module):
    def __init__(self, input_length):
        super(Generator,self).__init__()
        self.Fc1 = nn.Linear(128, input_length^2)
        self.Fc2 = nn.Linear(input_length ^2, input_length ^2)
        #reshape output to be 28 x 28
    def forward(self,x):
        x = F.relu(self.Fc1(x))
        x = F.relu(self.Fc2(x))
        x = torch.flatten(x)

        outputs = x
        return outputs
        
class Discriminator(nn.Module):
        def __init__(self, input_length):
            super(Discriminator,self).__init__()
            self.Fc1 = nn.Linear(128, input_length^2)
            self.Fc2 = nn.Linear(input_length^2, input_length^2)
            self.endFC = nn.Linear(input_length^2, 1)
        def forward(self,x):
            x = self.Fc1(x)
            x = self.Fc2(x)
            x = self.endFc(x)
            outputs = x
            return outputs

def Train(epochs, saveModel):

    generator = Generator(img_size).to(device)
    discriminator = Discriminator(img_size).to(device)
    generator_opt = torch.optim.Adam(generator.parameters(), lr = 0.001)
    discriminator_opt = torch.optim.Adam(discriminator.parameters(),lr = 0.001)
    loss = nn.BCELoss()

    for i in range(epochs):
        generator_opt.zero_grad()
        noise = np.random.rand(img_size, img_size)
        data = generator(noise)
        
        gen_discrim_out = discriminator(data)

        generator_loss = loss(gen_discrim_out, y_train_tensor)
        generator_loss.backward()
        generator_opt.step()

        discriminator_opt.zero_grad()
        true_labels_discriminator_out = discriminator(x_train_tensor)
        true_labels_discriminator_loss = loss(true_labels_discriminator_out, y_train_tensor)
        generator_discriminator_out = discriminator(data.detach())

        generator_discriminator_loss = loss(generator_discriminator_out, torch.zeros)
        discriminator_loss = (generator_discriminator_loss + true_labels_discriminator_loss)/2
        discriminator_loss.backward()
        discriminator_opt.step()


Train(3,False)

