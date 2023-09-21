# Image-Recognition-system

✨ **Alzheimer's Intelligent Diagnosis Web Application based on 3D Convolutional Neural Network and the ADNI Dataset**

A simple medical image recognition system, image recognition visualization interface, OCR, fast deployment of deep learning models as web applications, web prediction system, image recognition front-end web page, image recognition Demo display-Pywebio. AI artificial intelligence image recognition-Pytorch; nii medical image processing; ADNI dataset. 100% pure Python code, lightweight, easy to reproduce

[简体中文文档](./README.md)

[Personal website: www.bytesc.top](http://www.bytesc.top) includes online demonstrations

[Personal blog: blog.bytesc.top](http://blog.bytesc.top) 

🔔 If you have any project-related questions, feel free to raise an `issue` in this project, I will usually reply within 24 hours.

## Function introduction

- 1. Intelligent diagnosis of Alzheimer's disease based on brain MRI medical images
- 2. Written in pure python, lightweight, easy to reproduce and deploy
- 3. High code readability, with extremely detailed comments in the core part

## Interface display

- Upload image
  ![image](./readme_img/1.png)
- Return result
  ![image](./readme_img/2.png)
- Model output chart
  ![image](./readme_img/3.png)
- View uploaded images
  ![image](./readme_img/3-1.png)

## How to use

python version 3.9

Requires `4GB` or more memory

First install dependencies

```bash
pip install -r requirement.txt
```

zlzheimer-diagnostic-system.py is the project entry point, run this file to start the server

```bash
python zlzheimer-diagnostic-system.py
```

Copy the link to the browser and open it

![image](./readme_img/4.png)

Click "Demo" to enter the Web interface

![image](./readme_img/5.png)

After that, you can click "Use example images" to use the default test cases. You can also upload .nii image files for testing.
I provide a small number of sample image data in the [`lqdata` repository](https://github.com/bytesc/lqdata).

```bash
git clone https://github.com/bytesc/lqdata.git
```

- If an error is reported after uploading the image, you may need to manually create a folder `uploaded_img` in the root directory.

## Project structure

```
.
│ zlzheimer-diagnostic-system.py
│ datasets.py
│ model.py
│ train.py
│ myModel_109.pth
│ README.md
│ requirements.txt
│
├─demodata
│ │ demo.nii
├─readme_img
└─uploaded_img
```

- zlzheimer-diagnostic-system.py main project file for starting Web applications
- datasets.py processing dataset
- model.py defining model
- train.py training model
- myModel_109.pth trained model
- readme_img folder for storing uploaded medical images and rendered pictures
- demodata folder for storing some medical image files for testing.

## Core code of the classifier

```python
from torch import nn
import torch

class ClassificationModel3D(nn.Module):
    """Classification model"""
    def __init__(self, dropout=0.4, dropout2=0.4):
        nn.Module.__init__(self)

        # Define four Conv3d layers
        self.Conv_1 = nn.Conv3d(1, 8, 3)  # Input channel is 1, output channel is 8, kernel size is 3x3x3
        self.Conv_2 = nn.Conv3d(8, 16, 3)  # Input channel is 8, output channel is 16, kernel size is 3x3x3
        self.Conv_3 = nn.Conv3d(16, 32, 3)  # Input channel is 16, output channel is 32, kernel size is 3x3x3
        self.Conv_4 = nn.Conv3d(32, 64, 3)  # Input channel is 32, output channel is 64, kernel size is 3x3x3

        # Define four BatchNorm3d layers, one after each convolution layer
        self.Conv_1_bn = nn.BatchNorm3d(8)
        self.Conv_2_bn = nn.BatchNorm3d(16)
        self.Conv_3_bn = nn.BatchNorm3d(32)
        self.Conv_4_bn = nn.BatchNorm3d(64)

        # Define four MaxPool3d layers, one after each convolution layer
        self.Conv_1_mp = nn.MaxPool3d(2)  # Pooling kernel size is 2
        self.Conv_2_mp = nn.MaxPool3d(3)  # Pooling kernel size is 3
        self.Conv_3_mp = nn.MaxPool3d(2)  # Pooling kernel size is 2
        self.Conv_4_mp = nn.MaxPool3d(3)  # Pooling kernel size is 3

        # Define two fully connected layers
        self.dense_1 = nn.Linear(4800, 128)  # Input dimension is 4800, output dimension is 128
        self.dense_2 = nn.Linear(128, 5)     # Input dimension is 128, output dimension is 5. Since this is a five-class problem, the final output dimension must be 5

        # Define ReLU activation function and dropout layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)   # Prevent overfitting
        self.dropout2 = nn.Dropout(dropout2) # Enhance robustness

    def forward(self, x):
        # First convolutional layer
        x = self.relu(self.Conv_1_bn(self.Conv_1(x)))
        """
        This line of code performs convolutional, batch normalization and ReLU activation operations on the input x.

        self.Conv_1(x) performs a 3D convolution operation on the input x and outputs a feature map.

        self.Conv_1_bn(...) performs batch normalization on the feature map output by the convolution operation to obtain a normalized feature map.

        self.relu(...) performs a ReLU activation function operation on the normalized feature map to obtain an activated feature map.

        The purpose of this operation is to extract features from the input x and nonlinearize them so that the network can better learn these features. The batch normalization technique used here can accelerate the training process of the model and improve its generalization ability. The final output result is a feature map x processed by convolutional, batch normalization and ReLU activation functions.
        """
        # Max pooling of the first convolutional layer
        x = self.Conv_1_mp(x)
        """
        This line of code performs a maximum pooling operation on the input x to reduce the size of the feature map by half.

        self.Conv_1_mp(...) performs a maximum pooling operation on the input x with a pooling kernel size of 2.

        The pooling operation extracts the maximum value in each pooling window in the feature map as the value at the corresponding position in the output feature map, thereby reducing the size of the feature map by half.

        Maximum pooling can help the network achieve spatial invariance so that it can recognize the same features when there are slight changes in the input. In this model, after maximum pooling, the feature map x will be passed to the next convolutional layer for feature extraction and nonlinear processing.
        """
        
        # Second convolutional layer
        x = self.relu(self.Conv_2_bn(self.Conv_2(x)))
        # Max pooling of second convolutional layer
        x = self.Conv_2_mp(x)
        # Third convolutional layer
        x = self.relu(self.Conv_3_bn(self.Conv_3(x)))
        # Max pooling of third convolutional layer
        x = self.Conv_3_mp(x)
        # Fourth convolutional layer
        x = self.relu(self.Conv_4_bn(self.Conv_4(x)))
        # Max pooling of fourth convolutional layer
        x = self.Conv_4_mp(x)
        # Flatten tensor into a one-dimensional vector
        x = x.view(x.size(0), -1)
        """
        This line of code flattens the input tensor x into a one-dimensional vector.

        x.size(0) gets the size of the first dimension of the input tensor x, which is the batch size of the tensor.

        -1 means to flatten the second dimension and all dimensions after it into one dimension.

        x.view(...) performs a shape transformation on the input tensor x, flattening it into a one-dimensional vector.

        The purpose of this operation is to transform the feature map x processed by convolution and pooling into a one-dimensional vector so that it can be passed to the fully connected layer for classification or regression tasks. The size of the flattened vector is (batch_size, num_features), where batch_size is the batch size of the input tensor and num_features is the number of elements in the flattened vector, which is also the number of features after convolution and pooling processing.
        """

        # dropout layer
        x = self.dropout(x)
        """
        This line of code performs a dropout operation on the input tensor x, i.e., sets some elements of the input tensor to zero with a certain probability.

        self.dropout(...) performs a dropout operation on the input tensor x, with a dropout probability of dropout.

        The dropout operation sets some elements of the input tensor to zero with a certain probability, achieving the purpose of random deactivation. This can reduce overfitting and enhance the generalization ability of the model.

        In this model, the dropout operation is applied before the fully connected layer, which can help the model better learn the features of the data and prevent overfitting. The resulting x tensor is the result after the dropout operation and will be passed to the next fully connected layer for processing.
        """
        # fully connected layer 1
        x = self.relu(self.dense_1(x))
        """
        This line of code performs a fully connected operation on the input tensor x and applies the ReLU activation function.

        self.dense_1(x) performs a fully connected operation on the input tensor x, mapping it to a feature space of size 128.

        self.relu(...) applies the ReLU activation function to the output of the fully connected layer to obtain an activated feature vector.

        In this model, the role of the fully connected layer is to map the feature vector processed by convolution, pooling, and dropout to a new feature space for classification or regression tasks. The role of the ReLU activation function is to nonlinearize the feature vector so that the network can better learn the nonlinear correlation in the data. The resulting x tensor is the result after processing by the fully connected layer and ReLU activation function and will be passed to the next dropout layer for processing.
        """
        # dropout2 layer
        x = self.dropout2(x)
        # fully connected layer 2
        x = self.dense_2(x)
        # return output result
        return x


if __name__ == "__main__":
    # create an instance of ClassificationModel3D class named model, i.e., create a 3D image classification model
    model = ClassificationModel3D()

    # create a test tensor test_tensor with shape (1, 1, 166, 256, 256),
    # where 1 represents batch size, 1 represents input channel number, 166, 256 and 256 represent depth, height and width of input data respectively
    test_tensor = torch.ones(1, 1, 166, 256, 256)

    # perform forward propagation on test tensor test_tensor to obtain output result output from model
    output = model(test_tensor)

    # print shape of output result, i.e., (batch_size, num_classes), where batch_size is batch size of test tensor and num_classes is number of classes in classification task
    print(output.shape)

```

If you need to train the model yourself, please go to the [ADNI official website](https://adni.loni.usc.edu) to obtain complete data. The dataset file structure should be as follows:

```txt
/lqdata/date/
├─AD
│  1.nii
│  2.nii
│  ...
│
├─CN
│  1.nii
│  ...
├─EMCI
├─LMCI
├─MCI
│
└─test
    ├─AD
    │  1.nii
    │  ...
    ├─CN
    ├─EMCI
    ├─LMCI
    └─MCI
```

ref: https://github.com/moboehle/Pytorch-LRP

Dataset: https://adni.loni.usc.edu
