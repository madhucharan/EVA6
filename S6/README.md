<h1 align="center"> Normalization</h1>

1. [Plots](https://github.com/madhucharan/EVA6/blob/main/S6/README.md#accuracy-and-loss-plots)
2. [Misclassified Images](https://github.com/madhucharan/EVA6/blob/main/S6/README.md#misclassified-images)

## What is your code all about

The Net class defines a torch model with the architecture specified. When calling the object, an argument is passed specifying the type of normalisation layers to implement. The norm_layer function is called from within the model definition and returns the appropriate normalisation technique. 

## How to perform the 3 covered normalization (cannot use values from the excel sheet shared)

The different normalization methods work to standardise the input to a layer for each mini-batch. 

batch_size = 4
num_channels =  6 

## Batch Norm ( with L1 )
For each layer, the parameters mu and sigma are calculated by taking the mean and variance respectively, of each channel across all the input images within a batch. 

### Output
For this reason, the output is a mu and sigma for each channel. 

##### Example
mu & sigma of Batch size 4 each

![image](https://user-images.githubusercontent.com/7797349/121754676-0e79cd00-caca-11eb-85b6-4ddd0049db49.png)


## Layer Norm 
For each layer, the parameters mu and sigma are calculated by taking the mean and variance respectively, for all of the channels but for each image within a batch. 

### Output
The output then is a mu and sigma for each image in the batch. 
##### Example
mu & sigma of size 4 each

![image](https://user-images.githubusercontent.com/7797349/121755709-bee8d080-cacc-11eb-984e-504eecd8a3d3.png)


## Group Norm
Group norm is somewhat of a combination of the two concepts described above. The parameters are calculated by taking the mean and variance for each image, and for a user defined group of channels. This includes one group of all the channels (which would make it identical to Layer Norm), group of individual channels (identical to Instance Norm). 

### Output
Each group consisting of 2 channels (3 groups)

The output then is mu and sigma of dimension: 
number of grouped channels by number of images. 

##### Example
Output then is mu and sigma of size (3,4) each.

![image](https://user-images.githubusercontent.com/7797349/121755932-56e6ba00-cacd-11eb-902f-65f1375bcc3b.png)
![image](https://user-images.githubusercontent.com/7797349/121755956-6534d600-cacd-11eb-82a9-cef57ee5605f.png)



# Accuracy and Loss Plots

![enter image description here](https://i.postimg.cc/nV7yNjLd/image.png)

We notice that Group & Layer normalization perform well in lower epochs, all 3 eventually converge by the time we get to the 20th epoch. It should also be noted that batch normalization appears to perform better at the start up until around the second epoch. 


## Misclassified Images

---
<h2 align="center">Batch Normalization (with L1)<h2>
---

![enter image description here](https://i.postimg.cc/d3v4b0WX/image.png)

---
<h2 align="center">Group Normalization<h2>
---
  
![enter image description here](https://i.postimg.cc/13gcY6mX/image.png)

---
<h2 align="center">Layer Normalization<h2>
---
  
![enter image description here](https://i.postimg.cc/L8ZLVpDZ/image.png)




### Our Team
- [Madhu Charan](https://github.com/madhucharan)
- [Sijuade](https://github.com/cydal)
- [Siddharth Aggarwal](https://github.com/aggarwalsiddharth)
- [Deepika](https://github.com/dpkeee)
