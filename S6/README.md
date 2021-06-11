
# what is your code all about

The Net class defines a torch model with the architecture specified. When calling the object, an argument is passed specifying the type of normalisation layers to implement. The norm_layer function is called from within the model definition and returns the appropriate normalisation technique. 

# How to perform the 3 covered normalization (cannot use values from the excel sheet shared)

The different normalization methods work to standardise the input to a layer for each mini-batch. 

batch_size = 4
num_channels =  6 

## Batch Norm
For each layer, the parameters mu and sigma are calculated by taking the mean and variance respectively, of each channel across all the input images within a batch. 

### Output
For this reason, the output is a mu and sigma for each channel. 

##### Example
mu & sigma of size 6 each


## Layer Norm 
For each layer, the parameters mu and sigma are calculated by taking the mean and variance respectively, for all of the channels but for each image within a batch. 

### Output
The output then is a mu and sigma for each image in the batch. 
##### Example
mu & sigma of size 4 each


## Group Norm
Group norm is somewhat of a combination of the two concepts described above. The parameters are calculated by taking the mean and variance for each image, and for a user defined group of channels. This includes one group of all the channels (which would make it identical to Layer Norm), group of individual channels (identical to Instance Norm). 

### Output
Each group consisting of 2 channels (3 groups)

The output then is mu and sigma of dimension: 
number of grouped channels by number of images. 

##### Example
Output then is mu and sigma of size (3,4) each.





### Our Team
- [Madhu Charan](https://github.com/madhucharan)
- [Sijuade](https://github.com/cydal)
- [Siddharth Aggarwal](https://github.com/aggarwalsiddharth)
- [Deepika](https://github.com/dpkeee)
