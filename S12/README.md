
## Attention

- Attention process can be explained as a way to understand which part of an image(when it comes to vision domain) should given more priority or to be concentrated on by leaving the rest of the parts in the image.
- It has 3 parts
    - Key
    - Query
    - Value
- All the three are indirect representations of input image
- Before proceeding, input image is divided into patches i.e several smaller parts to be passed as an input one by one into the attention network
- The process is as follows with an example
    - Let us consider an Image divided into 16x16 patches _(see the example image below on how we made a 3X3 image into 9 patches)_
    - Each patch is passed as an input to the 3 different neural networks that produces 3 outputs __keys__(K),__values__(V) and __query__(Q)
    - Now we have 3 different outputs from the networks
    - As the next process, we will have to perform a matrix multiplication for K and Q followed by a softmax operation.
    - Now the V patch values will be multiplied with the output of softmax that gives us the attention values which tells us the importance of patches of one another
    - This above process will be repeated for each patch and in this case 16 times.

In the attention class, 3 learnable neural networks are created using Linear layers to act as Key, Value & Query vectors like in an information retrieval system. 

Matrix multiplication is applied to the Q & K layers, followed by a scaling operation and then a softmax. 

The output is then multiplied by the V layer to produce a new representation of the input. 

[![image.png](https://i.postimg.cc/0QFJCg7y/image.png)](https://postimg.cc/ZBPRTMfG)

## Embedding

Embedding in general is a vector representation of a specific input(ex: text,variable etc). Embeddings are replacement of one hot representation as one hot represenatations, 
if the dimensionality is high, we cannot manage the representations and also there is no way to map or group similar items(can be words incase of text) where as in case of embeddings,
we can group them into same space and also train them inroder to make the embeddings learn on how to map the inputs that belong to similar category will be close to each other and the 
embeddings are learnable.

Incase of Images, We can use the same mechanism of creating embdeddings where it can be formed as a vector representation by making the whole image into patches(like explained above).
The patches will also remember their positions in the big image(original).This can be called as positional embeddings

The process can be 
- Configure the patch size
- Make the image into patches
- Transform into patch embeddings and add positional embeddings

This defines the process of splitting the image into a 16x16 grid, and then creating a learnable patch-embedding layer. Afterwards, learnable positional encoding is added. 

[![image.png](https://i.postimg.cc/SRJY4YsN/image.png)](https://postimg.cc/zHYGSBK9)


[![image.png](https://i.postimg.cc/htjPhQR9/image.png)](https://postimg.cc/tYKHMTTT)



[![image.png](https://i.postimg.cc/52S4P5qd/image.png)](https://postimg.cc/w13YMmG0)


[![image.png](https://i.postimg.cc/02Df6rxH/image.png)](https://postimg.cc/RNCfySWc)


## Encoder
The encoder comprises of a specified number of Blocks followed by a layer normalization layer. 
![block](https://user-images.githubusercontent.com/62477860/127709902-a6983b88-da05-4731-8972-01ed932f74c7.png)



## MLP
The MLP class contains a definition for two fully connected layers, with a GeLu activation function right after the first linear layer. This layer receives the output of the attention layer.

[![image.png](https://i.postimg.cc/L6Gdzxq6/image.png)](https://postimg.cc/XpKDWccT)



## Block
A block consists of a multi-head attention layer, followed by a layer normalization layer and a Multi-Layer Perceptron. 
