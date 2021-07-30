

## Attention
In the attention class, 3 learnable neural networks are created using Linear layers to act as Key, Value & Query vectors like in an information retrieval system. 

Matrix multiplication is applied to the Q & K layers, followed by a scaling operation and then a softmax. 

The output is then multiplied by the V layer to produce a new representation of the input. 

[![image.png](https://i.postimg.cc/0QFJCg7y/image.png)](https://postimg.cc/ZBPRTMfG)

## Embedding
This defines the process of splitting the image into a 16x16 grid, and then creating a learnable patch-embedding layer. Afterwards, learnable positional encoding is added. 

[![image.png](https://i.postimg.cc/SRJY4YsN/image.png)](https://postimg.cc/zHYGSBK9)


[![image.png](https://i.postimg.cc/htjPhQR9/image.png)](https://postimg.cc/tYKHMTTT)



[![image.png](https://i.postimg.cc/52S4P5qd/image.png)](https://postimg.cc/w13YMmG0)


[![image.png](https://i.postimg.cc/02Df6rxH/image.png)](https://postimg.cc/RNCfySWc)


## Encoder
The encoder comprises of a specified number of Blocks followed by a layer normalization layer. 



## MLP
The MLP class contains a definition for two fully connected layers, with a GeLu activation function right after the first linear layer. This layer receives the output of the attention layer.

[![image.png](https://i.postimg.cc/L6Gdzxq6/image.png)](https://postimg.cc/XpKDWccT)



## Block
A block consists of a multi-head attention layer, followed by a layer normalization layer and a Multi-Layer Perceptron. 
