
# **LATE ASSIGNMENT**


### Assignment 7

1.  Run this [network](https://colab.research.google.com/drive/1qlewMtxcAJT6fIJdmMh8pSf2e-dh51Rw).
2.  Fix the network above:  
    1.  change the code such that it uses GPU
    2.  change the architecture to C1C2C3C40 (No MaxPooling, but 3 3x3 layers with stride of 2 instead)  **(If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)**
    3.  total RF must be more than  **52**
    4.  **two**  of the layers must use Depthwise Separable Convolution
    5.  one of the layers must use Dilated Convolution
    6.  use GAP (compulsory  **mapped to # of classes**):-  _CANNOT_ add FC after GAP to target #of classes
    7.  use albumentation library and apply:
        1.  horizontal flip
        2.  shiftScaleRotate
        3.  coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
        4.  **grayscale**
    8.  achieve  **87%**  accuracy, as many epochs as you want. Total Params to be less than  **100k**.
    9.  upload to Github
    10.  Attempt S7-Assignment Solution. Questions in the Assignment QnA are:
        1.  Which assignment are you submitting? (early/late)
        2.  Please mention the name of your partners who are submitting EXACTLY the same assignment. Please note if the assignments are different, then all the names mentioned here will get the lowest score. So please check with your team if they are submitting even a slightly different assignment.
            
        3.  copy paste your model code from your model.py file (full code) [125]
            
        4.  copy paste output of torchsummary [125]
            
        5.  copy-paste the code where you implemented albumentation transformation for all three transformations [125]
            
        6.  copy paste your training log (you must be running validation/text after each Epoch [125]
            
        7.  Share the link for your README.md file. [200]


# **Project Files**
model.py - Contains model architecture
augment.py - Albumentations augmentation definition
utils.py - General utility files
dataset.py - Cifar10 dataset class definition


### Output Shape Param 
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             864
       BatchNorm2d-2           [-1, 32, 32, 32]              64
              ReLU-3           [-1, 32, 32, 32]               0
            Conv2d-4           [-1, 32, 32, 32]             288
       BatchNorm2d-5           [-1, 32, 32, 32]              64
              ReLU-6           [-1, 32, 32, 32]               0
            Conv2d-7           [-1, 32, 34, 34]           1,024
       BatchNorm2d-8           [-1, 32, 34, 34]              64
              ReLU-9           [-1, 32, 34, 34]               0
depthwise_separable_conv-10           [-1, 32, 34, 34]               0
           Conv2d-11           [-1, 32, 34, 34]             288
      BatchNorm2d-12           [-1, 32, 34, 34]              64
             ReLU-13           [-1, 32, 34, 34]               0
           Conv2d-14           [-1, 32, 36, 36]           1,024
      BatchNorm2d-15           [-1, 32, 36, 36]              64
             ReLU-16           [-1, 32, 36, 36]               0
depthwise_separable_conv-17           [-1, 32, 36, 36]               0
           Conv2d-18           [-1, 32, 36, 36]             288
      BatchNorm2d-19           [-1, 32, 36, 36]              64
             ReLU-20           [-1, 32, 36, 36]               0
           Conv2d-21           [-1, 32, 38, 38]           1,024
      BatchNorm2d-22           [-1, 32, 38, 38]              64
             ReLU-23           [-1, 32, 38, 38]               0
depthwise_separable_conv-24           [-1, 32, 38, 38]               0
           Conv2d-25           [-1, 32, 17, 17]           9,216
      BatchNorm2d-26           [-1, 32, 17, 17]              64
             ReLU-27           [-1, 32, 17, 17]               0
           Conv2d-28           [-1, 64, 17, 17]          18,432
      BatchNorm2d-29           [-1, 64, 17, 17]             128
             ReLU-30           [-1, 64, 17, 17]               0
           Conv2d-31           [-1, 64, 17, 17]             576
      BatchNorm2d-32           [-1, 64, 17, 17]             128
             ReLU-33           [-1, 64, 17, 17]               0
           Conv2d-34           [-1, 64, 19, 19]           4,096
      BatchNorm2d-35           [-1, 64, 19, 19]             128
             ReLU-36           [-1, 64, 19, 19]               0
depthwise_separable_conv-37           [-1, 64, 19, 19]               0
           Conv2d-38           [-1, 64, 19, 19]             576
      BatchNorm2d-39           [-1, 64, 19, 19]             128
             ReLU-40           [-1, 64, 19, 19]               0
           Conv2d-41           [-1, 64, 21, 21]           4,096
      BatchNorm2d-42           [-1, 64, 21, 21]             128
             ReLU-43           [-1, 64, 21, 21]               0
depthwise_separable_conv-44           [-1, 64, 21, 21]               0
           Conv2d-45           [-1, 64, 21, 21]             576
      BatchNorm2d-46           [-1, 64, 21, 21]             128
             ReLU-47           [-1, 64, 21, 21]               0
           Conv2d-48           [-1, 64, 23, 23]           4,096
      BatchNorm2d-49           [-1, 64, 23, 23]             128
             ReLU-50           [-1, 64, 23, 23]               0
depthwise_separable_conv-51           [-1, 64, 23, 23]               0
           Conv2d-52           [-1, 16, 10, 10]           9,216
      BatchNorm2d-53           [-1, 16, 10, 10]              32
             ReLU-54           [-1, 16, 10, 10]               0
           Conv2d-55           [-1, 32, 10, 10]           4,608
      BatchNorm2d-56           [-1, 32, 10, 10]              64
             ReLU-57           [-1, 32, 10, 10]               0
           Conv2d-58           [-1, 32, 10, 10]             288
      BatchNorm2d-59           [-1, 32, 10, 10]              64
             ReLU-60           [-1, 32, 10, 10]               0
           Conv2d-61           [-1, 32, 12, 12]           1,024
      BatchNorm2d-62           [-1, 32, 12, 12]              64
             ReLU-63           [-1, 32, 12, 12]               0
depthwise_separable_conv-64           [-1, 32, 12, 12]               0
           Conv2d-65           [-1, 32, 12, 12]             288
      BatchNorm2d-66           [-1, 32, 12, 12]              64
             ReLU-67           [-1, 32, 12, 12]               0
           Conv2d-68           [-1, 32, 14, 14]           1,024
      BatchNorm2d-69           [-1, 32, 14, 14]              64
             ReLU-70           [-1, 32, 14, 14]               0
depthwise_separable_conv-71           [-1, 32, 14, 14]               0
           Conv2d-72           [-1, 32, 14, 14]             288
      BatchNorm2d-73           [-1, 32, 14, 14]              64
             ReLU-74           [-1, 32, 14, 14]               0
           Conv2d-75           [-1, 32, 16, 16]           1,024
      BatchNorm2d-76           [-1, 32, 16, 16]              64
             ReLU-77           [-1, 32, 16, 16]               0
depthwise_separable_conv-78           [-1, 32, 16, 16]               0
           Conv2d-79             [-1, 32, 6, 6]           9,216
      BatchNorm2d-80             [-1, 32, 6, 6]              64
             ReLU-81             [-1, 32, 6, 6]               0
           Conv2d-82             [-1, 32, 4, 4]           9,216
      BatchNorm2d-83             [-1, 32, 4, 4]              64
             ReLU-84             [-1, 32, 4, 4]               0
           Conv2d-85             [-1, 32, 4, 4]             288
      BatchNorm2d-86             [-1, 32, 4, 4]              64
             ReLU-87             [-1, 32, 4, 4]               0
           Conv2d-88             [-1, 32, 6, 6]           1,024
      BatchNorm2d-89             [-1, 32, 6, 6]              64
             ReLU-90             [-1, 32, 6, 6]               0
depthwise_separable_conv-91             [-1, 32, 6, 6]               0
           Conv2d-92             [-1, 42, 4, 4]          12,096
      BatchNorm2d-93             [-1, 42, 4, 4]              84
             ReLU-94             [-1, 42, 4, 4]               0
        AvgPool2d-95             [-1, 42, 1, 1]               0
           Conv2d-96             [-1, 32, 1, 1]           1,344
           Conv2d-97             [-1, 10, 1, 1]             320
================================================================
Total params: 99,956
Trainable params: 99,956
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 13.05
Params size (MB): 0.38
Estimated Total Size (MB): 13.44
----------------------------------------------------------------
```
# Data Augmentation

```python
train_transform = A.Compose([
                             A.HorizontalFlip(p=0.4),
                             A.ShiftScaleRotate(),
                             A.Normalize(mean=([0.49139968, 0.48215841, 0.44653091]), 
                                         std=(0.24703223, 0.24348513, 0.26158784)),
                             A.CoarseDropout(max_holes=1, 
                                             max_height=16, 
                                             max_width=16, 
                                             min_holes=1, 
                                             min_height=16,
                                             min_width=16,
                                             fill_value=([0.49139968, 0.48215841, 0.44653091])),
                             A.ToGray(),
                             ToTensorV2(),
])

test_transform = A.Compose([
                            A.Normalize(mean=(0.49139968, 0.48215841, 0.44653091), std=(0.24703223, 0.24348513, 0.26158784)),
                            ToTensorV2(),
])
```
## Training 
* Max Pooling replaced with Strided Convolution
* Depth-Wise separable convolution used in multiple places
* Dilation of 2 applied in multiple places

## Receptive Field
| Operation | K | P | S | nin | nout | channel_in | channel_out | dilation | jin | jout | rin | rout |
|-----------|---|---|---|-----|------|------------|-------------|----------|-----|------|-----|------|
| Conv      | 3 | 1 | 1 | 32  | 32   | 3          | 32          | 1        | 1   | 1    | 1   | 3    |
| ConvDS    | 3 | 1 | 1 | 32  | 32   | 32         | 32          | 1        | 1   | 1    | 3   | 5    |
| ConvP     | 1 | 1 | 1 | 32  | 34   | 32         | 32          | 1        | 1   | 1    | 5   | 5    |
| ConvDS    | 3 | 1 | 1 | 34  | 34   | 32         | 32          | 1        | 1   | 1    | 5   | 7    |
| ConvP     | 1 | 1 | 1 | 34  | 36   | 32         | 32          | 1        | 1   | 1    | 7   | 7    |
| ConvDS    | 3 | 1 | 1 | 36  | 36   | 32         | 32          | 1        | 1   | 1    | 7   | 9    |
| ConvP     | 1 | 1 | 1 | 36  | 38   | 32         | 32          | 1        | 1   | 1    | 9   | 9    |
| Conv      | 3 | 0 | 2 | 38  | 17   | 32         | 32          | 2        | 1   | 2    | 9   | 11   |
| Conv      | 3 | 1 | 1 | 17  | 17   | 32         | 64          | 1        | 2   | 2    | 11  | 15   |
| ConvDS    | 3 | 1 | 1 | 17  | 17   | 64         | 64          | 1        | 2   | 2    | 15  | 19   |
| ConvP     | 1 | 1 | 1 | 17  | 19   | 64         | 64          | 1        | 2   | 2    | 19  | 19   |
| ConvDS    | 3 | 1 | 1 | 19  | 19   | 64         | 64          | 1        | 2   | 2    | 19  | 23   |
| ConvP     | 1 | 1 | 1 | 19  | 21   | 64         | 64          | 1        | 2   | 2    | 23  | 23   |
| ConvDS    | 3 | 1 | 1 | 21  | 21   | 64         | 64          | 1        | 2   | 2    | 23  | 27   |
| ConvP     | 1 | 1 | 1 | 21  | 23   | 64         | 64          | 1        | 2   | 2    | 27  | 27   |
| Conv      | 3 | 0 | 2 | 23  | 10   | 64         | 16          | 2        | 2   | 4    | 27  | 31   |
| Conv      | 3 | 1 | 1 | 10  | 10   | 16         | 32          | 1        | 4   | 4    | 31  | 39   |
| ConvDS    | 3 | 1 | 1 | 10  | 10   | 32         | 32          | 1        | 4   | 4    | 39  | 47   |
| ConvP     | 1 | 1 | 1 | 10  | 12   | 32         | 32          | 1        | 4   | 4    | 47  | 47   |
| ConvDS    | 3 | 1 | 1 | 12  | 12   | 32         | 32          | 1        | 4   | 4    | 47  | 55   |
| ConvP     | 1 | 1 | 1 | 12  | 14   | 32         | 32          | 1        | 4   | 4    | 55  | 55   |
| ConvDS    | 3 | 1 | 1 | 14  | 14   | 32         | 32          | 1        | 4   | 4    | 55  | 63   |
| ConvP     | 1 | 1 | 1 | 14  | 16   | 32         | 32          | 1        | 4   | 4    | 63  | 63   |
| Conv      | 3 | 0 | 2 | 16  | 6    | 32         | 32          | 2        | 4   | 8    | 63  | 71   |
| Conv      | 3 | 0 | 1 | 6   | 4    | 32         | 32          | 1        | 8   | 8    | 71  | 87   |


## Training Log (Last 15 epochs)

EPOCH: 254
Loss=0.48513874411582947 Batch_id=390 Accuracy=78.99: 100%|██████████| 391/391 [00:23<00:00, 16.32it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0034, Accuracy: 8507/10000 (85.07%)

EPOCH: 255
Loss=0.5073024034500122 Batch_id=390 Accuracy=79.17: 100%|██████████| 391/391 [00:23<00:00, 16.39it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0034, Accuracy: 8524/10000 (85.24%)

EPOCH: 256
Loss=0.5886918902397156 Batch_id=390 Accuracy=79.01: 100%|██████████| 391/391 [00:23<00:00, 16.34it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0034, Accuracy: 8550/10000 (85.50%)

EPOCH: 257
Loss=0.6579458117485046 Batch_id=390 Accuracy=79.19: 100%|██████████| 391/391 [00:23<00:00, 16.37it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0036, Accuracy: 8460/10000 (84.60%)

EPOCH: 258
Loss=0.5604768395423889 Batch_id=390 Accuracy=79.17: 100%|██████████| 391/391 [00:24<00:00, 16.27it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0034, Accuracy: 8519/10000 (85.19%)

EPOCH: 259
Loss=0.8374205827713013 Batch_id=390 Accuracy=79.24: 100%|██████████| 391/391 [00:23<00:00, 16.45it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0034, Accuracy: 8504/10000 (85.04%)

EPOCH: 260
Loss=0.5479215383529663 Batch_id=390 Accuracy=79.26: 100%|██████████| 391/391 [00:24<00:00, 16.23it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0035, Accuracy: 8487/10000 (84.87%)

EPOCH: 261
Loss=0.5098845362663269 Batch_id=390 Accuracy=79.39: 100%|██████████| 391/391 [00:23<00:00, 16.35it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0035, Accuracy: 8534/10000 (85.34%)

EPOCH: 262
Loss=0.7909579873085022 Batch_id=390 Accuracy=79.09: 100%|██████████| 391/391 [00:23<00:00, 16.32it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0033, Accuracy: 8528/10000 (85.28%)

EPOCH: 263
Loss=0.49288877844810486 Batch_id=390 Accuracy=79.06: 100%|██████████| 391/391 [00:23<00:00, 16.32it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0034, Accuracy: 8550/10000 (85.50%)

EPOCH: 264
Loss=0.48313555121421814 Batch_id=390 Accuracy=79.21: 100%|██████████| 391/391 [00:23<00:00, 16.38it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0034, Accuracy: 8533/10000 (85.33%)

EPOCH: 265
Loss=0.7641388177871704 Batch_id=390 Accuracy=78.96: 100%|██████████| 391/391 [00:23<00:00, 16.44it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0034, Accuracy: 8490/10000 (84.90%)

EPOCH: 266
Loss=0.6971480846405029 Batch_id=390 Accuracy=79.20: 100%|██████████| 391/391 [00:23<00:00, 16.39it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0034, Accuracy: 8535/10000 (85.35%)

EPOCH: 267
Loss=0.3924857974052429 Batch_id=390 Accuracy=79.35: 100%|██████████| 391/391 [00:24<00:00, 16.29it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0034, Accuracy: 8529/10000 (85.29%)

EPOCH: 268
Loss=0.48865407705307007 Batch_id=390 Accuracy=79.40: 100%|██████████| 391/391 [00:24<00:00, 16.28it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0034, Accuracy: 8520/10000 (85.20%)

EPOCH: 269
Loss=0.5213862061500549 Batch_id=223 Accuracy=79.66:  57%|█████▋    | 224/391 [23:41<14:57:46, 322.55s/it]


## Results Achieved 

Best Training Accuracy - 79.66
Best Test Accuracy - 85.20
Total Number of Parameters - 99,956
Number of Epochs - 269
Receptor Field Calculation - 87


## Comment
Model is not overfitting
Late assignment requirements attempted but early assignment results
Model trained for 300 epochs achieves 87% but Colab crashed the final time it was trained



----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             864
       BatchNorm2d-2           [-1, 32, 32, 32]              64
              ReLU-3           [-1, 32, 32, 32]               0
            Conv2d-4           [-1, 32, 32, 32]             288
       BatchNorm2d-5           [-1, 32, 32, 32]              64
              ReLU-6           [-1, 32, 32, 32]               0
            Conv2d-7           [-1, 32, 34, 34]           1,024
       BatchNorm2d-8           [-1, 32, 34, 34]              64
              ReLU-9           [-1, 32, 34, 34]               0
depthwise_separable_conv-10           [-1, 32, 34, 34]               0
           Conv2d-11           [-1, 32, 34, 34]             288
      BatchNorm2d-12           [-1, 32, 34, 34]              64
             ReLU-13           [-1, 32, 34, 34]               0
           Conv2d-14           [-1, 32, 36, 36]           1,024
      BatchNorm2d-15           [-1, 32, 36, 36]              64
             ReLU-16           [-1, 32, 36, 36]               0
depthwise_separable_conv-17           [-1, 32, 36, 36]               0
           Conv2d-18           [-1, 32, 36, 36]             288
      BatchNorm2d-19           [-1, 32, 36, 36]              64
             ReLU-20           [-1, 32, 36, 36]               0
           Conv2d-21           [-1, 32, 38, 38]           1,024
      BatchNorm2d-22           [-1, 32, 38, 38]              64
             ReLU-23           [-1, 32, 38, 38]               0
depthwise_separable_conv-24           [-1, 32, 38, 38]               0
           Conv2d-25           [-1, 32, 17, 17]           9,216
      BatchNorm2d-26           [-1, 32, 17, 17]              64
             ReLU-27           [-1, 32, 17, 17]               0
           Conv2d-28           [-1, 64, 17, 17]          18,432
      BatchNorm2d-29           [-1, 64, 17, 17]             128
             ReLU-30           [-1, 64, 17, 17]               0
           Conv2d-31           [-1, 64, 17, 17]             576
      BatchNorm2d-32           [-1, 64, 17, 17]             128
             ReLU-33           [-1, 64, 17, 17]               0
           Conv2d-34           [-1, 64, 19, 19]           4,096
      BatchNorm2d-35           [-1, 64, 19, 19]             128
             ReLU-36           [-1, 64, 19, 19]               0
depthwise_separable_conv-37           [-1, 64, 19, 19]               0
           Conv2d-38           [-1, 64, 19, 19]             576
      BatchNorm2d-39           [-1, 64, 19, 19]             128
             ReLU-40           [-1, 64, 19, 19]               0
           Conv2d-41           [-1, 64, 21, 21]           4,096
      BatchNorm2d-42           [-1, 64, 21, 21]             128
             ReLU-43           [-1, 64, 21, 21]               0
depthwise_separable_conv-44           [-1, 64, 21, 21]               0
           Conv2d-45           [-1, 64, 21, 21]             576
      BatchNorm2d-46           [-1, 64, 21, 21]             128
             ReLU-47           [-1, 64, 21, 21]               0
           Conv2d-48           [-1, 64, 23, 23]           4,096
      BatchNorm2d-49           [-1, 64, 23, 23]             128
             ReLU-50           [-1, 64, 23, 23]               0
depthwise_separable_conv-51           [-1, 64, 23, 23]               0
           Conv2d-52           [-1, 16, 10, 10]           9,216
      BatchNorm2d-53           [-1, 16, 10, 10]              32
             ReLU-54           [-1, 16, 10, 10]               0
           Conv2d-55           [-1, 32, 10, 10]           4,608
      BatchNorm2d-56           [-1, 32, 10, 10]              64
             ReLU-57           [-1, 32, 10, 10]               0
           Conv2d-58           [-1, 32, 10, 10]             288
      BatchNorm2d-59           [-1, 32, 10, 10]              64
             ReLU-60           [-1, 32, 10, 10]               0
           Conv2d-61           [-1, 32, 12, 12]           1,024
      BatchNorm2d-62           [-1, 32, 12, 12]              64
             ReLU-63           [-1, 32, 12, 12]               0
depthwise_separable_conv-64           [-1, 32, 12, 12]               0
           Conv2d-65           [-1, 32, 12, 12]             288
      BatchNorm2d-66           [-1, 32, 12, 12]              64
             ReLU-67           [-1, 32, 12, 12]               0
           Conv2d-68           [-1, 32, 14, 14]           1,024
      BatchNorm2d-69           [-1, 32, 14, 14]              64
             ReLU-70           [-1, 32, 14, 14]               0
depthwise_separable_conv-71           [-1, 32, 14, 14]               0
           Conv2d-72           [-1, 32, 14, 14]             288
      BatchNorm2d-73           [-1, 32, 14, 14]              64
             ReLU-74           [-1, 32, 14, 14]               0
           Conv2d-75           [-1, 32, 16, 16]           1,024
      BatchNorm2d-76           [-1, 32, 16, 16]              64
             ReLU-77           [-1, 32, 16, 16]               0
depthwise_separable_conv-78           [-1, 32, 16, 16]               0
           Conv2d-79             [-1, 32, 6, 6]           9,216
      BatchNorm2d-80             [-1, 32, 6, 6]              64
             ReLU-81             [-1, 32, 6, 6]               0
           Conv2d-82             [-1, 32, 4, 4]           9,216
      BatchNorm2d-83             [-1, 32, 4, 4]              64
             ReLU-84             [-1, 32, 4, 4]               0
           Conv2d-85             [-1, 32, 4, 4]             288
      BatchNorm2d-86             [-1, 32, 4, 4]              64
             ReLU-87             [-1, 32, 4, 4]               0
           Conv2d-88             [-1, 32, 6, 6]           1,024
      BatchNorm2d-89             [-1, 32, 6, 6]              64
             ReLU-90             [-1, 32, 6, 6]               0
depthwise_separable_conv-91             [-1, 32, 6, 6]               0
           Conv2d-92             [-1, 42, 4, 4]          12,096
      BatchNorm2d-93             [-1, 42, 4, 4]              84
             ReLU-94             [-1, 42, 4, 4]               0
        AvgPool2d-95             [-1, 42, 1, 1]               0
           Conv2d-96             [-1, 32, 1, 1]           1,344
           Conv2d-97             [-1, 10, 1, 1]             320
================================================================
Total params: 99,956
Trainable params: 99,956
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 13.05
Params size (MB): 0.38
Estimated Total Size (MB): 13.44
----------------------------------------------------------------
