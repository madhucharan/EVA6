# Final Model

Our best notebook has been uploaded  [here](https://github.com/madhucharan/EVA6/blob/main/S9/EVA6_Assignment9.ipynb) 
We have achieved the test accuracy of 85.46 in 24 Epochs.

Epochs - 24  
LR Scheduler - OneCycleLR

Augnmentation Techniques - 
-   RandomCrop(32, padding=4)
-  Horizontal Flip
-   CutOut(8x8)



# Other useful files 

Our model can be found  [here](https://github.com/madhucharan/EVA6/blob/main/S9/EVA6_Assignment9.ipynb) 
Our augmentations can be found  [here](https://github.com/madhucharan/EVA6/blob/main/S9/EVA6_Assignment9.ipynb) 
Our maxLR finding file can be found  [here](https://github.com/madhucharan/EVA6/blob/main/S9/EVA6_Assignment9.ipynb) 

## MaxLR

LR suggestion: steepest gradient 
Suggested LR: **1.73E-02**
  
![enter image description here](https://i.ibb.co/MssxZLV/Unknown.png)

## Logs

The training and testing logs are as follows:
EPOCH: 0

Loss=1.4941747188568115 Batch_id=97 Accuracy=30.52: 100%|██████████| 98/98 [00:27<00:00,  3.61it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.0028, Accuracy: 5062/10000 (50.62%)

EPOCH: 1

Loss=1.3062143325805664 Batch_id=97 Accuracy=46.88: 100%|██████████| 98/98 [00:26<00:00,  3.63it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.0023, Accuracy: 5920/10000 (59.20%)

EPOCH: 2

Loss=1.2733609676361084 Batch_id=97 Accuracy=53.90: 100%|██████████| 98/98 [00:26<00:00,  3.65it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.0022, Accuracy: 6068/10000 (60.68%)

EPOCH: 3

Loss=0.961903989315033 Batch_id=97 Accuracy=58.64: 100%|██████████| 98/98 [00:26<00:00,  3.63it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.0018, Accuracy: 6964/10000 (69.64%)

EPOCH: 4

Loss=1.0599894523620605 Batch_id=97 Accuracy=62.21: 100%|██████████| 98/98 [00:26<00:00,  3.65it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.0016, Accuracy: 7261/10000 (72.61%)

EPOCH: 5

Loss=0.9891334772109985 Batch_id=97 Accuracy=64.77: 100%|██████████| 98/98 [00:26<00:00,  3.64it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.0015, Accuracy: 7440/10000 (74.40%)

EPOCH: 6

Loss=0.8864437937736511 Batch_id=97 Accuracy=67.23: 100%|██████████| 98/98 [00:26<00:00,  3.66it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.0016, Accuracy: 7299/10000 (72.99%)

EPOCH: 7

Loss=0.8651148676872253 Batch_id=97 Accuracy=68.67: 100%|██████████| 98/98 [00:26<00:00,  3.67it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.0013, Accuracy: 7719/10000 (77.19%)

EPOCH: 8

Loss=0.7534120678901672 Batch_id=97 Accuracy=70.73: 100%|██████████| 98/98 [00:26<00:00,  3.67it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.0013, Accuracy: 7823/10000 (78.23%)

EPOCH: 9

Loss=0.8804699182510376 Batch_id=97 Accuracy=72.06: 100%|██████████| 98/98 [00:26<00:00,  3.67it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.0012, Accuracy: 7866/10000 (78.66%)

EPOCH: 10

Loss=0.7707906365394592 Batch_id=97 Accuracy=73.39: 100%|██████████| 98/98 [00:26<00:00,  3.71it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.0011, Accuracy: 8117/10000 (81.17%)

EPOCH: 11

Loss=0.7665610313415527 Batch_id=97 Accuracy=74.05: 100%|██████████| 98/98 [00:27<00:00,  3.62it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.0011, Accuracy: 8053/10000 (80.53%)

EPOCH: 12

Loss=0.6723630428314209 Batch_id=97 Accuracy=74.97: 100%|██████████| 98/98 [00:26<00:00,  3.66it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.0011, Accuracy: 8085/10000 (80.85%)

EPOCH: 13

Loss=0.619142472743988 Batch_id=97 Accuracy=75.46: 100%|██████████| 98/98 [00:26<00:00,  3.63it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.0010, Accuracy: 8341/10000 (83.41%)

EPOCH: 14

Loss=0.5685301423072815 Batch_id=97 Accuracy=76.73: 100%|██████████| 98/98 [00:26<00:00,  3.67it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.0010, Accuracy: 8403/10000 (84.03%)

EPOCH: 15

Loss=0.6351163387298584 Batch_id=97 Accuracy=77.47: 100%|██████████| 98/98 [00:26<00:00,  3.65it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.0010, Accuracy: 8246/10000 (82.46%)

EPOCH: 16

Loss=0.6599920988082886 Batch_id=97 Accuracy=78.08: 100%|██████████| 98/98 [00:26<00:00,  3.64it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.0010, Accuracy: 8412/10000 (84.12%)

EPOCH: 17

Loss=0.6329468488693237 Batch_id=97 Accuracy=78.76: 100%|██████████| 98/98 [00:27<00:00,  3.63it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.0010, Accuracy: 8300/10000 (83.00%)

EPOCH: 18

Loss=0.6096887588500977 Batch_id=97 Accuracy=79.18: 100%|██████████| 98/98 [00:26<00:00,  3.66it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.0010, Accuracy: 8331/10000 (83.31%)

EPOCH: 19

Loss=0.5970520973205566 Batch_id=97 Accuracy=79.17: 100%|██████████| 98/98 [00:27<00:00,  3.63it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.0009, Accuracy: 8474/10000 (84.74%)

EPOCH: 20

Loss=0.5074135065078735 Batch_id=97 Accuracy=79.86: 100%|██████████| 98/98 [00:26<00:00,  3.66it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.0009, Accuracy: 8462/10000 (84.62%)

EPOCH: 21

Loss=0.5541311502456665 Batch_id=97 Accuracy=80.39: 100%|██████████| 98/98 [00:26<00:00,  3.64it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.0009, Accuracy: 8524/10000 (85.24%)

EPOCH: 22

Loss=0.5840592980384827 Batch_id=97 Accuracy=81.19: 100%|██████████| 98/98 [00:27<00:00,  3.60it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.0009, Accuracy: 8612/10000 (86.12%)

EPOCH: 23

Loss=0.4878651201725006 Batch_id=97 Accuracy=81.82: 100%|██████████| 98/98 [00:26<00:00,  3.66it/s]

Test set: Average loss: 0.0009, Accuracy: 8543/10000 (85.43%)

## Plots

Accuracy and Error Plots:
  
![enter image description here](https://i.ibb.co/nsZxgyH/Unknown.png)

## Rightly Classified Images
![enter image description here](https://i.ibb.co/3cj7GFL/Unknown.png)

## Misclassified Images

![enter image description here](https://i.ibb.co/r2q2X7C/Unknown.png)


# Our Team
-   [Madhu Charan](https://github.com/madhucharan)
-   [Sijuade](https://github.com/cydal)
-   [Siddharth Aggarwal](https://github.com/aggarwalsiddharth)
-   [Deepika](https://github.com/dpkeee)


