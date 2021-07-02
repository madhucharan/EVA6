

# ASSIGNMENT

Assignment:

1.  Check this Repo out:  [https://github.com/kuangliu/pytorch-cifar (Links to an external site.)](https://github.com/kuangliu/pytorch-cifar)
2.  You are going to follow the same structure for your Code from now on. So Create:
    1.  models folder - this is where you'll add all of your future models. Copy resnet.py into this folder, this file should only have ResNet 18/34 models.  **Delete Bottleneck Class**
    2.  main.py - from Google Colab, now onwards, this is the file that you'll import (along with model). Your main file shall be able to take these params or you should be able to pull functions from it and then perform operations, like (including but not limited to):  
        1.  training and test loops
        2.  data splits between test and train
        3.  epochs
        4.  batch size
        5.  which optimizer to run
        6.  do we run a scheduler?
    3.  utils.py file (or a folder later on when it expands) - this is where you will add all of your utilities like:
        1.  image transforms,
        2.  gradcam,
        3.  misclassification code,
        4.  tensorboard related stuff
        5.  advanced training policies, etc
        6.  etc
    4.  Name this main repos something, and don't call it Assignment8. This is what you'll import for all the rest of the assignments. Add a proper readme describing all the files.
3.  Your assignment is to build the above training structure. Train ResNet18 on Cifar10 for 20 Epochs. The assignment must:
    1.  pull your Github code to google colab (don't copy-paste code)
    2.  prove that you are following the above structure
    3.  that the code in your google collab notebook is NOTHING.. barely anything. There should not be any function or class that you can define in your Google Colab Notebook. Everything must be imported from all of your other files
    4.  your colab file must:
        1.  train resnet18 for 20 epochs on the CIFAR10 dataset
        2.  show loss curves for test and train datasets
        3.  show a gallery of 10 misclassified images
        4.  show gradcam output on 10 misclassified images.  **Remember if you are applying GradCAM on a channel that is less than 5px, then please don't bother to submit the assignment. 😡🤬🤬🤬🤬**
    5.  Once done, upload the code to GitHub, and share the code. This readme must link to the main repo so we can read your file structure.

### Project files

> ! git clone https://github.com/cydal/Pytorch_CIFAR10_gradcam.git

### CIFAR10 Images

![enter image description here](https://i.postimg.cc/s21tVWXL/image.png)


----------Training Model----------
EPOCH: 0
Loss=1.7113145589828491 Batch_id=390 Accuracy=30.26: 100%|██████████| 391/391 [01:30<00:00,  4.31it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0118, Accuracy: 4501/10000 (45.01%)

EPOCH: 1
Loss=1.5079586505889893 Batch_id=390 Accuracy=41.37: 100%|██████████| 391/391 [01:30<00:00,  4.34it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0100, Accuracy: 5495/10000 (54.95%)

EPOCH: 2
Loss=1.480538010597229 Batch_id=390 Accuracy=47.44: 100%|██████████| 391/391 [01:30<00:00,  4.34it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0091, Accuracy: 6055/10000 (60.55%)

EPOCH: 3
Loss=1.299292802810669 Batch_id=390 Accuracy=51.42: 100%|██████████| 391/391 [01:30<00:00,  4.34it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0078, Accuracy: 6533/10000 (65.33%)

EPOCH: 4
Loss=1.5847713947296143 Batch_id=390 Accuracy=54.78: 100%|██████████| 391/391 [01:30<00:00,  4.34it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0068, Accuracy: 6960/10000 (69.60%)

EPOCH: 5
Loss=1.2168904542922974 Batch_id=390 Accuracy=56.91: 100%|██████████| 391/391 [01:30<00:00,  4.34it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0081, Accuracy: 6574/10000 (65.74%)

EPOCH: 6
Loss=1.0546923875808716 Batch_id=390 Accuracy=59.20: 100%|██████████| 391/391 [01:30<00:00,  4.34it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0060, Accuracy: 7336/10000 (73.36%)

EPOCH: 7
Loss=1.3188886642456055 Batch_id=390 Accuracy=60.93: 100%|██████████| 391/391 [01:29<00:00,  4.35it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0052, Accuracy: 7681/10000 (76.81%)

EPOCH: 8
Loss=1.01651930809021 Batch_id=390 Accuracy=62.23: 100%|██████████| 391/391 [01:29<00:00,  4.35it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0055, Accuracy: 7698/10000 (76.98%)

EPOCH: 9
Loss=0.8788882493972778 Batch_id=390 Accuracy=63.02: 100%|██████████| 391/391 [01:30<00:00,  4.34it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0063, Accuracy: 7461/10000 (74.61%)

EPOCH: 10
Loss=0.9900673627853394 Batch_id=390 Accuracy=64.27: 100%|██████████| 391/391 [01:29<00:00,  4.35it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0050, Accuracy: 7830/10000 (78.30%)

EPOCH: 11
Loss=1.0271011590957642 Batch_id=390 Accuracy=65.03: 100%|██████████| 391/391 [01:30<00:00,  4.34it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0056, Accuracy: 7672/10000 (76.72%)

EPOCH: 12
Loss=0.905640721321106 Batch_id=390 Accuracy=66.05: 100%|██████████| 391/391 [01:30<00:00,  4.34it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0048, Accuracy: 7993/10000 (79.93%)

EPOCH: 13
Loss=0.7253869771957397 Batch_id=390 Accuracy=67.00: 100%|██████████| 391/391 [01:30<00:00,  4.33it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0043, Accuracy: 8135/10000 (81.35%)

EPOCH: 14
Loss=1.0565019845962524 Batch_id=390 Accuracy=67.53: 100%|██████████| 391/391 [01:30<00:00,  4.34it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0046, Accuracy: 8061/10000 (80.61%)

EPOCH: 15
Loss=0.9869354367256165 Batch_id=390 Accuracy=68.66: 100%|██████████| 391/391 [01:29<00:00,  4.35it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0042, Accuracy: 8228/10000 (82.28%)

EPOCH: 16
Loss=0.7849057912826538 Batch_id=390 Accuracy=69.06: 100%|██████████| 391/391 [01:29<00:00,  4.35it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0041, Accuracy: 8247/10000 (82.47%)

EPOCH: 17
Loss=0.8045971989631653 Batch_id=390 Accuracy=69.23: 100%|██████████| 391/391 [01:29<00:00,  4.35it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0042, Accuracy: 8287/10000 (82.87%)

EPOCH: 18
Loss=0.6713516116142273 Batch_id=390 Accuracy=69.98: 100%|██████████| 391/391 [01:29<00:00,  4.35it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0041, Accuracy: 8291/10000 (82.91%)

EPOCH: 19
Loss=0.8335630297660828 Batch_id=390 Accuracy=70.68: 100%|██████████| 391/391 [01:29<00:00,  4.35it/s]

Test set: Average loss: 0.0037, Accuracy: 8520/10000 (85.20%)


![enter image description here](https://i.postimg.cc/y87KzW0q/image.png)


### Misclassified Images - T(True class) - P(Predicted)

![enter image description here](https://i.postimg.cc/s21tVWXL/image.png)


#### Correctly Classified

![enter image description here](https://i.postimg.cc/TwMQZ5Wg/image.png)


Misclassified Classes Heatmap
![enter image description here](https://i.postimg.cc/vmW5xqcx/image.png)


### Our Team
- [Madhu Charan](https://github.com/madhucharan)
- [Sijuade](https://github.com/cydal)
- [Siddharth Aggarwal](https://github.com/aggarwalsiddharth)
- [Deepika](https://github.com/dpkeee)
---

> Written with [StackEdit](https://stackedit.io/).
