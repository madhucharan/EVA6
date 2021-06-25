
import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, OneCycleLR
import torch.nn.functional as F
from tqdm import tqdm


train_losses = []
test_losses = []
train_acc = []
test_acc = []




class depthwise_separable_conv(nn.Module):
 def __init__(self, nin, kernels_per_layer, nout): 
   super(depthwise_separable_conv, self).__init__() 
   
   self.depthwise = nn.Sequential(
          nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin, bias=False), 
          nn.BatchNorm2d(nin * kernels_per_layer),
          nn.ReLU()
   )

   self.pointwise = nn.Sequential(
          nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1, padding=1, bias=False),
          nn.BatchNorm2d(nout),
          nn.ReLU()
   )
  
 def forward(self, x): 
   out = self.depthwise(x) 
   out = self.pointwise(out) 
   return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            depthwise_separable_conv(32, 1, 32), 
            depthwise_separable_conv(32, 1, 32), 
            depthwise_separable_conv(32, 1, 32), 
            #nn.Dropout(dropout_value)
        )

        self.transblock1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=2, bias=False, dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            depthwise_separable_conv(64, 1, 64), 
            depthwise_separable_conv(64, 1, 64), 
            depthwise_separable_conv(64, 1, 64), 
        )

        self.transblock2 = nn.Sequential( 
            nn.Conv2d(64, 16, 3, stride=2, bias=False, dilation=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            depthwise_separable_conv(32, 1, 32), 
            depthwise_separable_conv(32, 1, 32), 
            depthwise_separable_conv(32, 1, 32), 
        )

        self.transblock3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=2, bias=False, dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            depthwise_separable_conv(32, 1, 32), 

            nn.Conv2d(32, 42, 3, bias=False),
            nn.BatchNorm2d(42),
            nn.ReLU(),


            nn.AvgPool2d(4),
            nn.Conv2d(42, 32, 1, bias=False),
            nn.Conv2d(32, 10, 1, bias=False)
            
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.transblock1(x)

        x = self.block2(x)
        x = self.transblock2(x)

        x = self.block3(x)
        x = self.transblock3(x)

        x = self.block4(x)
        x = x.view(-1, 10)
        x = F.log_softmax(x, dim=-1)
        return x


def train(model, device, train_loader, optimizer, epoch):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.nll_loss(y_pred, target)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm
    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_acc.append(100. * correct / len(test_loader.dataset))



def train_model(model, criterion, device, train_loader, test_loader, optimizer, scheduler, EPOCHS):
  """
  Args:
      model (torch.nn Model): Original data with no preprocessing
      criterion (criterion) - Loss Function
      device (str): cuda/CPU
      train_loader (DataLoader) - DataLoader Object
      optimizer (optimizer) - Optimizer Object
      scheduler (scheduler) - scheduler object
      EPOCHS (int) - Number of epochs
  Returns:
      results (list): Train/test - Accuracy/Loss 
  """
  for epoch in range(EPOCHS):
      print("EPOCH:", epoch)
      train(model, device, train_loader, optimizer, epoch)
      scheduler.step()
      test(model, device, test_loader)

  results = [train_losses, test_losses, train_acc, test_acc]
  return(results)