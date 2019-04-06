# libraries 

import argparse
import torch 
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from workspace_utils import active_session

def arg_parser():
  # parser arguments
  parser = argparse.ArgumentParser(description = 'Training script with pytorch with a dataset provided by the user, the possibility of choosing the architecture, setting the hyperparameters and using GPU for training') 
  parser.add_argument('data_directory', help='data directory',type=str)
  parser.add_argument('--save_dir', dest="save_dir", type=str,help='checkpoint directory', default='.')
  parser.add_argument('--arch', dest="arch", type=str,help='model architecture', default="vgg13")
  parser.add_argument('--learning_rate', dest="learning_rate", type=float,help='specify learning rate', default=0.001)
  parser.add_argument('--hidden_units', dest="hidden_units", type=int,help='specify learning rate', default=4096)
  parser.add_argument('--epochs', dest="epochs", type=int,help='specify epochs', default=3)
  parser.add_argument('--gpu', dest="gpu", action='store_true',help='gpu', default=False)
  args = parser.parse_args()
  return args

def load_model(model_name):
  """
  download the model from torchvision.models and returns the model
  """
  exec("model = models.{}(pretrained =True)".format(model_name), globals())
  print(model)
  #freeze the parameters
  for param in model.parameters():
    param.requires_grad = False

  return model   

def create_classifier(model, hidden_units, model_name):
  """
  create the classifier based on the provided number of hidden units
  return a classifier
  """
  if 'vgg' in model_name :
    input_features = model.classifier[0].in_features
  elif 'resnet' in model_name or 'inception' in model_name:
    input_features = model.fc.in_features
  elif 'densenet' in model_name:
    input_features = model.classifier.in_features
  else:
    input_features = model.classifier[1].in_channels

  classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_features, hidden_units,bias = True)), 
                          ('relu', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.4)),
                          ('fc2', nn.Linear(hidden_units, 102, bias = True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
  return classifier

# validation check 
def validation_accuracy(model, validationloader, criterion):
    
    valid_loss = 0
    accuracy = 0
    model.eval()
    
    for inputs, labels in validationloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)

        valid_loss += batch_loss.item()

        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    return valid_loss, accuracy

def train(model, epochs, device, trainloader, validationloader):
  """
  train the neural network and return the trained model
  """
  print('start the training')
  steps = 0
  print_every = 30 # print every 30 images
  train_loss = 0

  train_losses, valid_losses = [], []
  with active_session():
    # do long-running work here
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device) 
            
            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if steps % print_every == 0:
                with torch.no_grad():
                    valid_loss = 0 
                    valid_loss, accuracy = validation_accuracy(model, validationloader, criterion)
                    
                    train_losses.append(train_loss/print_every)
                    valid_losses.append(valid_loss/len(validationloader))

                print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {train_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validationloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validationloader):.3f}")
                    
            train_loss = 0
            model.train()
  print("Training Complete")   
  return model

def test_accuracy(model, testloader, criterion):
  print('measure the accuracy for the test set')
  with torch.no_grad():
    test_loss = 0
    accuracy = 0
    model.eval()

    for inputs, labels in testloader:
      inputs, labels = inputs.to(device), labels.to(device)
      logps = model.forward(inputs)
      batch_loss = criterion(logps, labels)

      test_loss += batch_loss.item()

      # Calculate accuracy
      ps = torch.exp(logps)
      top_p, top_class = ps.topk(1, dim=1)
      equals = top_class == labels.view(*top_class.shape)
      accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
  return test_loss, accuracy


def create_checkpoint(model, save_dir, train_data, model_name):
  print('create checkpoint')
  model.class_to_idx = train_data.class_to_idx
  checkpoint = {'model': model_name,
             'classifier': model.classifier,
             'class_to_idx':model.class_to_idx,
             'state_dict': model.state_dict()}
  torch.save(checkpoint, save_dir+'/checkpoint.pth')
             
# specify the train, validation, test data
args = arg_parser()
data_dir = args.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define transforms for the train data, validation and test_data

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, .225])])

validation_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# data with the required transformation
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
validation_data = datasets.ImageFolder(data_dir + '/valid', transform=validation_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

#loaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validationloader = torch.utils.data.DataLoader(validation_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

#label mapping
    
model = load_model(args.arch)
model.classifier = create_classifier(model, args.hidden_units, args.arch)

# use GPU if available 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if args.gpu else 'cpu'


# defining the parameters for the training , loss and the optimizer
criterion = nn.NLLLoss()

learning_rate = args.learning_rate

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
model.to(device);

epochs = args.epochs

trained_model = train(model, epochs, device, trainloader, validationloader)


## Testing your network
test_loss, accuracy = test_accuracy(trained_model, testloader, criterion)
print('Accuracy of the test images: {}'.format(accuracy/len(testloader)*100))

## Save the checkpoint
create_checkpoint(trained_model, args.save_dir, train_data, args.arch)
