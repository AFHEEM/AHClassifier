import copy
import time
import torch
from fastai.vision.all import ImageDataLoaders
from matplotlib import pyplot as plt
from torch import nn
from torch import optim
from torchvision import transforms, datasets

from model.model import DeepModel


class TrainModel:
    """
    Train model
    """
    def __init__(self, *args, **kwargs):
        """
        Initiate variables
        """
        # Batch size for training (change depending on how much memory you have)
        self.batch_size = args[0]['batch_size']

        # Number of epochs to train for
        self.num_epochs = args[0]['num_epochs']

        # Run on CPU
        self.device = kwargs['system']['device']

        self.losses_save_path = args[0]['losses_save_path']
        self.accuracies_save_path = args[0]['accuracies_save_path']
        self.data_location = args[0]['data_location']
        self.seed = args[0]['seed']

        # TODO: Remove class variables which are not part of this class. Refactoring needed.
        self.val_loss_history = []
        self.val_acc_history = []
        self.train_loss_history = []
        self.train_acc_history = []
        self.best_model_wts = None
        self.data = None

    def split_data(self):
        # Data augmentation and normalization for training
        # Just normalization for validation
        ah_transforms = transforms.Compose([
            transforms.Resize(80),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        print("Initializing Datasets and Dataloaders...")

        # Create training and validation datasets
        image_datasets = {
            "train": datasets.ImageFolder(f"{self.data_location}/train", ah_transforms),
            "test": datasets.ImageFolder(f"{self.data_location}/test", ah_transforms),
        }
        dataloaders_dict = {
            "train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=self.batch_size, shuffle=True),
            "test": torch.utils.data.DataLoader(image_datasets["test"], batch_size=self.batch_size, shuffle=True),
        }

        self.data = ImageDataLoaders.from_folder(self.data_location, train='train', valid='test', bs=self.batch_size, num_workers=4, seed=self.seed)
        classes = self.data.vocab
        print("Classes:", classes)
        print("Dimensions:", self.get_dimensions(dataloaders_dict), "(batch x channels x height x width)")
        self.data.show_batch()
        print("Initializing Datasets and Dataloaders...")
        return ah_transforms, dataloaders_dict

    def train_model(self, dataloaders_dict):
        """
        Train model
        :param :
        :return:
        """
        # TODO: Break this function into smaller functions to reduce cognitive complexity
        deep_model = DeepModel()

        # Print the model we just instantiated
        print(deep_model)

        # Observe that all parameters are being optimized
        deep_optimizer = optim.Adam(deep_model.parameters(), lr=1e-4)
        deep_criterion = nn.CrossEntropyLoss()
        since = time.time()

        self.best_model_wts = copy.deepcopy(deep_model.state_dict())
        best_acc = 0.0
        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch+1, self.num_epochs))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'test']:
                if phase == 'train':
                    deep_model.train()  # Set model to training mode
                else:
                    deep_model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                total_batches = len(dataloaders_dict[phase])
                for batch_index, (inputs, labels) in enumerate(dataloaders_dict[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    deep_optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        outputs = deep_model(inputs)
                        loss = deep_criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            deep_optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders_dict[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders_dict[phase].dataset)

                # statistics
                if phase == "train":
                    self.train_loss_history.append(epoch_loss)
                    self.train_acc_history.append(epoch_acc)
                else:
                    self.val_loss_history.append(epoch_loss)
                    self.val_acc_history.append(epoch_acc)
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    self.best_model_wts = copy.deepcopy(deep_model.state_dict())
            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        return deep_model

    def save_model_metrics(self):
        """
        Save model metrics
        :return:
        """
        deep_history = (self.val_loss_history, self.val_acc_history, self.train_loss_history, self.train_acc_history)
        self.save_loss(self.losses_save_path, deep_history)
        self.save_accuracy(self.accuracies_save_path, deep_history)

    def save_loss(self, location, deep_history):
        """
        Save losses.jpg
        :param location: Location to the path to save the picture
        :param deep_history: Deep learning model information
        :return:
        """
        plt.figure(figsize=(16,9))
        plt.title("losses")
        plt.plot(deep_history[0], '-o', label="val loss")
        plt.plot(deep_history[2], '-o', label="train loss")
        plt.legend()
        plt.savefig(location)

    def save_accuracy(self, location, deep_history):
        """
        Save accuracy.jpg file
        :param location:
        :param deep_history:
        :return:
        """
        plt.figure(figsize=(16,9))
        plt.title("accuracy")
        plt.plot(deep_history[1], '-o', label="val acc")
        plt.plot(deep_history[3], '-o', label="train acc")
        plt.legend()
        plt.savefig(location)

    def save_model(self, deep_model, path='model/pickle/model.pth'):
        """
        Save model
        :return:
        """
        # load best model weights
        deep_model.load_state_dict(
            self.best_model_wts)
        torch.save(deep_model.state_dict(), path)

    def get_dimensions(self, dataloaders_dict):
        for (data, target) in dataloaders_dict['train']:
            return list(data.size())

    def get_data(self):
        """
        Get preprocessed data for image classifier
        :return:
        """
        return self.data
