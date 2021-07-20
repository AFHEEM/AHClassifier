import time

import torch

from model.model import DeepModel


class EvaluateModel:
    """
    Evaulate the model
    """

    def __init__(self):
        # Run on CPU
        self.device = 'cpu'

    def evaluate(self, dataloaders_dict):
        """
        Evaluate the model
        :return:
        """
        model = DeepModel()
        model.load_state_dict(torch.load("model/pickle/model.pth"))
        since = time.time()
        model.eval()  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for batch_index, (inputs, labels) in enumerate(dataloaders_dict['test']):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # forward
            # track history if only in train
            with torch.set_grad_enabled(False):
                # Get model outputs and calculate loss
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

            # statistics
            running_corrects += torch.sum(preds == labels.data)

        epoch_acc = running_corrects.double() / len(dataloaders_dict['test'].dataset)

        print('Test Acc: {:.4f}'.format(epoch_acc))
        print()

        time_elapsed = time.time() - since
        print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
