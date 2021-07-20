import torch

from model.model import DeepModel
from util.utility import SingleImage


class ScoreModel:
    """
    Score model based on test data
    """

    def __init__(self):
        # Run on CPU
        self.device = 'cpu'

    def score(self, ah_transforms, data, path='model/pickle/model.pth'):
        image_path = 'scoring/score_test/test_f.jpg'  # './test_nf.jpg'
        single_image_set = SingleImage(image_path, ah_transforms)
        data_loader_single_image = torch.utils.data.DataLoader(single_image_set, batch_size=1)

        for inputs in data_loader_single_image:
            inputs = inputs.to(self.device)
            with torch.set_grad_enabled(False):
                # Get model outputs and calculate loss
                model = DeepModel()
                model.load_state_dict(torch.load("model/pickle/model.pth"))
                model.eval()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                print("image", image_path, "belongs to class", data.vocab[preds.item()])
                #         print(inputs.squeeze().size(), inputs.squeeze().permute(1, 2, 0))
                from IPython.display import Image
                print(Image(filename=image_path))
