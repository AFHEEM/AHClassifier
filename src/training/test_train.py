import unittest
from unittest.mock import patch

from train import TrainModel


class TestTrainModel(unittest.TestCase):
    """
    Class to unit test TrainModel class
    """

    def setUp(self) -> None:
        self.tm = TrainModel({
            "batch_size": 16,
            "trans_resize": 80,
            "num_epochs": 100,
            "losses_save_path": "model/metrics/losses.jpg",
            "accuracies_save_path": "model/metrics/accuracies.jpg",
            "data_location": "../data",
            "seed": 0.2}, system={"device": "cpu"})

    def test_save_model_metrics(self):
        """
        Test Model output
        :return:
        """
        with patch("train.TrainModel.save_loss") as save_loss, patch("train.TrainModel.save_accuracy") as save_accuracy:
            self.tm.save_model_metrics()
            assert save_loss.call_count == 1
            assert save_accuracy.call_count == 1
