from training.train import TrainModel
from evaluate.evaluate_model import EvaluateModel
from scoring.score import ScoreModel


tm = TrainModel()
ah_transforms, data_dict = tm.split_data()
model = tm.train_model(data_dict)
tm.save_model_metrics()
tm.save_model(model)

em = EvaluateModel()
em.evaluate(data_dict)

scr = ScoreModel()
scr.score(ah_transforms, tm.get_data())

