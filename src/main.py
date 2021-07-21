import json

from evaluate.evaluate_model import EvaluateModel
from scoring.score import ScoreModel
from training.train import TrainModel


def main():
    # TODO: Create a factory class to make training, evaluation and scoring independent.
    #  Due to time constraints, I couldnot add this functionality !
    #   These files can then be run individually and parameters can be captured using *args and **kwargs

    # Load applicaton parameters from parameters.json
    params = json.load(open('parameters.json'))

    # Initialize training
    tm = TrainModel(params['training'], system=params['system'])
    ah_transforms, data_dict = tm.split_data()
    model = tm.train_model(data_dict)
    tm.save_model_metrics()
    tm.save_model(model)

    # Initialize Evaluation
    em = EvaluateModel(system=params['system'])
    em.evaluate(data_dict)

    # Score Model
    scr = ScoreModel(system=params['system'])
    scr.score(ah_transforms, tm.get_data())


if __name__ == "__main__":
    main()
