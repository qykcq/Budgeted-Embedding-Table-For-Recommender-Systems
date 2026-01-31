from module import config
import os
import time
from util.train_util import logtxt
from module.PopulationEvaluator import PopulationEvaluator


IDs = {
    'gowalla': {
        0: 0.2
    }
}
config.BASE_MODEL = 'lightgcn'
config.NUM_SAMPLES = 100


# use this to evaluate the actions
def eval_population():
    for dataset_name in IDs:
        config.DATASET_NAME = dataset_name
        budgets = IDs[dataset_name]
        config.MAX_PATIENCE = 3
        for ID in budgets:
            while True:
                folder = 'tmp/population/{}/'.format(ID)
                exist = os.path.exists(folder)
                if not exist:
                    time.sleep(10)
                else:
                    break

            logtxt('=' * 90)
            config.ID = ID
            config.BUDGET = budgets[ID]
            logtxt('ID = {}; BUDGET = {}'.format(config.ID, config.BUDGET))
            # steps = list(range(0, 50, 1))
            steps = list(range(49, -1, -1))

            pe = PopulationEvaluator()
            pe.eval_population(
                steps=[0],
                k=3  # select the top k=3
            )


eval_population()
