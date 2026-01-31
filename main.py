from module import config
from util.train_util import initialize_dataset, train_till_convergence
from util.eval_util import eval_rec
from base_recsys.LightGCN import LightGCN


dataset = initialize_dataset(with_validation=True)
LightGCN(dataset, user_sizes, item_sizes).to(config.device)
eval_rec(recsys, dataset, 1)
