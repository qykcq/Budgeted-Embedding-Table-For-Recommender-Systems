
dataset = initialize_dataset(with_validation=True)
LightGCN(dataset, user_sizes, item_sizes).to(config.device)
eval_rec(recsys, dataset, 1)
