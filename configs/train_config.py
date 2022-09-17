from easydict import EasyDict


cfg = EasyDict()

cfg.evaluate_before_training = True
cfg.lr = 0.025
cfg.fin_lr = 0.0001
cfg.epochs = 1
cfg.hidden_layer_size = 200
