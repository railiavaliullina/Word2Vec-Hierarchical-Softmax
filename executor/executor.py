from trainer import Trainer
from configs.dataset_config import cfg as dataset_cfg


class Executor(object):
    """
    Class for running main class methods which run whole algorithm.
    """
    @staticmethod
    def run():
        trainer = Trainer(dataset_cfg)


if __name__ == '__main__':
    executor = Executor()
    executor.run()
