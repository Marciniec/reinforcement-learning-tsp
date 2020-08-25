import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from train.PointerNet import PointerNet


class ModelLoader:
    def __init__(self, path):
        self.path = path
        self.model_batch_size = 256
        self.model = None

    def load_model(self):
        model = torch.load(self.path)
        model.eval()
        self.model = model

    def evaluate(self, tsp_dataset):
        data_loader = DataLoader(tsp_dataset,
                                 batch_size=self.model_batch_size,
                                 shuffle=True,
                                 num_workers=1)
        for dev_batch_idx, dev_batch in enumerate(data_loader):
            batch = Variable(dev_batch['Points'])
            target_batch = Variable(dev_batch['Solution'])
            batch = batch.cuda()
            o, p = self.model(batch)
            yield list(zip(batch, target_batch, o, p))
