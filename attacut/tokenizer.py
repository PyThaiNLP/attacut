import re
import torch

from attacut import utils, models, dataloaders, preprocessing, artifacts
from typing import Dict


class Tokenizer:
    def __init__(self, model: str):
        # resolve model's directory
        model_path = artifacts.get_path(model)

        params: Dict = utils.load_training_params(model_path)

        model_name = params["model_name"]
        print("loading model %s" % model_name)

        model_cls: models.BaseModel = models.get_model(model_name)

        # instantiate dataset
        dataset: dataloaders.SequenceDataset = model_cls.dataset()

        # load necessary dicts into memory
        data_config: Dict  = dataset.setup_featurizer(model_path)

        # instantiate model
        self.model = model_cls.load(
            model_path,
            data_config,
            params["model_params"] # architecture of the model
        )

        self.dataset = dataset

    def tokenize(self, txt:str, sep="|", device="cpu", pred_threshold=0.5):
        tokens, features = self.dataset.make_feature(txt)
        inputs = (
            features,
            torch.Tensor(0) # dummy label when won't need it here
        )
        x, _, _ = self.dataset.prepare_model_inputs(inputs, device=device)
        preds = torch.sigmoid(self.model(x)).cpu().numpy() > pred_threshold

        words = preprocessing.find_words_from_preds(tokens, preds)

        return sep.join(words)


    # todo: tokenize_batch(self, generator):
    #     ...
