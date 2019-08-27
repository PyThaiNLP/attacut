import re
import torch
from typing import List

from attacut import utils, models, dataloaders, preprocessing, artifacts, logger
from typing import Dict

log = logger.get_logger(__name__)

class Tokenizer:
    def __init__(self, model: str="attacut-sc"):
        # resolve model's path
        model_path = artifacts.get_path(model)

        params = utils.load_training_params(model_path)

        model_name = params.name
        log.info("loading model %s" % model_name)

        model_cls: models.BaseModel = models.get_model(model_name)

        # instantiate dataset
        dataset: dataloaders.SequenceDataset = model_cls.dataset()

        # load necessary dicts into memory
        data_config: Dict  = dataset.setup_featurizer(model_path)

        # instantiate model
        self.model = model_cls.load(
            model_path,
            data_config,
            params.params
        )

        self.dataset = dataset

    def tokenize(self, txt:str, sep="|", device="cpu", pred_threshold=0.5) -> List[str]:
        tokens, features = self.dataset.make_feature(txt)

        inputs = (
            features,
            torch.Tensor(0) # dummy label when won't need it here
        )

        x, _, _ = self.dataset.prepare_model_inputs(inputs, device=device)
        logits = torch.sigmoid(self.model(x))

        preds = logits.cpu().detach().numpy() > pred_threshold

        words = preprocessing.find_words_from_preds(tokens, preds)

        return words


    # todo: tokenize_batch(self, generator):
    #     ...
