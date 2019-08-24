#!/usr/bin/env python

import os
import glob
import sys
import json
import shutil

from collections import defaultdict

import numpy as np
import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

from attacut import models, utils, dataloaders

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def _create_metrics(metrics=["true_pos", "false_pos", "false_neg"]):

    return dict(zip(metrics, [0]*len(metrics)))

def accumuate_metrics(m1, m2):
    for k, v in m1.items():
        m1[k] = v + m2[k]
    return m1

def evaluate_model(logits, labels):
    labels = labels.cpu().detach().numpy()
    preds = torch.sigmoid(logits).cpu().detach().numpy() > 0.5

    return {
        "true_pos": np.sum(preds * labels),
        "false_pos": np.sum((1-preds) * labels),
        "false_neg": np.sum(preds * (1-labels))
    }

def precision_recall(true_pos, false_pos, false_neg):
    precision = true_pos/(true_pos+false_pos)
    recall = true_pos/(true_pos+false_neg)
    f1 = 2*precision*recall / (precision+recall)

    return precision, recall, f1

def prepare_embedding_matrix(gensim_w2v):
    
    shape = gensim_w2v.wv.vectors.shape 
    print("Our embedding is size %dx%d" % (shape[0], shape[1]))
    embedding = np.zeros((shape[0]+1, shape[1]))
    embedding[1:, :] = gensim_w2v.wv.vectors
    print("Final dims", embedding.shape)
    return embedding

def load_data(data_path, filenames=("x-training", "x-val", "y-training", "y-val")):
    data = dict()

    for n in filenames:
        data[n] = np.load("%s/%s.npy" % (data_path, n))

    return data

def print_floydhub_metrics(metrics, step=0, prefix=""):
    if 'FLOYDHUB' in os.environ and os.environ['FLOYDHUB']:
        for k, v in metrics.items():
            print('{"metric": "%s:%s", "value": %f, "step": %d}' % (k, prefix, v, step))

def tensor_to_device(tensor, device):
    if "torch" in str(type(tensor)):
        return tensor.to(device)
    return tensor

def do_iterate(model, generator, device,
    optimizer=None, criterion=None, prefix="", step=0):

        total_loss, total_preds = 0, 0
        metrics = _create_metrics()

        for i, inputs in enumerate(generator):
            xd, yd, total_batch_preds = generator.dataset.prepare_model_inputs(
                inputs, device
            )

            if optimizer:
                model.zero_grad()

            logits = model(xd).view(-1)
            loss = criterion(logits, yd)

            if optimizer:
                loss.backward()
                optimizer.step()

            total_preds += total_batch_preds
            total_loss += loss.item() * total_batch_preds

            accumuate_metrics(metrics, evaluate_model(logits, yd))

        avg_loss = total_loss / total_preds 
        pc_values = precision_recall(**metrics)
        print("[%s] loss %f | precision %f | recall %f | f1 %f" % (
            prefix,
            avg_loss,
            *pc_values
        ))

        print_floydhub_metrics(
            dict(
                loss=avg_loss,
                precision=pc_values[0],
                recall=pc_values[1],
                f1=pc_values[2]
            ),
            step=step, prefix=prefix
        )

# taken from https://stackoverflow.com/questions/52660985/pytorch-how-to-get-learning-rate-during-training
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

""" main
character_dict only used for SyllableCharacterSeq models
"""
def main(
        model_name, data_dir, syllable_dict, character_dict="",
        epoch=10, lr=0.001, batch_size=64, weight_decay=0.0, checkpoint=5,
        model_params="", output_dir="", no_workers=4, lr_schedule="",
        prev_model=""
    ):

    with open("%s/config.json" % data_dir, "r") as f:
        data_config = json.load(f)

    if character_dict:
        ch2idx = utils.load_dict(character_dict)
        data_config["num_char_tokens"] = len(ch2idx)

    device = get_device()
    print("Using device: %s" % device)

    sy2idx = utils.load_dict(syllable_dict)
    data_config['num_tokens'] = len(sy2idx)

    print(data_config)

    params = {}

    if model_params:
        params['model_config'] = model_params
        print(">> model configuration: %s" % model_params)
    
    if prev_model:
        print("Initiate model from %s" % prev_model)
        model = models.get_model(model_name).load(
            prev_model,
            data_config,
            **params
        )
    else:
        model = models.get_model(model_name)(
            data_config,
            **params
        )
        
    model = model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if prev_model:
        print("Loading prev optmizer's state")
        optimizer.load_state_dict(torch.load("%s/optimizer.pth" % prev_model))
        print("Previous learning rate", get_lr(optimizer))

        # force torch to use the given lr, not previous one
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            param_group['initial_lr'] = lr

        print("Current learning rate", get_lr(optimizer))

    if lr_schedule:
        schedule_params = utils.parse_model_params(lr_schedule)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=schedule_params['step'],
            gamma=schedule_params['gamma'],
        )
    
    ds_class = getattr(dataloaders, data_config['dataset'])
    params = {
        'batch_size': batch_size,
        'num_workers': no_workers,
    }

    if hasattr(ds_class, "collate_fn"):
        print("using collate_fn", ds_class)
        params["collate_fn"] = ds_class.collate_fn

    print("Using dataset: %s" % data_config['dataset'])

    training_set = ds_class("%s/%s" % (data_dir, "training.txt"))
    training_generator = data.DataLoader(training_set, shuffle=True, **params)

    validation_set = ds_class("%s/%s" % (data_dir, "val.txt"))
    validation_generator = data.DataLoader(validation_set, shuffle=False, **params)

    total_train_size = len(training_set) 
    total_test_size = len(validation_set)

    print("We have %d train samples and %d test samples" %
        (total_train_size, total_test_size)
    )

    print(
        '{"metric": "%s:%s", "value": %s}' %
        ("model", model_name, model.total_trainable_params())
    )

    utils.maybe_create_dir(output_dir)
    utils.save_training_params(
        output_dir,
        dict(
            **data_config,
            lr=lr,
            batch_size=batch_size,
            model_name=model_name,
            model_params=model.model_params
        )
    )

    shutil.copy(syllable_dict, "%s/%s" % (output_dir, syllable_dict.split("/")[-1]))

    if character_dict:
        shutil.copy(character_dict, "%s/%s" % (output_dir, character_dict.split("/")[-1]))

    for e in range(epoch):
        e = e + 1

        print("===EPOCH %d ===" % (e))
        if lr_schedule:
            curr_lr = get_lr(optimizer)
            print_floydhub_metrics(dict(lr=curr_lr), step=e, prefix="global")
            print("lr: ", curr_lr)

        with utils.Timer("epoch-training") as timer:
            do_iterate(model, training_generator,
                prefix="training",
                step=e,
                device=device,
                optimizer=optimizer,
                criterion=criterion,
            )

        with utils.Timer("epoch-validation") as timer, \
            torch.no_grad():
            do_iterate(model, validation_generator,
                prefix="validation",
                step=e,
                device=device,
                criterion=criterion,
            )

        if lr_schedule:
            scheduler.step()

        if checkpoint and e % checkpoint == 0:
            model_path = "%s/model-e-%d.pth" % (output_dir, e)
            print("Saving model to %s" % model_path)
            torch.save(model.state_dict(), model_path)


    model_path = "%s/model.pth" % output_dir
    opt_path = "%s/optimizer.pth" % output_dir


    print("Saving model to %s" % model_path)
    torch.save(model.state_dict(), model_path)
    torch.save(optimizer.state_dict(), opt_path)

if __name__ == "__main__":
    fire.Fire(main)