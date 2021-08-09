#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, shutil
from os.path import join as joinpath
from os import listdir


# In[2]:


# Expected input format: an `images` folder with all images in flat structure, i.e. as direct children.
datadir = "/mnt/beegfs/home/vu/Codalab-MetaDL/data/resisc45"
imagedir = joinpath(datadir, "images")
resultdir = "/mnt/beegfs/home/vu/Codalab-MetaDL/result"

# path to metadata file with at least 2 columns: filename and label
metadata_filepath = joinpath(datadir, "resisc45-filtered-metadata.csv")
# names of these 2 columns
labelcolumn = "category"
filecolumn = "filename"

# result are saved in this folder inside the resultdir
dataname = "mini_resisc"
# prefix for output file
resultprefix = "mini_resisc-reptile-nopretrain"

random_seed = 2021


# In[3]:


figdir = joinpath(resultdir, dataname, resultprefix, 'fig')
modeldir = joinpath(resultdir, dataname, resultprefix, 'model')
metricdir = joinpath(resultdir, dataname, resultprefix, 'metric')

for outputdir in (figdir, modeldir, metricdir):
    os.makedirs(outputdir, exist_ok=True)


# In[4]:


# import os, shutil
# from os.path import join as joinpath

from pathlib import Path

import sys, copy
import itertools, math

from functools import partial
import json

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# from tqdm import tqdm_notebook as tqdm
# from tqdm.autonotebook import tqdm
# from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics

import PIL

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import torch.autograd as autograd

import captum.attr
import scipy


# In[5]:


pd.options.mode.chained_assignment = None 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

pltparams = {
    'legend.fontsize': 'x-large',
    'axes.labelsize': 'x-large',
    'axes.titlesize': 'x-large',
    'xtick.labelsize': 'x-large',
    'ytick.labelsize': 'x-large',
    'figure.titlesize': 'x-large',
    'savefig.dpi': 600,
}
plt.rcParams.update(pltparams)

sns.set(font_scale = 1.2)


# ## Load data

# In[6]:


filtered_metadata = pd.read_csv(metadata_filepath, index_col=0)
filtered_frequency = filtered_metadata.value_counts(labelcolumn)
filtered_metadata.sample(5)


# In[7]:


plt.figure(figsize=(12, 4))
ax = filtered_frequency.plot(kind="bar")
ax.set_xticks([])
plt.suptitle(dataname)

# ax.figure.savefig(joinpath(figdir, f'{dataname}-distribution.png'), bbox_inches='tight')


# In[8]:


def tensor_to_display(imagetensor):
    return imagetensor.numpy().transpose((1, 2, 0))

def show_images(metadata, imagedir, rows=2, columns=5, figsize=(16, 8), title=None):
    sns.set_style("dark")

    fig, axes = plt.subplots(rows, columns, figsize=figsize)
    axes = axes.flatten()
    
    for index, ax in enumerate(axes):
        imageinfo = metadata.iloc[index]
        image = PIL.Image.open(joinpath(imagedir, imageinfo[filecolumn]))
        width, height = image.size
#         print(width,height)
        ax.imshow(image)
        ax.set_title(imageinfo[labelcolumn])
#         plt.axis('off')

    if title is not None:
        plt.suptitle(title)
    
    return fig

fig = show_images(filtered_metadata.sample(10), imagedir, title=dataname)

# fig.savefig(joinpath(basedir, 'fig', f'{dataname}-sample-image.png'))


# ## Few shot setting

# In[9]:


def meta_train_test_split(frequency, test_fraction=0.25, randomseed=None):
    """
    Shuffle the labels, then split
    """
    np.random.seed(randomseed)
    shuffled_index = np.random.permutation(frequency.index)
    
    split_index = int((1 - test_fraction) * len(frequency))
    metatrain_classes = shuffled_index[:split_index]
    metatest_classes = shuffled_index[split_index:]
    
    return metatrain_classes, metatest_classes

metatrain_classes, metatest_classes = meta_train_test_split(filtered_frequency, randomseed=random_seed)
# print(metatrain_classes, metatest_classes)
# print(len(metatrain_classes), len(metatest_classes))


# In[10]:


def train_valid_split(metadata, kshot, randomseed):
    """
    support and query split for each label
    """
    
    traindata = metadata.groupby(labelcolumn).sample(kshot, random_state=randomseed)
    traindata['partition'] = "train"

    valid_index = metadata.index.difference(traindata.index)
    validdata = metadata.loc[valid_index, :]
    validdata['partition'] = "valid"

    return pd.concat((traindata, validdata))

# train_valid_split(filtered_metadata, 20, random_seed)


# In[11]:


def split_label_per_episode(all_classes, nway):
    episode_count = len(all_classes) // nway
    episode_classes = np.split(all_classes[:episode_count*nway], episode_count)
    
    return episode_classes

def support_query_generator(metadata, all_classes, nway, kshot):
    shuffled_classes = np.random.permutation(all_classes)    
    for episode_classes in split_label_per_episode(shuffled_classes, nway):
        support, query = [],[]

        support = metadata[(metadata[labelcolumn].isin(episode_classes)) & (metadata.partition == "train")]
        query = metadata[(metadata[labelcolumn].isin(episode_classes)) & (metadata.partition == "valid")]

        labelcode = LabelEncoder()
        query["labelcode"] = labelcode.fit_transform(query[labelcolumn])
        support["labelcode"] = labelcode.transform(support[labelcolumn])

        yield support, query, labelcode


# In[12]:


def train_transform(pil_image, imagesize):
    width, height = pil_image.size
    transform = transforms.Compose([
        transforms.CenterCrop(min(width, height)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.Resize(imagesize),
        transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(pil_image)

def test_transform(pil_image, imagesize):
    transform = transforms.Compose([
        transforms.Resize(imagesize),
        transforms.ToTensor(),
    ])
    return transform(pil_image)


# In[13]:


class FewShotDataset(torch.utils.data.Dataset):
    def __init__(self, metadata, imagedir, transform=None):
        self.metadata = metadata
        self.imagedir = imagedir
        self.transform = transform
    
    def __getitem__(self, index):
        imageinfo = self.metadata.iloc[index]
        
        imagedata = PIL.Image.open(os.path.join(self.imagedir, imageinfo[filecolumn])) 
        transformed_image = self.transform(imagedata) if self.transform else imagedata
        label = imageinfo["labelcode"]
#         print(transformed_image.shape)

        return transformed_image, label

    def __len__(self):
        return len(self.metadata)
    
def build_data(metadata, imagedir, batchsize, ifshuffle, transform=None):
    dataset = FewShotDataset(metadata, imagedir, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=ifshuffle)
    return dataset, dataloader


# In[14]:


def preprocess_metadata(metadata, random_seed=None):
    processed_metadata, labelcode = encode_label(metadata)
    train_metadata, valid_metadata = train_test_split(processed_metadata, test_size=0.5, stratify=processed_metadata[labelcolumn], random_state=random_seed)
#     print(train_metadata.shape, valid_metadata.shape)

    numberclass = valid_metadata["labelcode"].max() + 1
    labelnames = labelcode.inverse_transform(range(numberclass))
    
    return train_metadata, valid_metadata, labelcode, numberclass, labelnames


# In[15]:


def display_tensor(dataloader, labelcode):
    inputs, labels = next(iter(dataloader))
    print(inputs.shape, labels.shape)
    for inpt, label in zip(inputs[:4], labels[:4]):
        plt.figure()
        plt.imshow(tensor_to_display(inpt))
        plt.title(labelcode.inverse_transform([label]))


# In[16]:


imagesize=(128, 128)
transform0 = partial(train_transform, imagesize=(imagesize))

support, query, labelcode = next(support_query_generator(train_valid_split(filtered_metadata, 20, random_seed),
    metatrain_classes, nway=5, kshot=20))
dataset, dataloader = build_data(query, imagedir, 5, True, transform0)

display_tensor(dataloader, labelcode)


# ## Model

# In[17]:


def get_model_resnet_dropout(outsize, with_pretrained=True, with_dropout=True):
    model = models.resnet18(pretrained=with_pretrained)

    if with_dropout:
        drop_proba = dict(layer4=0.1, layer3=0.1)

        for layer, sublayer, module in itertools.product(drop_proba.keys(), [0, 1], "conv1 conv2".split()):
            modulename = f"""model.{layer}[{sublayer}].{module}"""
            command = f"""{modulename} = nn.Sequential(nn.Dropout({drop_proba[layer]}), {modulename})"""
    #         print(command)
            exec(command)

        for layer, proba in drop_proba.items():
            command = f"""conv_outchannels = model.{layer}[0].conv2[1].out_channels"""
            exec(command)
            modulename = f"""model.{layer}[0].bn2"""
            command = f"""{modulename} = nn.Sequential(nn.Dropout(proba), {modulename})"""
            exec(command)    

    conv_outsize = model.fc.in_features
    model.fc = nn.Linear(conv_outsize, outsize)

    return model.to(device)

def clone_model(model):
    return copy.deepcopy(model).to(device)   


# ## Define training

# In[18]:


def get_optim_param_dict_resnet(model, lrrange):
    """
    Geometric progression of learning rate per layer block
    """
    paramgroups = {}
    namegroup = {}
    
    for name, param in model.named_parameters():
#         print(name)
        groupname = '.'.join(name.split('.')[:2])
        
        if groupname.startswith("fc"):
            groupname = "fc"
        
        layergroup = namegroup.setdefault(groupname, [])
        layergroup.append(name)
        namegroup[groupname] = layergroup
    
        paramgroup = paramgroups.setdefault(groupname, [])
        paramgroup.append(param)
        paramgroups[groupname] = paramgroup
    
    lrgroup = np.geomspace(*lrrange, num=len(paramgroups))
    optim_param = [dict(params=group, lr=lr, weight_decay=0.1) for group, lr in zip(paramgroups.values(), lrgroup)]

    return optim_param

def get_lrrange_optimizer(model, lrrange, get_paramdict_func):
    optim = torch.optim.SGD(get_paramdict_func(model, lrrange), lr=lrrange[-1])
#     task_optim = torch.optim.Adam(task_model.parameters(), lr=params['inner_lr'], betas=(0, 0))
    
    return optim


# In[19]:


def init_task_model_optim(meta_model, params):
    task_model = clone_model(meta_model).train()
#     task_optim = torch.optim.Adam(task_model.parameters(), lr=params['inner_lr'], betas=(0, 0))
    task_optim = torch.optim.SGD(task_model.parameters(), lr=params['inner_lr'], weight_decay=0.1)
#     task_optim = get_lrrange_optimizer(task_model, (1e-6, params['inner_lr']), get_optim_param_dict_resnet)

    return task_model, task_optim


# In[20]:


def update_model(lossfunc, optim, model, xbatch, ybatch):
    optim.zero_grad()
    loss = lossfunc(model(xbatch.to(device)), ybatch.to(device))
    loss.backward()
    optim.step()
    
    return loss

def compute_task_gradient(meta_model, task_model):
    # record weights
    for oldparams, newparams in zip(meta_model.parameters(), task_model.parameters()):
        if oldparams.grad is None:
            oldparams.grad = autograd.Variable(torch.zeros_like(oldparams)).to(device)
        # invert loss eqn. to use descent optimization
        oldparams.grad.data.add_(oldparams.data - newparams.data)
#         oldparams.grad.data.add_(newparams.grad.data)


# In[21]:


def score_predictions(predictions, targets):
    score = sklearn.metrics.accuracy_score(targets, predictions)
    return score

def one_batch(model, inputs, labels, lossfunc, device, optim=None):
    if optim: optim.zero_grad()
        
    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = model(inputs)
    loss = lossfunc(outputs, labels)
    
    predictions = torch.argmax(outputs, dim=1)  

    if optim:
        loss.backward()
        optim.step()

    return loss, predictions, labels


# In[22]:



def init_result():
    result = dict(loss=0., predictions=[], labels=[], samplecount=0.)
    return result

def update_result(result, loss, predictions, labels):
    result['loss'] += loss.item()
    result['predictions'].append(predictions)
    result['labels'].append(labels)
    result['samplecount'] += len(labels)
    
    return result

def get_metric(result):
    result['predictions'] = torch.cat(result['predictions'])
    result['labels'] = torch.cat(result['labels'])
    
    metric = dict(
        loss = result['loss'] / result['samplecount'],
        score = score_predictions(result['predictions'].detach().cpu(), result['labels'].detach().cpu())
    )
    
    return metric


# In[23]:


# class BatchIterator():
#     def __init__(self, metadata, imagedir, batchsize, ifshuffle, transform):
#         self.metadata, self.imagedir, self.batchsize, self.ifshuffle, self.transform = metadata, imagedir, batchsize, ifshuffle, transform
#         self.dataset, self.dataloader = build_data(
#             self.metadata, self.imagedir, self.batchsize, self.ifshuffle, self.transform
#         )
        
#     def next_batch(self):
#         try:
#             return next(iter(self.dataloader))
        
#         except StopIteration as exception:
#             self.dataset, self.dataloader = build_data(
#                 self.metadata, self.imagedir, self.batchsize, self.ifshuffle, self.transform
#             )
#             print('Exception, rebuild dataloader')
#             return next(iter(self.dataloader))


# In[24]:


def train_evaluate_episode(params, meta_model, lossfunc, support_metadata, query_metadata, 
    imagedir, train_transform, test_transform):
    
    task_model, task_optim = init_task_model_optim(meta_model, params)
    
    # train on batches of the dataset
    stepcount = 0
    support_result = init_result()
#     support_batch_iterator = BatchIterator(support_metadata, imagedir, 
#         batchsize=params["inner_batchsize"], ifshuffle=True, transform=train_transform)
    support_dataset, support_dataloader = build_data(support_metadata, imagedir, 
        batchsize=params["inner_batchsize"], ifshuffle=True, transform=train_transform)
    
    while (stepcount < params["update_count"]) :
#         xbatch, ybatch = support_batch_iterator.next_batch()
        xbatch, ybatch = next(iter(support_dataloader))
        loss, predictions, labels = one_batch(task_model, xbatch, ybatch, lossfunc, device, optim=task_optim)
        support_result = update_result(support_result, loss, predictions, labels)
        stepcount += 1
        if stepcount == params["update_count"]: 
#             print(f"{stepcount = }")
            break

    support_metric = get_metric(support_result)

    # evaluate with episode query
    query_result = init_result()
    query_dataset, query_dataloader = build_data(query_metadata, imagedir, 
        batchsize=params["inner_batchsize"], ifshuffle=False, transform=test_transform)
    
    for (xbatch, ybatch) in (query_dataloader):
        loss, predictions, labels = one_batch(task_model, xbatch, ybatch, lossfunc, device, optim=None)
        query_result = update_result(query_result, loss, predictions, labels)
    
    query_metric = get_metric(query_result)

    return task_model, support_metric, query_metric


"""
def train_evaluate_episode(params, meta_model, lossfunc, support_metadata, query_metadata, 
    imagedir, train_transform, test_transform):
    
    task_model, task_optim = init_task_model_optim(meta_model, params)
    
    # train on batches of the dataset
    stepcount = 0
    support_result = init_result()
    support_dataset, support_dataloader = build_data(support_metadata, imagedir, 
        batchsize=params["inner_batchsize"], ifshuffle=True, transform=train_transform)
    
    while (stepcount < params["update_count"]) :
        for (xbatch, ybatch) in (support_dataloader):
            loss, predictions, labels = one_batch(task_model, xbatch, ybatch, lossfunc, device, optim=task_optim)
            support_result = update_result(support_result, loss, predictions, labels)
            stepcount += 1
            if stepcount == params["update_count"]: break

    support_metric = get_metric(support_result)

    # evaluate with episode query
    query_result = init_result()
    query_dataset, query_dataloader = build_data(query_metadata, imagedir, 
        batchsize=params["inner_batchsize"], ifshuffle=False, transform=test_transform)
    
    for (xbatch, ybatch) in (query_dataloader):
        loss, predictions, labels = one_batch(task_model, xbatch, ybatch, lossfunc, device, optim=None)
        query_result = update_result(query_result, loss, predictions, labels)
    
    query_metric = get_metric(query_result)

    return task_model, support_metric, query_metric
"""


# In[25]:


def init_assessment_record(filepath, nway):
    if os.path.exists(filepath): os.remove(filepath)
    headers = "assess_iteration, " +     ", ".join([f"label_{i}" for i in range(1, nway+1)]) +     ", supportloss, supportscore, queryloss, queryscore" +     "\n"
    
    with open(filepath, 'a') as file:
        file.write(headers)
    
    return
        
def record_assessment(task_labelcode, assess_iteration, support_metric, query_metric, filepath):
#     assess_record = ", ".join(task_labelcode.classes_) + f", {metric['score']:.4f}, {assess_iteration}" + "\n"
    assess_record = f"{assess_iteration}, " +     ", ".join(task_labelcode.classes_) +     f", {support_metric['loss']}, {support_metric['score']}" +     f", {query_metric['loss']}, {query_metric['score']}" +     "\n"

    with open(filepath, 'a') as file:
        file.write(assess_record)

    return 


# In[26]:


def check_duplicate_task(task_labelcode, indexed_tasks, labeldict, duplicate_count):
    task_indexes = [str(labeldict[label]) for label in task_labelcode.classes_]
    task = ' '.join(sorted(task_indexes))
    is_duplicate = True
    
    if task in indexed_tasks:
        duplicate_count += 1
    else:
        indexed_tasks.add(task)
        is_duplicate = False
    
    return is_duplicate, indexed_tasks, duplicate_count


# In[27]:


def one_meta_iteration(metadata, all_classes, params, meta_model, lossfunc, imagedir,
    train_transform, test_transform, assess_iteration, assess_path, indexed_tasks, is_learning):
    
    task_count = 1; duplicate_count = 0
    labeldict = {label:index for index, label in enumerate(all_classes)}

    # inner train loop, learn task
    max_taskcount = params["learn_task_per_iteration"] if is_learning else params["assess_task_per_iteration"]    
    episode_generator = support_query_generator(metadata, all_classes,
        nway=params["nway"], kshot=params["kshot"])    
    
    while (task_count <= max_taskcount):
        try:
            support_metadata, query_metadata, task_labelcode = next(episode_generator)
            is_duplicate, indexed_tasks, duplicate_count = check_duplicate_task(task_labelcode, 
                indexed_tasks, labeldict, duplicate_count)
            
            # too many duplicate failure, too few classes
            if duplicate_count > 2 * max_taskcount: break
            if is_duplicate: continue
                  
            task_model, support_metric, query_metric = train_evaluate_episode(params, meta_model, lossfunc, 
                support_metadata, query_metadata, imagedir, train_transform, test_transform)
            
            if is_learning:
                compute_task_gradient(meta_model, task_model)
            else:
                print("support", support_metric, "\nquery", query_metric, '\n')
                record_assessment(task_labelcode, assess_iteration, support_metric, query_metric, assess_path)
            
            task_count += 1

        except StopIteration as exception:
            episode_generator = support_query_generator(metadata, all_classes,
                nway=params["nway"], kshot=params["kshot"])

    return


# In[28]:


def updata_meta_model(meta_model, meta_optim, params):
    for metaparams in meta_model.parameters():
        metaparams.grad.data.div_(params['learn_task_per_iteration'])
        meta_optim.step()
    
    return


# In[29]:


#https://github.com/gebob19/REPTILE-Metalearning/blob/master/omniglot_trainer.ipynb

def train_reptile(params, meta_optim, meta_model, lossfunc, metatrain_classes, metatest_classes,
    metadata, imagedir, train_transform, test_transform, assess_path, metamodel_path):  
    
    init_assessment_record(assess_path, params['nway'])
    assess_iteration = 0
    learn_indexed_tasks = set(); assess_indexed_tasks = set()

    # outer meta loop
    for meta_iteration in range(1, params['meta_iteration']+1):

        meta_optim.zero_grad()
        
        one_meta_iteration(metadata, metatrain_classes, params, meta_model, lossfunc, imagedir,
            train_transform, test_transform, assess_iteration, assess_path, learn_indexed_tasks, is_learning=True)
                
        # update metamodel with avg over mini batches 
#         for metaparams in meta_model.parameters():
#             metaparams.grad.data.div_(params['learn_task_per_iteration'])
#         meta_optim.step()
        updata_meta_model(meta_model, meta_optim, params)
            
        # metatest validation episodes
        assess_records = [];
        
        if (meta_iteration % params['assess_frequency']) == 0 :
            print(f"\n{meta_iteration = }", '=' * 20)
            assess_iteration += 1
            
            one_meta_iteration(metadata, metatest_classes, params, meta_model, lossfunc, imagedir,
                train_transform, test_transform, assess_iteration, assess_path, assess_indexed_tasks, is_learning=False)
        
        torch.save(meta_model, metamodel_path)
    # end meta epoch loop
    return


# ## Run Training

# In[44]:


way5_shot20_params = {
    'nway': 5,
    'kshot': 20,
    # inner loop parameters
    'inner_lr': 1e-4,
    'inner_batchsize': 128,
    'update_count': 10,
    # outter loop parameters
    'meta_lr': 1e-2,
    'meta_iteration': 100,
    'learn_task_per_iteration': 4,
    # assessment
    'assess_frequency': 10,
    'assess_task_per_iteration': 4,
}

params = way5_shot20_params


# In[45]:


metatrain_classes, metatest_classes = meta_train_test_split(filtered_frequency)
print(len(metatrain_classes), len(metatest_classes))
split_metadata = train_valid_split(filtered_metadata, params['kshot'], random_seed)

imagesize=(128, 128)
train_transform0 = partial(train_transform, imagesize=imagesize)
test_transform0 = partial(test_transform, imagesize=imagesize)


# In[46]:


meta_model = get_model_resnet_dropout(params['nway'], with_pretrained=False, with_dropout=False)
meta_optim = torch.optim.SGD(meta_model.parameters(), lr=params['meta_lr'])
lossfunc = nn.CrossEntropyLoss(reduction='sum')

model_name = resultprefix + '-resnet18-way{}-shot{}'.format(params['nway'], params['kshot'])
assess_path = joinpath(metricdir, resultprefix + "-assessment.csv")         
metamodel_path = joinpath(modeldir, model_name + ".pth")


# In[47]:


train_reptile(params, meta_optim, meta_model, lossfunc, metatrain_classes, metatest_classes, 
    split_metadata, imagedir, train_transform0, test_transform0, assess_path, metamodel_path)


# ## Metric summary

# In[48]:


# assess_path = joinpath(metricdir, resultprefix + "-assessment.csv")         

assessment_metric = pd.read_csv(assess_path)
assessment_metric


# In[49]:


summary_assessment_metric = assessment_metric.groupby('assess_iteration').agg([np.mean, np.std])
summary_assessment_metric


# In[50]:


print(summary_assessment_metric)


# In[ ]:




