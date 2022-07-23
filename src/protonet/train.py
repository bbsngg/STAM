import argparse
import os
import sys

import torch
from sentence_transformers import SentencesDataset, SentenceTransformer
from sentence_transformers import models
from torch.utils.data import DataLoader

sys.path.append(os.pardir)

from protonet.PrototypicalNetwork import PrototypicalNetwork
from utils.LabelAccuracyEvaluator2 import LabelAccuracyEvaluator
from utils.LabelSentenceReader import LabelSentenceReader
from utils.prototypical_batch_sampler import PrototypicalBatchSampler

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='set-up')
parser.add_argument('--bert-model', type=str, default='/path/to/models/bert-base-uncased')
parser.add_argument('--file-path', type=str, default=r'../../../data/')
parser.add_argument('--task-name', type=str, help='liu, clinc/mul_dom_dom, huff, snips')
parser.add_argument('--n-way', type=int, default=5)
parser.add_argument('--k-shot', type=int, default=5)
parser.add_argument('--max-seq-length', type=int, default=256, help='Max sequence length')
parser.add_argument('--loss-method', type=str, default='None', help='None KL SimCLR SimCLR+KL')
parser.add_argument('--contrast-mode', type=str, default='all')
parser.add_argument('--num-epochs', type=int, default=10)
parser.add_argument('--seed', type=int, default=2000)
parser.add_argument('--cuda-id', type=int, default=0)
parser.add_argument('--aug-method', type=str, default='bt', help='dropout, mix, bt, none is selected')
parser.add_argument('--contrast-weight', type=float, default=0.1, help='Contrastive loss weight')
parser.add_argument('--kl-weight', type=float, default=1, help='KL loss weight')
parser.add_argument('--train-epi', type=int, default=500, help='Train episode')
parser.add_argument('--dev-epi', type=int, default=100, help='Dev episode')
parser.add_argument('--test-epi', type=int, default=150, help='Test episode')
parser.add_argument('--eval-step', type=int, default=300, help='Evaluation steps')
parser.add_argument('--save-model', action='store_true', default=False, help='Save model checkpoint')
args = parser.parse_args()

print('{}\ndevice:{}\n'.format(vars(args), device))
torch.cuda.set_device(args.cuda_id)
# Loading from pre-trained-model file
model_name = args.bert_model
# Set seed
torch.manual_seed(args.seed)
print(args.file_path + args.task_name)
# Dataset Reader
fewjoint_reader = LabelSentenceReader(args.file_path + args.task_name, label_col_idx=0, aug_method=args.aug_method)
train_examples, train_labels = fewjoint_reader.get_examples('train_aug.in')
test_examples, test_labels = fewjoint_reader.get_examples('test_aug.in')
dev_examples, dev_labels = fewjoint_reader.get_examples('dev_aug.in')

# Sentence Bert
word_embedding_model = models.Transformer(model_name, max_seq_length=args.max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=False,
                               pooling_mode_cls_token=True,
                               pooling_mode_max_tokens=False)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


# init n-way  k-shot  episode
def init_sampler(labels, mode='train'):
    if 'train' in mode:
        classes_per_it = args.n_way  # n-way
        num_samples = args.k_shot * 2  # shot (support+query)
        episode = args.train_epi
    elif 'test' in mode:
        classes_per_it = args.n_way
        num_samples = args.k_shot * 2
        episode = args.test_epi
    else:
        classes_per_it = args.n_way
        num_samples = args.k_shot * 2
        episode = args.dev_epi

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=episode)  # episode


# train data
train_data = SentencesDataset(train_examples, model=model)
train_sampler = init_sampler(train_labels)
train_dataloader = DataLoader(train_data, batch_sampler=train_sampler)

# test data
test_data = SentencesDataset(test_examples, model=model)
test_sampler = init_sampler(test_labels, 'test')
test_dataloader = DataLoader(test_data, batch_sampler=test_sampler)

# dev data
dev_data = SentencesDataset(dev_examples, model=model)
dev_sampler = init_sampler(dev_labels, 'dev')
dev_dataloader = DataLoader(dev_data, batch_sampler=dev_sampler)

# prototypical network (Can be viewed as a LossFunction)
fsl_model = PrototypicalNetwork(model=model,
                                sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                n_support=args.k_shot,
                                contrast_mode=args.contrast_mode,
                                loss_method=args.loss_method,
                                contrast_weight=args.contrast_weight,
                                kl_weight=args.kl_weight,
                                )
# Setup
evaluator = LabelAccuracyEvaluator(dev_dataloader, softmax_model=fsl_model)
# Model ckpt path
model_save_path = None
if args.save_model:
    if not os.path.exists(f'/path/to/save/{args.task_name}'):
        os.makedirs(f'/path/to/save/{args.task_name}')
    model_save_path = f'/path/to/save/{args.task_name}/' \
                      f'ckpt-{args.task_name}-{args.contrast_mode}_{args.loss_method}' \
                      f'_N{args.n_way}_K{args.k_shot}' \
                      f'_s{args.seed}_ep{args.num_epochs}_base_{args.aug_method}' \
                      f'_cw{str(args.contrast_weight).replace(".", "")}'
# Train
model.fit(train_objectives=[(train_dataloader, fsl_model)],
          evaluator=evaluator,
          epochs=args.num_epochs,
          evaluation_steps=args.eval_step,
          output_path=model_save_path
          )
print(model.evaluate(evaluator))

# Test
evaluator = LabelAccuracyEvaluator(test_dataloader, softmax_model=fsl_model)
print(model.evaluate(evaluator))
print(vars(args))
