from sentence_transformers.evaluation import SentenceEvaluator
import torch
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from sentence_transformers.util import batch_to_device
import os
import csv
import json
import time

logger = logging.getLogger(__name__)


class LabelAccuracyEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on its accuracy on a labeled dataset
    This requires a model with LossFunction.SOFTMAX
    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(self,
                 dataloader: DataLoader,
                 name: str = "",
                 softmax_model=None,
                 write_csv: bool = True,
                 write_emb: bool = False):
        """
        Constructs an evaluator for the given dataset
        :param dataloader:
            the data for the evaluation
        """
        self.dataloader = dataloader
        self.name = name
        self.softmax_model = softmax_model

        if name:
            name = "_" + name

        self.write_csv = write_csv
        self.write_emb = write_emb
        self.csv_file = "accuracy_evaluation" + name + "_results.csv"
        self.emb_file = "embedding" + name + ".json"
        self.csv_headers = ["epoch", "steps", "accuracy"]

    def __call__(self,
                 model,
                 output_path: str = None,
                 epoch: int = -1,
                 steps: int = -1) -> float:
        model.eval()
        total = 0
        correct = 0
        emb = []
        lab = []

        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("Evaluation on the " + self.name + " dataset" + out_txt)
        print("Evaluation on the " + self.name + " dataset" + out_txt)
        self.dataloader.collate_fn = model.smart_batching_collate
        for step, batch in enumerate(self.dataloader):
            features, label_ids = batch
            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], model.device)
            label_ids = label_ids.to(model.device)
            with torch.no_grad():
                reps, prediction = self.softmax_model(features, labels=label_ids, mode='test')

            total += prediction.size(0)
            correct += prediction.sum().item()
            #correct += torch.argmax(prediction, dim=1).eq(label_ids).sum().item()

            # save embedding and label
            if self.write_emb:
                emb += reps.data.cpu().numpy().tolist()
                lab += label_ids.data.cpu().numpy().tolist()

        accuracy = correct / total
        print(f" ---> EvaluateTime={int(round(time.time()*1000))} Accuracy={accuracy}")  # NOTE: use for calculate time

        logger.info("Accuracy: {:.4f} ({}/{})\n".format(accuracy, correct, total))
        print("Accuracy: {:.4f} ({}/{})\n".format(accuracy, correct, total))

        if output_path is not None:
            # output result to .csv
            if self.write_csv:
                csv_path = os.path.join(output_path, self.csv_file)
                if not os.path.isfile(csv_path):
                    with open(csv_path, mode="w", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(self.csv_headers)
                        writer.writerow([epoch, steps, accuracy])
                else:
                    with open(csv_path, mode="a", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch, steps, accuracy])
            # output embedding to json
            if self.write_emb:
                emb_path = os.path.join(output_path, self.emb_file)
                if not os.path.isfile(emb_path):
                    with open(emb_path, mode="w", encoding="utf-8") as f:
                        json.dump({'embedding': emb, 'label': lab}, f, indent=2)
                else:
                    with open(emb_path, mode="a", encoding="utf-8") as f:
                        json.dump({'embedding': emb, 'label': lab}, f, indent=2)
        return accuracy
