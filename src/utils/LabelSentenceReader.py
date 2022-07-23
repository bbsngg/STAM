from sentence_transformers.readers import InputExample
import csv
import gzip
import os


class LabelSentenceReader:
    """Reads in a file that has at least two columns: a label and a sentence.
    This reader can for example be used with the BatchHardTripletLoss.
    Maps labels automatically to integers"""

    def __init__(self, folder, label_col_idx=0, sentence_col_idx=1, separator='\t', aug_method='eda'):
        self.folder = folder
        self.label_map = {}
        self.label_col_idx = label_col_idx
        self.sentence_col_idx = sentence_col_idx
        self.separator = separator
        self.aug_method = aug_method

    def get_examples(self, filename, max_examples=0):
        examples = []
        labels = []
        id = 0
        for line in open(os.path.join(self.folder, filename), encoding="utf-8"):
            splits = line.strip().split(self.separator)
            label = splits[self.label_col_idx]
            sentence = splits[self.sentence_col_idx]

            if self.aug_method in ['eda']:
                context_aug_sentence = splits[2]
                random_aug_sentence = splits[3]
                back_translation_aug = splits[4]
                texts = [sentence, context_aug_sentence, random_aug_sentence, back_translation_aug]
            elif self.aug_method in ['dropout']:
                texts = [sentence, sentence]
            elif self.aug_method in ['mix']:
                back_translation_aug = splits[4]
                texts = [sentence, sentence, back_translation_aug, back_translation_aug]
            elif self.aug_method in ['bt']:
                back_translation_aug = splits[4]
                texts = [sentence, back_translation_aug]
            elif self.aug_method in ['none']:
                texts = [sentence]
            else:
                texts = []

            if label not in self.label_map:
                self.label_map[label] = len(self.label_map)

            label_id = self.label_map[label]
            guid = "%s-%d" % (filename, id)
            id += 1
            examples.append(InputExample(guid=guid, texts=texts, label=label_id))

            if 0 < max_examples <= id:
                break

            labels.append(label_id)

        return examples, labels
