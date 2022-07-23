from cProfile import label
import logging
import os
import random
from datetime import datetime
from pathlib import Path

import spacy
from label_studio_ml.model import LabelStudioMLBase
from spacy.cli.train import train
from spacy.tokens import DocBin, Doc

# Constants

# Map `from_name` to spacy model layers
LABEL_CONFIG = {
    'ner': [],
    'spancat': [],
    'textcat': []
}

# GPU ID's to use. -1 means use the CPU
TRAIN_GPU_ID = -1
PREDICTION_GPU_ID = -1

# Fraction of data to use for evaluation
EVAL_SPLIT = 0.2

# Batch size for predictions
PREDICTION_BATCH_SIZE = 16

# Score threshold for a category to be accepted
TEXTCAT_SCORE_THRESHOLD = 0.5

# Multiple categories per doc?
TEXTCAT_MULTI = False

# END constants

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SpacyModel(LabelStudioMLBase):
    TRAIN_EVENTS = ()

    def __init__(self, **kwargs):
        super(SpacyModel, self).__init__(**kwargs)

        self.model = self.load()
        self.model_version = self.train_output['checkpoint'] if 'checkpoint' in self.train_output else 'fallback'

        logger.info("MODEL CHECKPOINT: %s", self.model_version)

    def ner_labels(self):
        return label_dict_from_config(self.parsed_label_config, LABEL_CONFIG['ner'])

    def spancat_labels(self):
        return label_dict_from_config(self.parsed_label_config, LABEL_CONFIG['spancat'])

    def textcat_labels(self):
        return label_dict_from_config(self.parsed_label_config, LABEL_CONFIG['textcat'])

    def load(self):
        model_dir = os.path.dirname(os.path.realpath(__file__))
        fallback_dir = os.path.join(model_dir, "model-best")

        if PREDICTION_GPU_ID > -1:
            spacy.prefer_gpu(gpu_id=PREDICTION_GPU_ID)

        if 'model_path' in self.train_output and os.path.isdir(self.train_output['model_path']):
            return spacy.load(self.train_output['model_path'])
        elif os.path.isdir(fallback_dir):
            return spacy.load(fallback_dir)

        return None

    def predict(self, tasks, **kwargs):
        """ This is where inference happens: model returns 
            the list of predictions based on input list of tasks 
        """
        if not self.model:
            logger.error("model has not been trained yet")
            return []

        ner_labels = self.ner_labels()
        spancat_labels = self.spancat_labels()
        textcat_labels = self.textcat_labels()
        predictions = []

        docs = self.model.pipe([t['data']['text']
                               for t in tasks], batch_size=PREDICTION_BATCH_SIZE)
        for doc in docs:
            results = []

            for e in doc.ents:
                config = ner_labels[e.label_]
                results.append({
                    'from_name': config['from_name'],
                    'to_name': config['to_name'],
                    'type': 'labels',
                    'value': {
                        'start': e.start_char,
                        'end': e.end_char,
                        'text': e.text,
                        'labels': [e.label_]
                    }
                })

            for from_name, span_group in doc.spans:
                for span in span_group:
                    to_name = spancat_labels[span.label_]['to_name']
                    results.append({
                        'from_name': from_name,
                        'to_name': to_name,
                        'type': 'labels',
                        'value': {
                            'start': span.start_char,
                            'end': span.end_char,
                            'text': span.text,
                            'labels': [span.label_]
                        }
                    })

            choices = [choice for choice, score in doc.cats.items()
                       if score >= TEXTCAT_SCORE_THRESHOLD]
            if len(choices) > 0:
                config = textcat_labels[choices[0]]
                results.append({
                    'from_name': config['from_name'],
                    'to_name': config['to_name'],
                    'type': 'choices',
                    'value': {
                        'choices': choices
                    }
                })

            predictions.append({
                'model_version': self.model_version,
                'result': results
            })

        return predictions

    def fit(self, annotations, workdir=None, **kwargs):
        """ This is where training happens: train your model given list of annotations, 
            then returns dict with created links and resources
        """
        model_dir = os.path.dirname(os.path.realpath(__file__))
        checkpoint_name = datetime.now().strftime("%Y%m%d%H%M%S")
        checkpoint_dir = os.path.join(
            model_dir, 'checkpoints', checkpoint_name)
        config_path = os.path.join(model_dir, 'config.cfg')

        train_data_path = os.path.join(checkpoint_dir, 'train.spacy')
        dev_data_path = os.path.join(checkpoint_dir, 'dev.spacy')
        model_path = os.path.join(checkpoint_dir, 'model-best')
        latest_path = os.path.join(model_dir, "latest-model")
        latest_path_tmp = os.path.join(model_dir, "latest-model-tmp")

        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        annotations = list(filter(item_not_cancelled, list(annotations)))

        train_data, dev_data = split_annotations(annotations, EVAL_SPLIT)

        annotations_to_docbin(
            train_data,
            ner_labels=self.ner_labels(),
            spancat_labels=self.spancat_labels(),
            textcat_labels=self.textcat_labels()
        ).to_disk(train_data_path)

        annotations_to_docbin(
            dev_data,
            ner_labels=self.ner_labels(),
            spancat_labels=self.spancat_labels(),
            textcat_labels=self.textcat_labels()
        ).to_disk(dev_data_path)

        train(config_path, checkpoint_dir, use_gpu=TRAIN_GPU_ID, overrides={
              'paths.train': train_data_path, 'paths.dev': dev_data_path})

        os.symlink(model_path, latest_path_tmp)
        os.replace(latest_path_tmp, latest_path)

        return {'model_path': model_path, 'checkpoint': checkpoint_name}

# Helper functions


def label_dict_from_config(config, from_names: list[str]):
    map = {}

    for from_name in from_names:
        schema = config[from_name]
        to_name = schema['to_name'][0]
        labels = schema['labels']

        for label in labels:
            map[label] = {
                'from_name': from_name,
                'to_name': to_name
            }

    return map


def item_not_cancelled(item):
    return item['annotations'][0]['was_cancelled'] != True


def split_annotations(annotations, split):
    random.shuffle(annotations)

    dev_len = round(len(annotations) * split)
    train_data = annotations[dev_len:]
    dev_data = annotations[:dev_len]

    return train_data, dev_data


def annotations_to_docbin(annotations, ner_labels, spancat_labels, textcat_labels):
    nlp = spacy.blank("en")
    db = DocBin()

    docs = []
    for item in annotations:
        if not item['data']['text']:
            continue

        doc = nlp(item['data']['text'])
        annotation = item['annotations'][0]

        for a in annotation['result']:
            if a['type'] == 'labels':
                add_span_to_doc(
                    doc,
                    item,
                    a,
                    ner_labels=ner_labels,
                    spancat_labels=spancat_labels
                )
            elif a['type'] == 'choices':
                add_cat_to_doc(doc, a, textcat_labels)

        docs.append(doc)

    for doc in docs:
        if TEXTCAT_MULTI == True or textcat_labels or doc_has_one_cat(doc):
            db.add(doc)

    return db


def add_span_to_doc(doc: Doc, annotation, ner_labels, spancat_labels):
    val = annotation['value']
    label = val['labels'][0]

    if label not in ner_labels or label not in spancat_labels:
        return

    span = doc.char_span(
        val['start'], val['end'], label=label, alignment_mode='expand')

    if span and label in ner_labels:
        doc.ents = doc.ents + (span,)

    elif span and label in spancat_labels:
        from_name = spancat_labels[label]['from_name']
        if from_name in doc.spans:
            doc.spans[from_name].append(span)
        else:
            doc.spans[from_name] = [span]


def add_cat_to_doc(doc: Doc, annotation, label_dict):
    val = annotation['value']
    selected = val['choices']

    for choice in label_dict.keys():
        doc.cats[choice] = choice in selected


def doc_has_one_cat(doc: Doc):
    positive_cats = [cat for cat, val in doc.cats.items() if val == True]
    return len(positive_cats) == 1
