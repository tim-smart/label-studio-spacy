import logging
import os
import random
from datetime import datetime
from pathlib import Path

import spacy
from label_studio_ml.model import LabelStudioMLBase
from spacy.cli.train import train
from spacy.tokens import DocBin

logger = logging.getLogger(__name__)


def item_not_cancelled(item):
    return item['annotations'][0]['was_cancelled'] != True


def split_annotations(annotations, split=0.15):
    random.shuffle(annotations)

    dev_len = round(len(annotations) * split)
    train_data = annotations[dev_len:]
    dev_data = annotations[:dev_len]

    return train_data, dev_data


def annotations_to_docbin(annotations, valid_labels: list[str]):
    nlp = spacy.blank("en")
    db = DocBin()

    for item in annotations:
        if not item['data']['text']:
            continue

        text = item['data']['text']
        annotation = item['annotations'][0]
        doc = nlp(text)
        ents = []

        for a in annotation['result']:
            if a['type'] != 'labels':
                continue

            val = a['value']
            label = val['labels'][0]

            if label not in valid_labels:
                continue

            span = doc.char_span(val['start'], val['end'], label=label)
            if span:
                ents.append(span)
            else:
                print("BAD SPAN", text[val['start']:val['end']])

        doc.ents = ents
        db.add(doc)

    return db


class SpacyModel(LabelStudioMLBase):
    TRAIN_EVENTS = ()

    def __init__(self, **kwargs):
        super(SpacyModel, self).__init__(**kwargs)

        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema['to_name'][0]
        self.labels = schema['labels']
        self.model = self.latest_model()
        self.model_version = self.train_output['checkpoint'] if 'checkpoint' in self.train_output else 'fallback'

    def latest_model(self):
        model_dir = os.path.dirname(os.path.realpath(__file__))
        fallback_dir = os.path.join(model_dir, "model-best")

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
            return {}

        predictions = []

        for task in tasks:
            doc = self.model(task['data']['text'])

            results = []
            for e in doc.ents:
                results.append({
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'type': 'labels',
                    'value': {
                        'start': e.start_char,
                        'end': e.end_char,
                        'text': e.text,
                        'labels': [e.label_]
                    }
                })

            predictions.append({
                'model_version': self.model_version,
                'result': results
            })

        return predictions

    def fit(self, annotations, workdir=None, dev_split=0.15, **kwargs):
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

        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        annotations = list(filter(item_not_cancelled, list(annotations)))

        train_data, dev_data = split_annotations(annotations, dev_split)
        annotations_to_docbin(train_data, self.labels).to_disk(train_data_path)
        annotations_to_docbin(dev_data, self.labels).to_disk(dev_data_path)

        print(train_data_path, dev_data_path)
        train(config_path, checkpoint_dir, overrides={
              'paths.train': train_data_path, 'paths.dev': dev_data_path})

        return {'model_path': model_path, 'checkpoint': checkpoint_name}
