# spaCy powered Label Studio ML backend

NER label tasks are only supported currently.

## Usage

1. Clone this repo

2. Install requirements

```
pip install -r requirements.txt
```

3. Initialize a new backend

```
label-studio-ml init my_ml_backend
```

4. In the `my_ml_backend` directory, add your spaCy `config.cfg` file. You can optionally add a `model-best` folder from a pre-trained model, to get started with predictions straight away.

5. Start the backend and add the URL to your Label Studio project settings.

```
label-studio-ml start my_ml_backend
```

6. As you train new models, they will appear in a `checkpoints` directory.