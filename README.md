# spaCy powered Label Studio ML backend

spaCy integration for Label Studio. Created as an Open Source alternative to Prodigy.

Benefits:

* Speed up annotation of data with integrated predictions
* Quickly iterate on your spaCy models

## Demo video

[![Demo video](https://img.youtube.com/vi/F19NT-21uT4/0.jpg)](https://youtu.be/F19NT-21uT4)

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

6. As you train new models, they will appear in a `checkpoints` directory. The latest checkpoint will be symlinked to `latest-model`.

