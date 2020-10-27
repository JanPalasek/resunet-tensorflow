# resunet-tensorflow
Implements [Deep Residual U-Net](https://arxiv.org/pdf/1711.10684.pdf) for semantic segmentation using Tensorflow 2.0.

## How to install model to your own project
Run following script from your target python virtual environment

```shell
python -m pip install git+https://github.com/JanPalasek/resunet-tensorflow
```

## How to train the model in your own project
After installing to your own project, you can import and instantiate the model. The model instance is a [standard Tensorflow 2.0 model](https://www.tensorflow.org/api_docs/python/tf/keras/Model), so you can use it accordingly.

```python
from resunet.model import ResUNet

# create model for inputs of sizes (128, 128, 1) for semantic segmentation into 2 classes
# architecture will have 16 filters in the root and the depth of 3 blocks
model = ResUNet(input_shape=(128, 128, 1), classes=2, filters_root=16, depth=3)

# compile the model
# categorical crossentropy is the preferred loss function
model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=["categorical_accuracy", "some other metrics"])

# obtain the dataset
x, y, validation_dataset = ...

# use model.fit, model.evalute as with any other tf2 model
model.fit(x=x, y=y, validation_data=validation_dataset, epochs=args.epochs, batch_size=args.batch_size)
```

Alternatively you can use repository's script *train.py*, although it is not recommended.

## Requirements
- *Tensorflow 2.0* (version can be also higher)
