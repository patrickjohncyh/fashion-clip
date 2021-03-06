# `FashionCLIP`

__NB: Repo is still WIP!__

We are awaiting the release of the fashion dataset, upon which model weights,
pre-processed image and text vectors will be made public. In the meanwhile, you
can use the model weights from the original `CLIP` [repo](https://github.com/openai/CLIP) 
by following the same model naming convention (i.e. `fclip = FashionCLIP('ViT-B/32', ... )`) or load
your own weights (i.e. `fclip = FashionCLIP('path/to/local/weights.pt', ... )`). See below for further
details!

## Overview

`FashionCLIP` is a CLIP-like model fine-tuned for the fashion industry. We fine tune 
`CLIP` ([Radford et al., 2021](https://arxiv.org/abs/2103.00020])) on over 700K 
<image, text> pairs from an open source fashion catalog[^1].

We evaluate FashionCLIP by applying it to open problems in industry such as retrieval, classificaiton,
and fashion parsing. Our results demonstrate that fine-tuning helps capture domain-specific concepts 
and generalize them in zero-shot scenarios. We also supplement quantitative tests with qualitative analyses, 
and offer preliminary insights into how concepts grounded in a visual space unlocks linguistic generalization. 
Please see our [paper](https://arxiv.org/abs/2204.03972) for more details.

In this repository, you will find an API for interacting with `FashionCLIP` and an interactive demo (coming soon!) 
 show casing the capabilities of `FashionCLIP` built using [streamlit](https://streamlit.io/).


[^1]: Pending official release.


## API & Demo

### Pre-requisites 

To access the private bucket necessary to retrieve model weights and dataset, be sure to include an `.env` 
file containing the following:

```
AWS_ACCESS_KEY_ID
AWS_SECRET_KEY
```

### FashionCLIP API

#### Installation
From project root, install the `fashion-clip` package locally with 
```
$ pip install -e . 
```

#### Usage

There are two main abstractions to facilitate easy use of `FashionCLIP`.

First, the __`FCLIPDataset`__ class which encapsulates information related to a given catalog
and exposes information critical for `FashionCLIP`. Additionally, it provides helper functions
for quick exploration and visualization of data. The main initialization parameters are

```
name: str -> Name of dataset
image_source_path: str -> absolute path to images (can be local or s3) 
image_source_type: str -> type of source (i.e. local or s3)
catalog: List[dict] = None -> list of dicts containing at miniumum the keys ['id', 'image', 'caption']
```

For ease of use, we also provide access to the catalog used in the paper for training `FahionCLIP` 
(_once it is made public_) by simply specifying the corresponding catalog name.

#### Pre-Included Dataset
```
from fashion_clip import FCLIPDataset
dataset = FCLIPDataset(name='FF', 
                       image_source_path='path/to/images', 
                       image_source_type='local')
```

#### Custom dataset

```
from fashion_clip import FCLIPDataset
my_catalog = [{'id': 1, 'image': 'x.jpg', 'caption': 'image x'}]
dataset = FCLIPDataset(name='my_dataset', 
                       image_source_path='path/to/images', 
                       image_source_type='local',
                       catalog=my_catalog)
```

The second abstraction is the __`FashionCLIP`__ class, which takes in a CLIP-like model and an `FCLIPDataset`
and provides methods to perform tasks such as multi-modal retrieval, zero-shot classification and localization.
The initialization parameters for `FashionCLIP` are as follows:

```
model_name: str -> Name of model OR path to local model
dataset: FCLIPDataset -> Dataset, 
normalize: bool -> option to convert embeddings to unit norm  
approx: bool -> option to use approximate nearest neighbors
```

Similar to the `FCLIPDataset` abstraction, we have included a pre-trained `FashionCLIP` model from the paper. 
If an unknown dataset and model combination is received, the image and caption vectors will be generated 
upon instantiation, otherwise pre-computed vectors/embeddings will be pulled from S3.

```
from fashion_clip import FCLIPDataset, FashionCLIP
dataset = FCLIPDataset(name='FF', 
                       image_source_path='path/to/images', 
                       image_source_type='local')
fclip = FashionCLIP('FCLIP', ff_dataset)
```

For further details on how to use the package, refer to the accompanying notebook!

### FashionCLIP Demo

The demo is built using streamlit, with further instructions and explanations included
inside.

Running the app requires access to the dataset/fine-tuned model. Stay tuned for more updates!

#### How to run
```
$ cd app
$ streamlit run app.py
```
