# FashionCLIP

[![Youtube Video](https://img.shields.io/badge/youtube-video-red)](https://www.youtube.com/watch?v=uqRSc-KSA1Y)
[![HuggingFace Model](https://img.shields.io/badge/HF%20Model-Weights-yellow)](https://huggingface.co/patrickjohncyh/fashion-clip)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Z1hAxBnWjF76bEi9KQ6CMBBEmI_FVDrW?usp=sharing)
[![Medium Blog Post](https://raw.githubusercontent.com/aleen42/badges/master/src/medium.svg)](https://towardsdatascience.com/teaching-clip-some-fashion-3005ac3fdcc3)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://huggingface.co/spaces/vinid/fashion-clip-app)


## Quick Start

| Name            | Link     | 
|-----------------|----------|
| FashionCLIP Feature Extraction and Classification | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Z1hAxBnWjF76bEi9KQ6CMBBEmI_FVDrW?usp=sharing)|
| Tutorial -  FashionCLIP Evaluation with RecList | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ek-TIT1ZJta59-O73GaXsOINvt46dnkz?usp=sharing)|


UPDATE (10/03/23): We have updated the model! We found that [laion/CLIP-ViT-B-32-laion2B-s34B-b79K](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K) checkpoint (thanks [Bin](https://www.linkedin.com/in/bin-duan-56205310/)!) worked better than original OpenAI CLIP on Fashion. We thus fine-tune a newer (and better!) version of FashionCLIP (henceforth FashionCLIP 2.0), while keeping the architecture the same. We postulate that the perofrmance gains afforded by `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` are due to the increased training data (5x OpenAI CLIP data). Our [thesis](https://www.nature.com/articles/s41598-022-23052-9), however, remains the same -- fine-tuning `laion/CLIP` on our fashion dataset improved zero-shot perofrmance across our benchmarks. See the below table comparing weighted macro F1 score across models.
`

| Model             | FMNIST        | KAGL          | DEEP          | 
| -------------     | ------------- | ------------- | ------------- |
| OpenAI CLIP       | 0.66          | 0.63          | 0.45          |
| FashionCLIP       | 0.74          | 0.67          | 0.48          |
| Laion CLIP        | 0.78          | 0.71          | 0.58          |
| FashionCLIP 2.0   | __0.83__          | __0.73__          | __0.62__          |

---

We are now on Hugging Face! The model is available [here](https://huggingface.co/patrickjohncyh/fashion-clip).

We are now on [Nature Scientific Reports](https://www.nature.com/articles/s41598-022-23052-9)!

## Citation
```
@Article{Chia2022,
    title="Contrastive language and vision learning of general fashion concepts",
    author="Chia, Patrick John
            and Attanasio, Giuseppe
            and Bianchi, Federico
            and Terragni, Silvia
            and Magalh{\~a}es, Ana Rita
            and Goncalves, Diogo
            and Greco, Ciro
            and Tagliabue, Jacopo",
    journal="Scientific Reports",
    year="2022",
    month="Nov",
    day="08",
    volume="12",
    number="1",
    pages="18958",
    abstract="The steady rise of online shopping goes hand in hand with the development of increasingly complex ML and NLP models. While most use cases are cast as specialized supervised learning problems, we argue that practitioners would greatly benefit from general and transferable representations of products. In this work, we build on recent developments in contrastive learning to train FashionCLIP, a CLIP-like model adapted for the fashion industry. We demonstrate the effectiveness of the representations learned by FashionCLIP with extensive tests across a variety of tasks, datasets and generalization probes. We argue that adaptations of large pre-trained models such as CLIP offer new perspectives in terms of scalability and sustainability for certain types of players in the industry. Finally, we detail the costs and environmental impact of training, and release the model weights and code as open source contribution to the community.",
    issn="2045-2322",
    doi="10.1038/s41598-022-23052-9",
    url="https://doi.org/10.1038/s41598-022-23052-9"
}
```


## Information 

We are awaiting the official release of the Farfetch dataset, upon which fine-tuned model weights,
pre-processed image and text vectors will be made public. In the meanwhile, we currently use the 
[Hugging Face](https://huggingface.co/) implementation of `CLIP` and can use the model weights
from [OpenAI](https://huggingface.co/openai/clip-vit-base-patch32) by following the standard hugginface 
naming convention (i.e. `fclip = FashionCLIP('<username>/<repo_name>', ... )`). We also support private
repositories (i.e. `fclip = FashionCLIP('<username>/<repo_name>', auth_token=<AUTH_TOKEN>, ... )`). 

See below for further details!

## Overview

`FashionCLIP` is a CLIP-like model fine-tuned for the fashion industry. We fine tune 
`CLIP` ([Radford et al., 2021](https://www.nature.com/articles/s41598-022-23052-9) on over 700K 
<image, text> pairs from the Farfetch dataset[^1].

We evaluate FashionCLIP by applying it to open problems in industry such as retrieval, classification
and fashion parsing. Our results demonstrate that fine-tuning helps capture domain-specific concepts 
and generalizes them in zero-shot scenarios. We also supplement quantitative tests with qualitative analyses, 
and offer preliminary insights into how concepts grounded in a visual space unlocks linguistic generalization. 
Please see our [paper](https://www.nature.com/articles/s41598-022-23052-9) for more details.

In this repository, you will find an API for interacting with `FashionCLIP` and an interactive demo built using [streamlit](https://streamlit.io/) (coming soon!) 
 which showcases the capabilities of `FashionCLIP`.


[^1]: Pending official release.


## API & Demo


### Quick How To

Need a quick way to generate embeddings? do you want to test retrieval performance? 

First of all, you should be able to quickly install this using pip.

```
$ pip install fashion-clip 
```

If you have lists of texts and image paths, it is very easy to generate embeddings:

```python

from fashion_clip.fashion_clip import FashionCLIP

fclip = FashionCLIP('fashion-clip')

# we create image embeddings and text embeddings
image_embeddings = fclip.encode_images(images, batch_size=32)
text_embeddings = fclip.encode_text(texts, batch_size=32)

# we normalize the embeddings to unit norm (so that we can use dot product instead of cosine similarity to do comparisons)
image_embeddings = image_embeddings/np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)
text_embeddings = text_embeddings/np.linalg.norm(text_embeddings, ord=2, axis=-1, keepdims=True)
```

**Use our [colab](https://colab.research.google.com/drive/1Z1hAxBnWjF76bEi9KQ6CMBBEmI_FVDrW?usp=sharing)** notebook to see more functionalities.


### HF API

```python

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

image = Image.open("images/image1.jpg")

inputs = processor(text=["a photo of a red shoe", "a photo of a black shoe"],
                   images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  
print(probs)
image.resize((224, 224))
```

### Additional Internal FashionCLIP API

#### Installation
From project root, install the `fashion-clip` package locally with 
```
$ pip install -e . 
```


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

For ease of use, the API also provides access to the dataset (_once it is officialy released_), used in the paper 
for training `FahionCLIP`, by simply specifying the corresponding catalog name.

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

The second abstraction is the __`FashionCLIP`__ class, which takes in a Hugging Face CLIP model name and 
an `FCLIPDataset`, and provides convenient functions to perform tasks such as multi-modal retrieval, 
zero-shot classification and localization. The initialization parameters for `FashionCLIP` are as follows:

```
model_name: str -> Name of model OR path to local model
dataset: FCLIPDataset -> Dataset, 
normalize: bool -> option to convert embeddings to unit norm  
approx: bool -> option to use approximate nearest neighbors
```

Similar to the `FCLIPDataset` abstraction, we have included a pre-trained `FashionCLIP` model from the paper, hosted
[here](https://huggingface.co/patrickjohncyh/fashion-clip). If an unknown dataset and model combination is received, 
the image and caption vectors will be generated upon object instantiation, otherwise pre-computed vectors/embeddings will 
be pulled from S3.

```
from fashion_clip import FCLIPDataset, FashionCLIP
dataset = FCLIPDataset(name='FF', 
                       image_source_path='path/to/images', 
                       image_source_type='local')
fclip = FashionCLIP('fasihon-clip', ff_dataset)
```

For further details on how to use the package, refer to the accompanying notebook!

## Fun Related Projects!

* Check [RustEmbed](https://github.com/yaman/RustEmbed) for an application to use gRPC to create embeddings with FashionCLIP.

