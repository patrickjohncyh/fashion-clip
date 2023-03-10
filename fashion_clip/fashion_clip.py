import os
import clip
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Union, Tuple
from torch.utils.data import DataLoader
from fashion_clip.utils import get_cache_directory, _is_hugging_face_repo, _model_processor_hash
from fashion_clip.utils import _download, file_sha256, display_images_from_s3, display_images_from_url, display_images
import fashion_clip.attention_map as attention_map
import PIL
import hashlib
import random
from annoy import AnnoyIndex
import time
import json
import validators
from transformers import CLIPModel, CLIPProcessor
from datasets import Dataset, Image

_MODELS = {
    "fashion-clip": "patrickjohncyh/fashion-clip",
}
_CATALOGS = {
    "FF": "s3://fashion-clip-internal-k1q9ahssz8dr4b0mn3on/catalogs/8721deef0bae2bef9a40703ccc7d931eb353fccc7ff3245a2647491cc8202761/ff_catalog.json",
}
_VECTORS = {
    "0a61c5023f8b30bee944a3fee22b9d9d5a5fe3008043fbe8823c8411ca8e0b80_8500088d34cbf5aae5f2ffd132da16ebba56e4104a002ec85cb81a4a2bebe72d": {
        'IMAGE': "s3://fashion-clip-internal-k1q9ahssz8dr4b0mn3on/vectors/bacb9e2b1602e192fd58138c68fc574590efd07b560d4031f1a54a078a95ba34/FCLIP_FF_IMAGE_VECTORS.npy",
        'TEXT': "s3://fashion-clip-internal-k1q9ahssz8dr4b0mn3on/vectors/b75e8817516a81922138e0502fedcce0248eef0f766b339f3b11b76837861a64/FCLIP_FF_TEXT_VECTORS.npy"
    },
    "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af_8500088d34cbf5aae5f2ffd132da16ebba56e4104a002ec85cb81a4a2bebe72d": {
        'IMAGE': "s3://fashion-clip-internal-k1q9ahssz8dr4b0mn3on/vectors/4724a8684ce2745c370276d74f1ce2dcf8654aae969a795dafa1d06d4578ae2c/CLIP_FF_IMAGE_VECTORS.npy",
        'TEXT': "s3://fashion-clip-internal-k1q9ahssz8dr4b0mn3on/vectors/9af4e453f63abe0411863e3a419c82a0c237020b45a073d731be753b76614aa2/CLIP_FF_TEXT_VECTORS.npy"
    }
}


_CACHE_DIR = get_cache_directory()

class FCLIPDataset:
    # Can initialize as
    # 1. Existing remotely stored catalog (i.e. the one from FF).
    # 2. Specific a new catalog with path to images + corresponding text
    def __init__(self, name: str, image_source_path: str, image_source_type: str,  catalog: List[dict] = None ):

        image_source_type = image_source_type.upper()
        assert image_source_type in ['LOCAL', 'S3', 'URL']

        if name in _CATALOGS:
            print('Loading dataset {}...'.format(name))
            catalog_path = _download(_CATALOGS[name], _CACHE_DIR)
            with open(catalog_path, 'rb') as f:
                catalog = json.load(f)
        elif catalog is None:
            raise RuntimeError(f"Catalog {name} not found; available catalogs = {list(_CATALOGS.keys())}")
        self.name = name
        self.catalog = np.array(catalog)
        # extract FCLIP related info
        self.ids = np.array([str(_['id']) for _ in catalog])
        self.images = np.array([_['image'] for _ in catalog])
        self.images_path = np.array([os.path.join(image_source_path, _['image']) for _ in catalog])
        self.captions = np.array([_['caption'] for _ in catalog])
        self.id_to_idx = {id: idx for idx, id in enumerate(self.ids)}
        self._display_images = display_images
        if image_source_type == 'S3':
            self._display_images = display_images_from_s3
        elif image_source_type == 'URL':
            self._display_images = display_images_from_url

    def display_products(self, ids: List[str], fields: Tuple[str] = ('id', 'short_description'), **kwargs):
        idxs = [self.id_to_idx[id] for id in ids]
        im_paths = list(self.images_path[idxs])
        prod_info = [{field: row[field] for field in fields}  for row in self.catalog[idxs]]
        return self._display_images(im_paths, product_info=prod_info, **kwargs)

    def random_product_id(self):
        return self.ids[random.randint(0, len(self.ids))]

    def _retrieve_row(self, id: str):
        return self.catalog[self.id_to_idx[id]]

    # def get_metadata(self, id, fields=['caption', 'price']):
    #     pass
    def _hash_list(self, l: List[str]):
        l_hash = map(lambda el: hashlib.sha256(str(el).encode()).hexdigest(), l)
        l_hash = ' '.join(l_hash)
        return hashlib.sha256(l_hash.encode()).hexdigest()

    def hash(self):
        id_hash = self._hash_list(self.ids)
        images_hash = self._hash_list(self.images)
        caption_hash = self._hash_list(self.captions)
        return hashlib.sha256((id_hash+images_hash+caption_hash).encode()).hexdigest()

class FashionCLIP:
    """
    FashionCLIP class takes:
        1. FCLIPModel Name / Path
        2. FCLIPDataset
    Then, it generates required embeddings based on the dataset
    OR if a pre-processed versions exists then pull it from S3?
    Need a reliable method of determining if pre-processed version exists -- use hash of the dataset_model?
    """

    def __init__(self, model_name, dataset: FCLIPDataset = None, normalize=True, approx=True, auth_token=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model, self.preprocess, self.model_hash = self._load_model(model_name, auth_token=auth_token)
        self.model = self.model.to(self.device)
        self.dataset = dataset
        if self.dataset:
            self.dataset_hash = self.dataset.hash()
            self.vector_hash = "_".join([self.model_hash, self.dataset_hash])
        # print(self.vector_hash)
        self.approx = approx
        if dataset is not None:
            self.image_vectors, self.textual_vectors = self._generate_vectors()
            assert self.image_vectors.shape == self.textual_vectors.shape

        if normalize and dataset:
            print('Normalizing Input Vectors...', end='')
            self.image_vectors = self.image_vectors / np.linalg.norm(self.image_vectors, ord=2, axis=-1, keepdims=True)
            self.textual_vectors = self.textual_vectors / np.linalg.norm(self.textual_vectors, ord=2, axis=-1, keepdims=True)
            print('Done!')
        if approx and dataset:
            print('Building Approx NN index...', end='')
            # build approx NN
            self.nn_index = AnnoyIndex(512, "dot")
            for idx, v in enumerate(self.image_vectors):
                self.nn_index.add_item(idx, v)
            self.nn_index.build(50)  # 10 trees
            print('Done!')

    def _generate_vectors(self, cache=True):

        # check if dataset + model embedding exists
        if self.vector_hash in _VECTORS:
            image_vectors_path = _download(_VECTORS[self.vector_hash]['IMAGE'], _CACHE_DIR)
            text_vectors_path = _download(_VECTORS[self.vector_hash]['TEXT'], _CACHE_DIR)
            image_vectors = np.load(image_vectors_path)
            textual_vectors = np.load(text_vectors_path)
        else:
            # generate image vectors
            image_vectors = self.encode_images(self.dataset.images_path, batch_size=32)
            # generate textual vectors
            textual_vectors = self.encode_text(self.dataset.captions, batch_size=32)
        # TODO: Implement some sort of caching mechanism?
        return image_vectors, textual_vectors

    def _load_model(self,
                    name: str,
                    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
                    auth_token = None):
        # model is one of know HF models
        if name in _MODELS or _is_hugging_face_repo(name, auth_token):
            # if using know short-hand, extract from dict
            name = _MODELS[name] if name in _MODELS else name
            model = CLIPModel.from_pretrained(name, use_auth_token=auth_token)
            preprocessing = CLIPProcessor.from_pretrained(name, use_auth_token=auth_token)
            hash = _model_processor_hash(name, model, preprocessing)

        # else it doesn't use HF, assume using OpenAI CLiP
        else:
            if os.path.isfile(name):
                model_path = name
            elif validators.url(name):
                # generic url or S3 path
                model_path = _download(_MODELS[name], _CACHE_DIR)
            else:
                raise RuntimeError(f"Model {name} not found or not valid; available models = {list(_MODELS.keys())}")

            model, preprocessing = clip.load(model_path, device=device, download_root=_CACHE_DIR)
            hash = file_sha256(model_path)

        return model, preprocessing, hash

    def encode_images(self, images: Union[List[str], List[PIL.Image.Image]], batch_size: int):
        def transform_fn(el):
             imgs = el['image'] if isinstance(el['image'][0], PIL.Image.Image) else [Image().decode_example(_) for _ in el['image']] 
             return self.preprocess(images=imgs, return_tensors='pt')
            
        dataset = Dataset.from_dict({'image': images})
        dataset = dataset.cast_column('image',Image(decode=False)) if isinstance(images[0], str) else dataset        
        # dataset = dataset.map(map_fn,
        #             batched=True,
        #             remove_columns=['image'])
        dataset.set_format('torch')
        dataset.set_transform(transform_fn)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        image_embeddings = []
        pbar = tqdm(total=len(images) // batch_size, position=0)
        with torch.no_grad():
            for batch in dataloader:
                batch = {k:v.to(self.device) for k,v in batch.items()}
                image_embeddings.extend(self.model.get_image_features(**batch).detach().cpu().numpy())
                pbar.update(1)
            pbar.close()
        return np.stack(image_embeddings)

    def encode_text(self, text: List[str], batch_size: int):
        dataset = Dataset.from_dict({'text': text})
        dataset = dataset.map(lambda el: self.preprocess(text=el['text'], return_tensors="pt",
                                                         max_length=77, padding="max_length", truncation=True),
                              batched=True,
                              remove_columns=['text'])
        dataset.set_format('torch')
        dataloader = DataLoader(dataset, batch_size=batch_size)
        text_embeddings = []
        pbar = tqdm(total=len(text) // batch_size, position=0)
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                text_embeddings.extend(self.model.get_text_features(**batch).detach().cpu().numpy())
                pbar.update(1)
            pbar.close()
        return np.stack(text_embeddings)

    def _cosine_similarity(self, key_vectors: np.ndarray, space_vectors: np.ndarray, normalize=True):
        if normalize:
            key_vectors = key_vectors / np.linalg.norm(key_vectors, ord=2, axis=-1, keepdims=True)
        return np.matmul(key_vectors, space_vectors.T)

    def _nearest_neighbours(self, k, key_vectors, space_vectors, normalize=True, debug=False):
        if type(key_vectors) == List:
            key_vectors = np.array(key_vectors)
        if type(space_vectors) == List:
            space_vectors = np.array(space_vectors)

        t1 = time.time()
        if self.approx:
            if debug:
                print('Using ANNOY')
            if normalize:
                key_vectors = key_vectors / np.linalg.norm(key_vectors, ord=2, axis=-1, keepdims=True)
            nn = [self.nn_index.get_nns_by_vector(v, k, search_k=-1, include_distances=False) for v in key_vectors]
        else:
            if debug:
                print('Using Dot Product')
            cosine_sim = self._cosine_similarity(key_vectors, space_vectors, normalize=normalize)
            nn = cosine_sim.argsort()[:, -k:][:, ::-1]
        t2 = time.time()
        if debug:
            print('Elapsed Time: {}s'.format(t2-t1))

        return nn

    def zero_shot_classification(self, images, text_labels: List[str], debug=False):
        """
        Perform zero-shot image classification
        :return:
        """
        # encode text
        text_vectors = self.encode_text(text_labels, batch_size=8)
        # encode images
        image_vectors = self.encode_images(images, batch_size=8)
        # compute cosine similarity
        cosine_sim = self._cosine_similarity(image_vectors, text_vectors)
        if debug:
            print(cosine_sim)
        preds = np.argmax(cosine_sim, axis=-1)
        return [text_labels[idx] for idx in preds]

    def retrieval(self, queries: List[str], top_k: int = 10):
        """
        Image retrieval from queries
        :return:
        """
        # encode text
        text_vectors = self.encode_text(queries, batch_size=8)
        # compute cosine similarity
        # cosine_sim = self._cosine_similarity(text_vectors, self.image_vectors)
        return self._nearest_neighbours(k=top_k, key_vectors=text_vectors, space_vectors=self.image_vectors)

        # return np.argmax(cosine_sim, axis=-1)
        # return cosine_sim.argsort()[:,-top_k:][:,::-1]

    def display_attention(self, image_path, query_text, pixel_size=15, iterations=5):
        heatmap, image = self._get_heatmap(image_path, query_text, pixel_size, iterations)
        attention_map.display_heatmap(image, query_text, heatmap)

    def _get_heatmap(self, image_path, text, pixel_size, iterations):
        images, masks = attention_map.generate_image_crops(image_path, pixel_size=pixel_size)
        text_vector = self.encode_text([text], batch_size=1)[0]
        text_vector = text_vector / np.linalg.norm(text_vector, ord=2)
        image_vectors = self.encode_images(images, batch_size=32)
        image_vectors = image_vectors / np.linalg.norm(image_vectors, axis=-1, ord=2, keepdims=True)

        sims = []
        scores = []
        for e, m in zip(image_vectors, masks):
            sim = np.matmul(e, text_vector.T)
            sims.append(sim)
            scores.append(sim * m)
        score = np.mean(np.clip(np.array(scores) - sims[0], 0, np.inf), axis=0)
        for i in range(iterations):
            score = np.clip(score - np.mean(score), 0, np.inf)
        score = (score - np.min(score)) / (np.max(score) - np.min(score))

        return np.asarray(score), images[0]

