import requests
from tqdm import tqdm
from appdirs import *
from pathlib import Path
from urllib.parse import urlparse
from PIL import Image
import matplotlib.pyplot as plt
import textwrap
from fashion_clip.s3_client import S3Client
from urllib.request import urlretrieve
import hashlib
from typing import List

HUGGING_FACE_REPO_URL = "https://huggingface.co/{}/{}"

def file_sha256(file_path: str):
    return hashlib.sha256(open(file_path, "rb").read()).hexdigest()


def _model_processor_hash(name, model, processor)->str:
    # TODO: Figure out how to get the right hashes, that is, we care about the model (and processor) weights/config and not their commit per se
    return model.config._commit_hash
def get_cache_directory():
    """
    Returns the cache directory on the system
    """
    appname = "fashion_clip"
    appauthor = "fashion_clip"
    cache_dir = user_cache_dir(appname, appauthor)

    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    return cache_dir


def _is_hugging_face_repo(path, api_token=None)->bool:
    try:
        username, repo_name = path.split('/')
    except:
        return False
    print(username, repo_name)

    url = HUGGING_FACE_REPO_URL.format(username, repo_name)
    headers = {"Authorization": "Bearer {}".format(api_token)} if api_token else {}
    response = requests.get(url, headers=headers)
    return response.status_code == 200

def _download(url, destination):
    """
    Downloads a file with a progress bar
    :param url: url from which to download from
    :destination: file path for saving data
    """
    expected_sha256 = url.split("/")[-2]
    filename = os.path.basename(url)
    download_target = os.path.join(destination, filename)
    print('Begin download of {} ...'.format(filename))

    if os.path.isfile(download_target):
        if file_sha256(download_target) == expected_sha256:
            print('Using cached version found at : {}'.format(download_target))
            return download_target
        else:
            os.remove(download_target)
            print('WARNING: Cached File SHA does not match expected SHA; re-downloading file')
    if urlparse(url).scheme == 's3':
        # Private S3 bucket
        print('Downloading {} from S3'.format(filename))
        assert os.getenv('AWS_ACCESS_KEY_ID')
        assert os.getenv('AWS_SECRET_KEY')
        s3_client = S3Client(aws_key=os.getenv('AWS_ACCESS_KEY_ID'), aws_secret=os.getenv('AWS_SECRET_KEY'))
        _, bucket, path, _, _, _ = urlparse(url)
        s3_client.download_file_from_bucket(bucket, path[1:], download_target)
    else:
        # Public URL
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise SystemExit(e)
        with tqdm.wrapattr(open(destination, "wb"), "write",
                           miniters=1, desc=url.split('/')[-1],
                           total=int(response.headers.get('content-length', 0))) as fout:
            for chunk in response.iter_content(chunk_size=4096):
                fout.write(chunk)
    if file_sha256(download_target) != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target

def display_images(image_paths,
                   product_info: List[dict] = None,
                   columns=5,
                   width=20,
                   height=8,
                   max_images=15,
                   font_size=8,
                   text_wrap_length=35):
        if not image_paths:
            print("No images to display.")
            return
        if len(image_paths) > max_images:
            print(f"Showing {max_images} images of {len(image_paths)}:")
            image_paths = image_paths[0:max_images]
        height = max(height, len(image_paths) // columns * height)
        # height = height
        fig = plt.figure(figsize=(width, height))
        # open images
        images = [Image.open(f) for f in image_paths]
        for i, image in enumerate(images):
            ax = plt.subplot(int(len(images) / columns + 1), columns, i + 1)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.box(False)
            ax.imshow(image)

            info = product_info[i] if product_info else {}
            display_text = ''
            for k,v in info.items():
                text = textwrap.wrap(str(v), text_wrap_length)
                display_text += '{}\n'.format("\n".join(text))
            ax.text(0, -0.1, display_text,
                     fontfamily='sans serif',
                     fontsize=font_size,
                     transform= ax.transAxes,
                     verticalalignment='top')
        fig.tight_layout(h_pad=3, w_pad=0.1)
        return fig
def display_images_from_url(urls, **kwargs):
    # download urls to local cache
    image_fnames = [os.path.basename(url) for url in urls]
    image_paths = [os.path.join(get_cache_directory(), fname) for fname in image_fnames]
    for url, path in zip(urls, image_paths):
        if os.path.isfile(path):
            continue
        urlretrieve(url, path)
    return display_images(image_paths, **kwargs)

def display_images_from_s3(s3_urls, **kwargs):
    assert os.getenv('AWS_ACCESS_KEY_ID')
    assert os.getenv('AWS_SECRET_KEY')
    s3_client = S3Client(aws_key=os.getenv('AWS_ACCESS_KEY_ID'), aws_secret=os.getenv('AWS_SECRET_KEY'))
    destination_folder = os.path.join(get_cache_directory(),'images')
    os.makedirs(destination_folder, exist_ok=True)
    image_paths = []
    for url in s3_urls:
        image_fname = os.path.basename(url)
        download_target = os.path.join(destination_folder, image_fname)
        image_paths.append(download_target)
        if os.path.isfile(download_target):
            continue
        _, bucket, path, _, _, _ = urlparse(url)
        s3_client.download_file_from_bucket(bucket, path[1:], download_target)
    return display_images(image_paths, **kwargs)