import os
import json
import itertools

import pandas as pd

from typing import List


def _load_images_df(images_path: str) -> pd.DataFrame:
    """ Returns a DataFrame with the path and id of images in the dataset """
    image_files = [
        os.path.join(images_path, path)
        for path in os.listdir(images_path)
    ]
    image_df = pd.DataFrame(image_files, columns=['image_path'])
    # Set image filename as id
    image_df['id'] = image_df['image_path'].apply(
        lambda x: os.path.splitext(os.path.basename(x))[0]
    ).astype(int)
    return image_df.set_index('id')


def _read_annotations(annotation_path: str) -> List[dict]:
    """ Returns the list of annotations in the annotation file """
    annotation_id = os.path.splitext(os.path.basename(annotation_path))[0]

    with open(annotation_path, 'r') as f:
        metadata = json.load(f)

    # Read each item in the image
    items = [metadata[k] for k in metadata.keys() if 'item' in k]
    
    # Keep other metadata, just in case
    image_metadata = {k: metadata[k] 
                      for k in metadata.keys() if 'item' not in k}
    
    # Add index in each of the items
    for item in items:
        item.update({'id': int(annotation_id)})
        item.update(image_metadata)
    
    return items


def _load_annotations_df(annotations_path: str) -> pd.DataFrame:
    """ Loads a DataFrame containing the """
    # Build absolute paths
    annotation_files = [
        os.path.join(annotations_path, path)
        for path in os.listdir(annotations_path)
    ]

    # Map each file into a list of annotations
    annotations = [_read_annotations(annotation_path)
                   for annotation_path in annotation_files]
    # Unpack lists
    annotations = list(itertools.chain.from_iterable(annotations))
    
    return pd.DataFrame(annotations).set_index('id')


def load_training_df(dataset_path: str) -> pd.DataFrame:
    """ Load data from training set in Deep Fashion 2 into DataFrame """
    training_dir = os.path.join(dataset_path, 'train')
    images_dir = os.path.join(training_dir, 'image')
    images_df = _load_images_df(images_dir)
    annotations_df = _load_annotations_df(os.path.join(training_dir, 'annos'))
    # Merge DataFrames
    df = annotations_df.join(images_df, how='left')
    # Make sure no image or annotation is missing
    assert len(df) == len(annotations_df)
    return df
