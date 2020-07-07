# Deep Fashion segmentation dataset

- [dataset_exploration.ipynb](dataset_exploration.ipynb): Notebook to have a 
  general view  of the Deep Fashion 2 dataset.

- [dataset_generation.ipynb](dataset_generation.ipynb): Notebook to generate 
  a segmentation dataset ready for training,

## Setting environment up

Make sure conda >=4.8.3 is installed in your system. Then, type:

```shell script
conda env create -f environment.yml
conda env activate segmentation
```

Moreover, make sure you have downloaded the Deep Fashion 2 dataset from their [Github page](https://github.com/switchablenorms/DeepFashion2).
 
## Exploring dataset

You can execute the dataset exploration notebook by typing:

```shell script
papermill dataset_exploration.ipynb <output_notebook> -p dataset_path <path>
```

Where:
  - `<output_notebook>`: path where you want to store the resulting
    notebook, where the corresponding output cells have been filled.
  - `<path>`: root directory of the Deep Fashion 2 dataset.

## Generating the dataset

The notebook generates two directories:

    - Deepfashion2 dataset for clothing segmentation.
    - Deepfashion2 toy dataset (i.e. 10 instances) for clothing segmentation,
      The purpose of this dataset is to check the sanity of our code during 
      development. If we can easily overfit a tiny subset of rows, it means 
      the model is learning.

To generate those, you can type:

```shell script
papermill dataset_generation.ipynb data_gen.out.ipynb \                                         
  -p dataset_path deep_fashion \
  -p output_dir deep_fashion_seg_dataset \ 
  -p toy_output_dir deep_fashion_toy_seg_dataset
```

Where:
  - `<output_notebook>`: path where you want to store the resulting notebook,
  where the corresponding output cells have been filled.
  - `<path>`: root directory of the Deep Fashion 2 dataset.
  - `<output_dir>`: output directory where to store the segmentation dataset.
  - `<toy_output_dir>`: output directory where to store the toy segmentation 
    dataset.
