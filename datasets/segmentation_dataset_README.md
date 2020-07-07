# Deep Fashion segmentation dataset

This dataset contains a clean version of the [Deep Fashion 2 dataset](https://github.com/switchablenorms/DeepFashion2)
for clothing segmentation.

Dataset contains 191961 images (which can be user generated), each of those
mapped to its corresponding segmentation mask. Masks are uint8 (i.e. interval
[0, 255]) images of the same size as the original image that indicate the
clothes from certain categories that appear in the image. Given a pixel of the
mask, it contains no relevant clothing item if it is 0. Otherwise, pixel is
tagged with the corresponding identifier.

6 categories have been defined:

| Category name | # of items |
|---------------|------------|
| Top           | 125789     |
| Shorts        | 36616      |
| Dress         | 49559      |
| Skirt         | 30835      |
| Trousers      | 55387      |
| Outwear       | 14000      |
 

Rows have been randomly split into 3 sets: training (184821 rows),
test(4800 rows) and validation (2340 rows).

# Folder structure:

- `data.json`: Contains the path (relative to this folder) to the original image
  (i.e. `image_path`) and the path to the mask image (`mask_path`). It also
  contains, per each image, the list of labels it has.
- `images`: Directory where original images are.
- `masks`: Directory where the mask for the images are stored.
- `labels.json`: Contains the mapping between category ids and names.
