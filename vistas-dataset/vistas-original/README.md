Welcome to the Mapillary Vistas Dataset (Research edition, v2.0)!
The public set comprises 20,000 images, out of which 18,000 shall be used for training and the remaining 2,000 for validation. The official test set now contains 5,000 RGB images, see changelog below. We provide pixel-wise labels based on polygon annotations for 124 object classes, where 70 are annotated in an instance-specific manner (i.e. individual instances are labeled separately).

The folder structures contain raw RGB images ({training,validation,testing}/images), class-specific labels for semantic segmentation (8-bit with color-palette) ({training,validation}/{v1.2,v2.0}/labels), instance-specific annotations (16-bit) ({training,validation}/{v1.2,v2.0}/instances), panoptic annotations (24-bit RGB images, {training,validation}/{v1.2,v2.0}/panoptic) and raw polygonal annotations (JSON files, {training,validation}/v2.0/polygons). The `v1.2` subfolders contain the older annotations with 65 object classes, while the `v2.0` subfolders contain the current annotations with 124 object classes. Note that polygons are only available for `v2.0`.

Please run 'python demo.py' from the extracted folder to get an idea about how to access label information and for mappings between label IDs and category names.

If you requested access to this dataset for your work as an employee of a company, you and the company will be bound by the license terms. Downloading the dataset implies your acceptance and the company’s acceptance of the terms. By downloading the dataset, you and the company certify that you have understood and accepted the definition of Research Purpose and the restrictions against commercial use as defined in the Mapillary Vistas Dataset Research Use License. A copy of the license terms is part of the downloaded file.

Please cite the following paper if you find Mapillary Vistas helpful for your work:

@InProceedings{MVD2017,
title=    {The Mapillary Vistas Dataset for Semantic Understanding of Street Scenes},
author=   {Neuhold, Gerhard and Ollmann, Tobias and Rota Bul\`o, Samuel and Kontschieder, Peter},
booktitle={International Conference on Computer Vision (ICCV)},
year=     {2017},
url=      {https://www.mapillary.com/dataset/vistas}
}


/Mapillary Research, December, 2020
https://research.mapillary.com/

# Changelog

## [2.0] - 2020-12-08
- Expanded the set of labels to 124 classes (70 instance-specific, 46 stuff, 8 void or crowd).
- Added raw polygonal annotations as json files. These reflect the ordering in which the segments where annotated by the original annotators, i.e. approximately from the background towards the camera.
- Both the old and new labels are available in the same distribution package, stored in the `v1.2` and `v2.0` subfolders, respectively.

## [1.2] - 2020-04-24
- All RGB images in {training,validation,testing}/images are now blurred in privacy-sensitive regions. In particular, we applied blurring for license-plates of vehicles and faces.

## [1.1] - 2018-06-26
### Added
- Panoptic segmentation annotations in {training,validation}/panoptic + corresponding json files, formatted according to COCO definitions from http://cocodataset.org/#format-data (§4, Panoptic Segmentation)
- Added 4911 RGB test images from v1.0 test set
- Added further 89 RGB test images to complete 5,000 test set

### Changed
- Correcting orientation for 9 RGB images by 180° in training/images
  wQKBmli3PVNgLnvn4lgP6Q.jpg
  Xp4QBzzqz2P6RByewwM51w.jpg
  yeCxqrt9b3nabN06KAy7DQ.jpg
  sj9an531eFVHaWGjRkRiGQ.jpg
  tK64qZXtSK0AQhUjCSmiMg.jpg
  drCqjhWopX9omPxc0yo6yw.jpg
  e4H7NbJiZn7QWuLSpvpVKA.jpg
  fG1XiWwwhPeT7HT9cDICdw.jpg
  MM-4wjjF2JySzJppC34xJg.jpg
- Updated demo.py to document panoptic format
