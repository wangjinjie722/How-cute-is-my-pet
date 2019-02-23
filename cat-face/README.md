# How Cute is My Pet

ECE 271B Project
Image preprocessing

## Getting Started

The goal of image processing is to extract cat/dog face features from training and test images, organize them in a convenient way fro further learning purposes.

The face detector uses pre-trained Detector Cascades.

The subtracted faces are converted into grayscale, histogram-normalized png file with resolution 128x128.

##### Useful files are:
* *haarcascade_frontalcatface.xml*: cat-face detector cascades
* *catface_detector.py*: a detector class for cat-face detection
* *image_preprocess.py*: image preprocess program, with user-friendly progress bar to monitor progess
* *image_preprocess_multithread.py*: multithread image preprocess program, with significantly speed improvement on multi-core CPUs

### Prerequisites

```
Python 3
numpy
opencv2
```

## Running the codes
The programs read and save files in relative path. Make sure to check the directories before running them.

### Running image_preprocess.py

Check the directory path:
Train image path: *./How-cute-is-my-pet/petfinder-adoption-prediction/train_images*
Test image path: *./How-cute-is-my-pet/petfinder-adoption-prediction/test_images*

In the main function, the settings are:
```
traindata=True: process traindata
testdata=True: process testdata
detect=True: use cat-face detector to generate face imgs
grayscale=True: convert face imgs to grayscale, then histogram regularize
organize=True: keep only one img for each petID, save as .png
```

If **detect=True**, the detected face imgs from train set and test set are saved to:
*./How-cute-is-my-pet/cat-face/data*
*./How-cute-is-my-pet/cat-face/testdata*
respectively.

If **grayscale=True**, the grayscale and histogram normalized imgs are saved to:
*./How-cute-is-my-pet/cat-face/data_regularize_hist*
*./How-cute-is-my-pet/cat-face/testdata_regularize_hist*
respectively.

If **organize=True**, the organized .png imgs are saved to:
*./How-cute-is-my-pet/cat-face/data_organized_hist*
*./How-cute-is-my-pet/cat-face/testdata_organized_hist*
respectively.

### Running image_preprocess.py

Same as running *image_preprocess.py*.

This program uses multithread processing to boost performance. May cause high CPU load.

## Authors

* **Mingwei Xu** - *UCSD ECE*

## Acknowledgments

* The cat-face detector cascades is contributed by Joseph Howse (josephhowse@nummist.com)

