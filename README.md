# Crowd Counting via Adversarial Cross-Scale Consistency Pursuit  IN PYTORCH (Unofficial)

Implementation of CVPR 2O18 paper [Crowd Counting via Adversarial Cross-Scale](http://openaccess.thecvf.com/content_cvpr_2018/papers/Shen_Crowd_Counting_via_CVPR_2018_paper.pdf)

## 1. Environment

We used the following enviroment:

* Python 3
* PyTorch
* OpenCV
* Numpy
* MatPlotLib
* Ubuntu 16.04

You can also run the code using the docker image of [ufoym/deepo](https://hub.docker.com/r/ufoym/deepo).

## 2. Preparing data

We make available UCF-CC-50 and Shanghai Tech datasets [here](http://www.liv.ic.unicamp.br/~quispe/publications/data/data-crowd-counting.zip), download and unzip it into the root of the repo. Directories should have the following hierarchy:

```
ROOT_OF_REPO
    data
        ucf_cc_50
            UCF_CC_50
                images
                labels
        ShanghaiTech
            part_A
                train_data
                    images
                    ground-truth
                test_data
                    images
                    ground-truth
            part_B
                train_data
                    images
                    ground-truth
                test_data
                    images
                    ground-truth
```

The code was developed such that data augmentation is computed before every other step and the results are stored in the hard drive. Thus, the first time you run the code it will take quite a long time. Augmented data is stored with the following hierarchy:

```
ROOT_OF_REPO
    data
        ucf_cc_50
            people_thr_0_gt_mode_same
        ShanghaiTech
            part_A
                people_thr_0_gt_mode_same
            part_B
                people_thr_0_gt_mode_same
```

## 3. Training

To train using UCF-CC-50 (with all folds) and save the results log in `log/ACSCP` you can run:

```
python3 train.py -d ucf-cc-50 --gt-mode same --people-thr 0 --train-batch 24 --save-dir log/ACSCP

```

In case you want to run a specific fold or part you can use flag `--units`, check the `Makefile` for more examples.

The training log is stored in `log_train.txt` inside the corresponding log/fold/part directory.

## 4. Testing

After training you can re-load the trained weights (using flag `--resume`) and use them for testing:

```
python3 train.py -d ucf-cc-50 --save-dir log/multi-stream --resume log/ACSCP/ucf-cc-50_people_thr_0_gt_mode_same --evaluate-only
```

The testing log is stored in `log_test.txt` inside the corresponding log/fold/part directory. You can also generate the plots of the predictions using flag `--save-plots`, results are stored in the directory `plot-results-test` inside the corresponding log/fold/part directory.

## 5. Final notes / TODO:

* Results for UCF_CC_50 with this code are MAE 281,73 MSE 415,56 (--people-thr 20). Reported results by the authors are MAE 291.0 MSE 404.6.
* Validation for other dataset may be done in the future.
* Batch normalization is not used because of inestable learning.
* Ground thruth is generated using a gaussian of fixed size.
* Number of channels of autoencoder in the middle layer is changed to 3, instead of 4.
* Network receives images of 1 channel, instead of 3.
* You can use the flag `--overlap-test` to overlap the sliding windows used for testing (as implemented by the authors).