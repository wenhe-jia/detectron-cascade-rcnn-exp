# detectron-cascadee-exp
Experiment results of implement Cascade RCNN under Detectron.
Using ResNet50 as feature extractor, as well as 1x iterations.
Learning rate start as 0.01 (4GPU, 2 images per GPU).

## Using statement
Folder **detectron_cascade** are codes to implement Cascade RCNN under Detectron, parallelizing with folder **$Detectron/detcectron**. 

Folder **configs/cascade/** contains yaml files conducting the Cascade RCNN model training.

# MSCOCO experiments
## mask iterative bbox rcnn results (using same IOU threshold in three stage of RCNN)
### model is trained on coco2017train + val
| experiments | dataset | box_ap | box_ap50 | box_ap75 | box_ap_small | box_ap_med | box_ap_large | mask_ap | mask_ap50 | mask_ap75 | mask_ap_small | mask_ap_med | mask_ap_large |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| mask-R50       | test-dev | 38.2% | 60.05 | 41.5% | 21.8% | 40.3% | 48.4% | 34.3% | 56.5% | 36.3% | 14.9% | 36.1% | 49.7% |
| cascade stage1 | test-dev | 38.3% |       |       |       |       |       | 34.2% |       |       |       |       |       |
| cascade stage2 | test-dev | 38.9% |       |       |       |       |       | 34.1% |       |       |       |       |       |
| cascade stage3 | test-dev | 38.9% | 59.5% | 42.1% | 21.5% | 40.7% | 50.2% | 34.0% | 56.1% | 35.9% | 14.8% | 35.5% | 49.5% |
| cascade stage 1~2 | test-dev |    |       |       |       |       |       |       |       |       |       |       |       |
| cascade stage 1~3 | test-dev |    |       |       |       |       |       |       |       |       |       |       |       |



## mask cascade rcnn results beta version 1
### (clip bbox and add invalid bbox check in DecodeBBoxOp)
### model is trained on coco2017train + val
| experiments | dataset | box_ap | box_ap50 | box_ap75 | box_ap_small | box_ap_med | box_ap_large | mask_ap | mask_ap50 | mask_ap75 | mask_ap_small | mask_ap_med | mask_ap_large |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| mask-R50       | test-dev | 38.2% | 60.05 | 41.5% | 21.8% | 40.3% | 48.4% | 34.3% | 56.5% | 36.3% | 14.9% | 36.1% | 49.7% |
| cascade stage1 | test-dev | 38.2% | 59.9% | 41.7% | 21.7% | 40.4% | 48.4% | 34.2% | 56.4% | 36.1% | 15.0% | 36.0% | 49.5% |
| cascade stage2 | test-dev | 38.1% | 58.5% | 41.3% | 18.2% | 39.7% | 53.4% | 34.7% | 56.5% | 36.8% | 15.1% | 36.5% | 50.6% |
| cascade stage3 | test-dev | 39.4% | 57.5% | 43.5% | 21.4% | 41.2% | 51.1% | 34.2% | 55.0% | 36.4% | 14.7% | 35.9% | 49.9% |
| cascade stage 1~2 | test-dev |    |       |       |       |       |       |       |       |       |       |       |       |
| cascade stage 1~3 | test-dev |    |       |       |       |       |       |       |       |       |       |       |       |



## mask cascade rcnn results beta version 2
### (screen out high iou boxes in DecodeBBoxOp)
### model is trained on coco2017train
| experiments | dataset | box_ap | box_ap50 | box_ap75 | box_ap_small | box_ap_med | box_ap_large | mask_ap | mask_ap50 | mask_ap75 | mask_ap_small | mask_ap_med | mask_ap_large |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| mask-R50       | test-dev | 38.6%  |        |        |        |        |        | 34.5%  |        |        |        |        |        |
| cascade stage1 | test-dev |        |        |        |        |        |        |        |        |        |        |        |        |
| cascade stage2 | test-dev |        |        |        |        |        |        |        |        |        |        |        |        |
| cascade stage3 | test-dev | 39.06% | 56.98% | 43.28% | 21.86% | 41.54% | 52.41% | 34.20% | 54.47% | 36.65% | 15.11% | 36.47% | 51.51% | 
| cascade stage 1~2 | test-dev |     |        |        |        |        |        |        |        |        |        |        |        |
| cascade stage 1~3 | test-dev |     |        |        |        |        |        |        |        |        |        |        |        |



## mask cascade rcnn results beta version 3
### (add weight to rcnn loss)
### model is trained on coco2017train
| experiments | dataset | box_ap | box_ap50 | box_ap75 | box_ap_small | box_ap_med | box_ap_large | mask_ap | mask_ap50 | mask_ap75 | mask_ap_small | mask_ap_med | mask_ap_large |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| mask-R50       | test-dev | 38.0% |        |        |        |        |        | 34.5%  |        |        |        |        |        |
| cascade stage1 | test-dev |        |        |        |        |        |        |        |        |        |        |        |        |
| cascade stage2 | test-dev |        |        |        |        |        |        |        |        |        |        |        |        |
| cascade stage3 | test-dev | 38.5% | 57.2% | 42.7% | 20.9% | 40.7% | 49.1% |        |        |        |        |        |        |
| cascade stage 1~2 | test-dev |     |        |        |        |        |        |        |        |        |        |        |        |
| cascade stage 1~3 | test-dev |     |        |        |        |        |        |        |        |        |        |        |        |



## mask cascade rcnn results beta version 4
### (use cls_agnostic_bbox_reg、specific lr_mult)
### model is trained on coco2017train
| experiments | dataset | box_ap | box_ap50 | box_ap75 | box_ap_small | box_ap_med | box_ap_large | mask_ap | mask_ap50 | mask_ap75 | mask_ap_small | mask_ap_med | mask_ap_large |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| mask-R50 | test-dev(val) | 38.00%(37.7%) | 59.7% | 41.3% | 21.2% | 40.2% | 48.1% | 34.20%(33.9%) | 56.4% | 36.0% | 14.8% | 36.0% | 49.7% |
| cascade stage1 | test-dev | 36.8% | 58.1% | 40.0% | 20.3% | 39.0% | 47.2% | 33.5% | 54.9% | 35.4% | 14.3% | 35.2% | 48.2% |
| cascade stage2 | test-dev | 38.9% | 58.6% | 42.8% | 21.0% | 40.9% | 50.5% | 34.4% | 55.6% | 36.6% | 14.5% | 36.0% | 50.2% |
| cascade stage3 | test-dev | 38.9% | 57.4% | 43.1% | 20.8% | 40.8% | 51.0% | 34.3% | 54.7% | 36.7% | 14.4% | 35.8% | 50.0% |
| cascade stage 1~2 | test-dev | 38.9% | 59.0% | 42.7% | 21.3% | 41.0% | 50.5% | 34.4% | 55.8% | 36.5% | 14.6% | 36.0% | 50.3% |
| cascade stage 1~3 | test-dev(val) | 39.50%(39.14%) | 58.90%(58.36%) | 43.40%(42.85%) | 21.50%(21.41%) | 41.40%(41.52%) | 51.30%(53.03%) | 34.60%(34.37%) | 55.80%(55.22%) | 36.80%(36.57%) | 14.80%(15.17%) | 36.20%(36.5%) | 50.40%(52.09%) |


## mask cascade rcnn results beta version 4 large iter
### model is trained on coco2017train, learning rate start at 0.01, reduce to 0.001 at 160000 iterations and 0.0001 at 240000 iterations
| experiments | dataset | box_ap | box_ap50 | box_ap75 | box_ap_small | box_ap_med | box_ap_large | mask_ap | mask_ap50 | mask_ap75 | mask_ap_small | mask_ap_med | mask_ap_large |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| cascade stage 1~3 | test-dev(val) | 40.10%(39.75%) | 59.40%(58.91%) | 43.90%(43.56%) | 22.00%(21.78%) | 41.90%(42.13%) | 51.90%(54.24%) | 35.00%(34.73%) | 56.30%(55.82%) | 37.20%(36.90%) | 15.10%(14.85%) | 36.60%(36.93%) | 51.00%(53.20%) |


## faster cascade rcnn results
### model is trained on coco2017train
| experiments | dataset | box_ap | box_ap50 | box_ap75 | box_ap_small | box_ap_med | box_ap_large |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| faster-R50 | test-dev(val) | (36.7%) | (58.45%) | (39.61%) | (21.12%) | (39.85%) | (48.13%) |
| cascade stage1 | test-dev(val) |  |  |  |  |  |  |
| cascade stage2 | test-dev(val) |  |  |  |  |  |  |
| cascade stage3 | test-dev(val) |  |  |  |  |  |  |
| cascade stage 1~2 | test-dev(val) |  |  |  |  |  |  |
| cascade stage 1~3 | test-dev(val) | (37.31%) | (55.51%) | (40.65%) | (20.30%) | (39.87%) | (49.21%) |


# PASCAL VOC experiments
## model is trained on voc0712 trainval, tested on voc2007 test, using coco evaluation metrics
| experiments | dataset | box_ap | box_ap50 | box_ap75 | box_ap_small | box_ap_med | box_ap_large |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| faster-R50 | voc2007_val | 46.75% | 77.06% | 50.32% | 16.54% | 35.10% | 54.36% |
| cascade stage1 | voc2007_test |  |  |  |  |  |  |
| cascade stage2 | voc2007_test | 46.61% | 74.41% | 50.68% | 16.44% | 33.90% | 54.52% |
| cascade stage3 | voc2007_test | 47.50% | 73.03% | 52.19% | 15.93% | 34.66% | 55.38% |
| cascade stage 1~2 | voc2007_test | 47.20% | 75.40% | 51.35% | 16.45% | 34.68% | 55.06% |
| cascade stage 1~3 | voc2007_test | 48.75% | 75.40% | 53.24% | 16.93% | 35.56% | 56.82% |

