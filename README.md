# detectron-cascadee-exp
experiment results of implement Cascade RCNN under Detectron

## mask iterative bbox rcnn results
### model is trained on coco2017train + val
| experiments | dataset | box_ap | box_ap50 | box_ap75 | box_ap_small | box_ap_medium | box_ap_large | mask_ap | mask_ap50 | mask_ap75 | mask_ap_small | mask_ap_medium | mask_ap_large |
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
| experiments | dataset | box_ap | box_ap50 | box_ap75 | box_ap_small | box_ap_medium | box_ap_large | mask_ap | mask_ap50 | mask_ap75 | mask_ap_small | mask_ap_medium | mask_ap_large |
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
| experiments | dataset | box_ap | box_ap50 | box_ap75 | box_ap_small | box_ap_medium | box_ap_large | mask_ap | mask_ap50 | mask_ap75 | mask_ap_small | mask_ap_medium | mask_ap_large |
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
| experiments | dataset | box_ap | box_ap50 | box_ap75 | box_ap_small | box_ap_medium | box_ap_large | mask_ap | mask_ap50 | mask_ap75 | mask_ap_small | mask_ap_medium | mask_ap_large |
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
| experiments | dataset | box_ap | box_ap50 | box_ap75 | box_ap_small | box_ap_medium | box_ap_large | mask_ap | mask_ap50 | mask_ap75 | mask_ap_small | mask_ap_medium | mask_ap_large |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| mask-R50 | test-dev(val) | 38.0%(37.7%) | 59.7% | 41.3% | 21.2% | 40.2% | 48.1% | 34.2%(33.9%) | 56.4% | 36.0% | 14.8% | 36.0% | 9.7% |
| cascade stage1 | test-dev | 36.8% | 58.1% | 40.0% | 20.3% | 39.0% | 47.2% | 33.5% | 54.9% | 35.4% | 14.3% | 35.2% | 48.2% |
| cascade stage2 | test-dev | 38.9% | 58.6% | 42.8% | 21.0% | 40.9% | 50.5% | 34.4% | 55.6% | 36.6% | 14.5% | 36.0% | 50.2% |
| cascade stage3 | test-dev | 38.9% | 57.4% | 43.1% | 20.8% | 40.8% | 51.0% | 34.3% | 54.7% | 36.7% | 14.4% | 35.8% | 50.0% |
| cascade stage 1~2 | test-dev | 38.9% | 59.0% | 42.7% | 21.3% | 41.0% | 50.5% | 34.4% | 55.8% | 36.5% | 14.6% | 36.0% | 50.3% |
| cascade stage 1~3 | test-dev(val) | 39.5%(39.14%) | 58.9%(58.36%) | 43.4%(42.85%) | 21.5%(21.41%) | 41.4%(41.52%) | 51.3%(53.03%) | 34.6%(34.37%) | 55.8%(55.22%) | 36.8%(36.57%) | 14.8%(15.17%) | 36.2%(36.5%) | 50.4%(52.09%) |


## faster cascade rcnn results
### model is trained on coco2017train
| experiments | dataset | box_ap | box_ap50 | box_ap75 | box_ap_small | box_ap_medium | box_ap_large | mask_ap | mask_ap50 | mask_ap75 | mask_ap_small | mask_ap_medium | mask_ap_large |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| mask-R50 | test-dev(val) |  |  |  |  |  |  |  |  |  |  |  |  |
| cascade stage1 | test-dev |  |  |  |  |  |  |  |  |  |  |  |  |
| cascade stage2 | test-dev |  |  |  |  |  |  |  |  |  |  |  |  |
| cascade stage3 | test-dev |  |  |  |  |  |  |  |  |  |  |  |  |
| cascade stage 1~2 | test-dev |  |  |  |  |  |  |  |  |  |  |  |  |
| cascade stage 1~3 | test-dev(val) |  |  |  |  |  |  |  |  |  |  |  |  |


## mask cascade rcnn results beta version 4 large iter
### model is trained on coco2017train, lr start at 0.01, reduce to 0.001 at 160000 iterations and 0.0001 at 240000 iterations
| experiments | dataset | box_ap | box_ap50 | box_ap75 | box_ap_small | box_ap_medium | box_ap_large | mask_ap | mask_ap50 | mask_ap75 | mask_ap_small | mask_ap_medium | mask_ap_large |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| cascade stage 1~3 | test-dev(val) | (39.75%) | (58.91%) | (43.56%) | (21.78%) | (42.13%) | (54.24%) | 35.0%(34.73%) | 56.3%(55.82%) | 37.2%(36.90%) | 15.1%(14.85%) | 36.6%(36.93%) | 51.0%(53.20%) |
