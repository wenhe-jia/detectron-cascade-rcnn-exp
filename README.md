# detectron-cascadee-exp
experiment results of implement Cascade RCNN under Detectron

## iterative bbox
model is trained on coco2017train + val
| experiments | dataset | box_ap | box_ap50 | box_ap75 | box_ap_small | box_ap_medium | box_ap_large | mask_ap | mask_ap50 | mask_ap75 | mask_ap_small | mask_ap_medium | mask_ap_large |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| mask-R50       | test-dev | 38.2% | 60.05 | 41.5% | 21.8% | 40.3% | 48.4% | 34.3% | 56.5% | 36.3% | 14.9% | 36.1% | 49.7% |
| cascade stage1 | test-dev | 38.3% |       |       |       |       |       | 34.2% |       |       |       |       |       |
| cascade stage2 | test-dev | 38.9% |       |       |       |       |       | 34.1% |       |       |       |       |       |
| cascade stage3 | test-dev | 38.9% | 59.5% | 42.1% | 21.5% | 40.7% | 50.2% | 34.0% | 56.1% | 35.9% | 14.8% | 35.5% | 49.5% |
| cascade stage 1~2 | test-dev |    |       |       |       |       |       |       |       |       |       |       |       |
| cascade stage 1~3 | test-dev |    |       |       |       |       |       |       |       |       |       |       |       |


## cascade rcnn results beta version 1 (clip bbox and add invalid bbox check in DecodeBBoxOp)
model is trained on coco2017train + val
| experiments | dataset | box_ap | box_ap50 | box_ap75 | box_ap_small | box_ap_medium | box_ap_large | mask_ap | mask_ap50 | mask_ap75 | mask_ap_small | mask_ap_medium | mask_ap_large |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| mask-R50       | test-dev | 38.2% | 60.05 | 41.5% | 21.8% | 40.3% | 48.4% | 34.3% | 56.5% | 36.3% | 14.9% | 36.1% | 49.7% |
| cascade stage1 | test-dev | 38.2% | 59.9% | 41.7% | 21.7% | 40.4% | 48.4% | 34.2% | 56.4% | 36.1% | 15.0% | 36.0% | 49.5% |
| cascade stage2 | test-dev | 38.1% | 58.5% | 41.3% | 18.2% | 39.7% | 53.4% | 34.7% | 56.5% | 36.8% | 15.1% | 36.5% | 50.6% |
| cascade stage3 | test-dev | 39.4% | 57.5% | 43.5% | 21.4% | 41.2% | 51.1% | 34.2% | 55.0% | 36.4% | 14.7% | 35.9% | 49.9% |
| cascade stage 1~2 | test-dev |    |       |       |       |       |       |       |       |       |       |       |       |
| cascade stage 1~3 | test-dev |    |       |       |       |       |       |       |       |       |       |       |       |


## cascade rcnn results beta version 2（screen out high iou boxes in DecodeBBoxOp)
model is trained on coco2017train
| experiments | dataset | box_ap | box_ap50 | box_ap75 | box_ap_small | box_ap_medium | box_ap_large | mask_ap | mask_ap50 | mask_ap75 | mask_ap_small | mask_ap_medium | mask_ap_large |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| mask-R50       | test-dev | 38.6%  |        |        |        |        |        | 34.5%  |        |        |        |        |        |
| cascade stage1 | test-dev |        |        |        |        |        |        |        |        |        |        |        |        |
| cascade stage2 | test-dev |        |        |        |        |        |        |        |        |        |        |        |        |
| cascade stage3 | test-dev | 39.06% | 56.98% | 43.28% | 21.86% | 41.54% | 52.41% | 34.20% | 54.47% | 36.65% | 15.11% | 36.47% | 51.51% | 
| cascade stage 1~2 | test-dev |     |        |        |        |        |        |        |        |        |        |        |        |
| cascade stage 1~3 | test-dev |     |        |        |        |        |        |        |        |        |        |        |        |


## cascade rcnn results beta version 3 (add weight to rcnn loss)
model is trained on coco2017train
| experiments | dataset | box_ap | box_ap50 | box_ap75 | box_ap_small | box_ap_medium | box_ap_large | mask_ap | mask_ap50 | mask_ap75 | mask_ap_small | mask_ap_medium | mask_ap_large |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| mask-R50       | test-dev | 38.6%  |        |        |        |        |        | 34.5%  |        |        |        |        |        |
| cascade stage1 | test-dev |        |        |        |        |        |        |        |        |        |        |        |        |
| cascade stage2 | test-dev |        |        |        |        |        |        |        |        |        |        |        |        |
| cascade stage3 | test-dev |        |        |        |        |        |        |        |        |        |        |        |        |
| cascade stage 1~2 | test-dev |     |        |        |        |        |        |        |        |        |        |        |        |
| cascade stage 1~3 | test-dev |     |        |        |        |        |        |        |        |        |        |        |        |


