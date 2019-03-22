train-ucf-fold1:
	python3 train.py -d ucf-cc-50 --gt-mode same --people-thr 20 --train-batch 16 --units ucf-fold1 --save-dir log/ACSCP
test-ucf-fold1:
	python3 train.py -d ucf-cc-50 --gt-mode same --people-thr 20 --train-batch 16 --units ucf-fold1 --save-dir log/ACSCP --evaluate-only --resume log/ACSCP/ucf-cc-50_people_thr_20_gt_mode_same/ucf-fold1/best_model.h5 --save-plots
