train-ucf-fold1:
	python3 train.py -d ucf-cc-50 --gt-mode same --people-thr 20 --train-batch 24 --units ucf-fold1 --save-dir log/ACSCP 
train-ucf-fold2:
	python3 train.py -d ucf-cc-50 --gt-mode same --people-thr 20 --train-batch 24 --units ucf-fold2 --save-dir log/ACSCP
train-ucf-fold3:
	python3 train.py -d ucf-cc-50 --gt-mode same --people-thr 20 --train-batch 24 --units ucf-fold3 --save-dir log/ACSCP
train-ucf-fold4:
	python3 train.py -d ucf-cc-50 --gt-mode same --people-thr 20 --train-batch 24 --units ucf-fold4 --save-dir log/ACSCP
train-ucf-fold5:
	python3 train.py -d ucf-cc-50 --gt-mode same --people-thr 20 --train-batch 24 --units ucf-fold5 --save-dir log/ACSCP
test-ucf-fold1:
	python3 train.py -d ucf-cc-50 --gt-mode same --people-thr 20 --train-batch 16 --units ucf-fold1 --save-dir log/ACSCP --evaluate-only --resume log/ACSCP/ucf-cc-50_people_thr_20_gt_mode_same/ucf-fold1/best_model.h5 --save-plots 
test-ucf-fold2:
	python3 train.py -d ucf-cc-50 --gt-mode same --people-thr 20 --train-batch 16 --units ucf-fold2 --save-dir log/ACSCP --evaluate-only --resume log/ACSCP/ucf-cc-50_people_thr_20_gt_mode_same/ucf-fold2/best_model.h5 --save-plots --overlap-test
test-ucf-fold3:
	python3 train.py -d ucf-cc-50 --gt-mode same --people-thr 20 --train-batch 16 --units ucf-fold3 --save-dir log/ACSCP --evaluate-only --resume log/ACSCP/ucf-cc-50_people_thr_20_gt_mode_same/ucf-fold3/best_model.h5 --save-plots --overlap-test
test-ucf-fold4:
	python3 train.py -d ucf-cc-50 --gt-mode same --people-thr 20 --train-batch 16 --units ucf-fold4 --save-dir log/ACSCP --evaluate-only --resume log/ACSCP/ucf-cc-50_people_thr_20_gt_mode_same/ucf-fold4/best_model.h5 --save-plots --overlap-test
test-ucf-fold5:
	python3 train.py -d ucf-cc-50 --gt-mode same --people-thr 20 --train-batch 16 --units ucf-fold5 --save-dir log/ACSCP --evaluate-only --resume log/ACSCP/ucf-cc-50_people_thr_20_gt_mode_same/ucf-fold5/best_model.h5 --save-plots --overlap-test
