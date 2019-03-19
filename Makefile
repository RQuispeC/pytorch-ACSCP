all:
	python3 train.py -d ucf-cc-50 --gt-mode same --people-thr 20 --train-batch 16 --units ucf-fold1 --save-dir log/ACSCP