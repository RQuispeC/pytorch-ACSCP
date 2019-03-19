import os.path as osp
import os
from shutil import copyfile
import sys
import errno

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def join_json(final_data, data, index, size):
	x = 0
	y = 0
	if index == 1 or index == 3: 
		x = size
	if index == 2 or index == 3:
		y = size
	for i in range(len(data)):
		final_data.append({"x":data[i]['x'] + x,"y":data[i]['y'] + y})
	return final_data

def resize(data, scale):
	for i in range(len(data)):
		data[i]['x'] = data[i]['x'] / scale; data[i]['y'] = data[i]['y'] / scale;
	return data

def copy_to_directory(files_list, output_dir):
    for file in files_list:
        file_name = file.split('/')[-1]
        out_file_path = osp.join(output_dir, file_name)
        copyfile(file, out_file_path)

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def intersec(first, second):
    overlap = False
    overlap = overlap or (first[0] <= second[0] and second[0] <= first[2] and first[1] <= second[1] and second[1] <= first[3])
    overlap = overlap or (first[0] <= second[2] and second[2] <= first[2] and first[1] <= second[1] and second[1] <= first[3])
    overlap = overlap or (first[0] <= second[0] and second[0] <= first[2] and first[1] <= second[3] and second[3] <= first[3])
    overlap = overlap or (first[0] <= second[2] and second[2] <= first[2] and first[1] <= second[3] and second[3] <= first[3])
    return overlap

def cnt_overlaps(boxes):
    boxes_overlap = []
    id_overlap = []
    for ind_first, first in enumerate(boxes):
        cnt = 0
        overlap = []
        for ind_second, second in enumerate(boxes):
            if ind_first != ind_second and intersec(first, second):
                cnt += 1
                overlap.append(ind_second)
        boxes_overlap.append(cnt)
        id_overlap.append(overlap)
    return boxes_overlap, id_overlap