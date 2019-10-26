import os
import glob
import pickle
import numpy as np
from PIL import Image
from alisuretool.Tools import Tools


"""
首先需要将val数据集转换成train数据集的格式.
"""


def get_ordered_folders_and_label_dict(map_file='map_clsloc.txt'):
    folders = []
    d = {}
    with open(map_file) as f:
        for line in f:
            tok = line.split()
            folders.append(tok[0])
            d[tok[0]] = int(tok[1])
            pass
        pass
    return folders, d


def _tran_images_files(images_files, out_file, alg, size):
    if os.path.exists(out_file):
        Tools.print("Exist {}".format(out_file))
        return [1000]*len(images_files)

    datas = []
    labels = []

    _, label_dict = get_ordered_folders_and_label_dict()
    for index, images_file in enumerate(images_files):
        try:
            if index % 10000 == 0:
                Tools.print("{}/{}".format(index, len(images_files)))

            folder = os.path.basename(os.path.split(images_file)[0])
            label = label_dict[folder]
            im = Image.open(images_file)
            if im.mode != "RGB":
                im = im.convert(mode="RGB")
            img = im.resize((size, size), alg)
            img = np.asarray(img)
            r = img[:, :, 0].flatten()
            g = img[:, :, 1].flatten()
            b = img[:, :, 2].flatten()

            arr = np.array(list(r) + list(g) + list(b), dtype=np.uint8)
            datas.append(arr)
            labels.append(label)
        except Exception:
            Tools.print('Cant process image %s' % images_file)
            with open("log_img2np.txt", "a") as f:
                f.write("Couldn't read: {}\n".format(images_file))
            continue
        pass

    y = labels
    x = np.row_stack(datas)
    x_mean = np.mean(x, axis=0)

    Tools.print('Label: %d: %s ' % (len(y), len(x)))
    d = {'data': x, 'labels': y, 'mean': x_mean}
    pickle.dump(d, open(out_file, 'wb'))

    return d['labels']


def tran_val_img(in_dir, out_dir, alg, size):
    images_files_all = glob.glob(os.path.join(in_dir, "*/*.*"))
    out_file = os.path.join(out_dir, "val_data")
    y_test = _tran_images_files(images_files_all, out_file, alg, size)
    return y_test


def tran_train_img(in_dir, out_dir, alg, size):
    images_files_all = glob.glob(os.path.join(in_dir, "*/*.*"))
    # np.random.shuffle(images_files_all)

    npf = len(images_files_all) // 10
    images_files_list = np.split(np.asarray(images_files_all), [npf, 2*npf, 3*npf, 4*npf,
                                                                5*npf, 6*npf, 7*npf, 8*npf, 9*npf])
    y_test = []
    for index, images_files in enumerate(images_files_list):
        out_file = os.path.join(out_dir, 'train_data_batch_%d' % index)
        _y_test = _tran_images_files(images_files, out_file, alg, size)
        y_test.extend(_y_test)
        pass
    return y_test


def tran_image_net(size=16, split="val_new", algorithm="box", out_dir=None,
                   root_dir="/media/ubuntu/ALISURE/data/DATASET/ILSVRC2015/Data/CLS-LOC"):

    _alg_dict = {'lanczos': Image.LANCZOS, 'nearest': Image.NEAREST, 'bilinear': Image.BILINEAR,
                 'bicubic': Image.BICUBIC, 'hamming': Image.HAMMING, 'box': Image.BOX}

    in_dir = os.path.join(root_dir, split)
    out_dir = os.path.join(root_dir, "{}_{}".format(split, size)) if out_dir is None else out_dir
    current_out_dir = Tools.new_dir(os.path.join(out_dir, algorithm))
    if "train" in split:
        y_test = tran_train_img(in_dir=in_dir, out_dir=current_out_dir, alg=_alg_dict[algorithm], size=size)
    else:
        y_test = tran_val_img(in_dir=in_dir, out_dir=current_out_dir, alg=_alg_dict[algorithm], size=size)
        pass

    count = np.zeros([1001])
    for i in y_test:
        count[i-1] += 1
    for i in range(1001):
        Tools.print('%d : %d' % (i, count[i]))
    Tools.print('SUM: %d' % len(y_test))

    Tools.print("Finished.")
    pass


if __name__ == '__main__':
    tran_image_net(size=96, split="train", algorithm="box",
                   out_dir="/home/ubuntu/Desktop/ALISURE/DownSampledImageNet/train_96")
    # tran_image_net(size=96, split="val_new", algorithm="box",
    #                out_dir="/home/ubuntu/Desktop/ALISURE/DownSampledImageNet/val_new_96")
    pass
