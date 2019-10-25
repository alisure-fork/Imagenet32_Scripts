import os
import glob
import pickle
import numpy as np
from PIL import Image
from alisuretool.Tools import Tools


def get_val_ground_dict(val_names_file='val.txt', val_labels_file='ILSVRC2015_clsloc_validation_ground_truth.txt'):
    # Table would be better? but keep dict
    d_labels = {}
    i = 1
    with open(val_labels_file) as f:
        for line in f:
            tok = line.split()
            d_labels[i] = int(tok[0])
            i += 1

    d = {}
    with open(val_names_file) as f:
        for line in f:
            tok = line.split()
            d[tok[0]] = d_labels[int(tok[1])]
    return d


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


def tran_val_img(in_dir, out_dir, alg, size):
    folders, label_dict = get_ordered_folders_and_label_dict()
    val_ground_dict = get_val_ground_dict()

    labels_searched = []
    for folder in folders:
        labels_searched.append(label_dict[folder])
        pass

    labels_list = []
    images = []
    all_png = glob.glob(os.path.join(in_dir, "*/*.*"))
    for image_index, image_name in enumerate(all_png):
        if image_index % 10000 == 0:
            Tools.print("{} {}".format(image_index, len(all_png)))

        basename = os.path.basename(image_name)
        label = val_ground_dict[basename[:-5]]
        if label not in labels_searched:
            continue
        try:
            im = Image.open(image_name)
            if im.mode != "RGB":
                im = im.convert(mode="RGB")
            img = im.resize((size, size), alg)
            img = np.asarray(img)
            r = img[:, :, 0].flatten()
            g = img[:, :, 1].flatten()
            b = img[:, :, 2].flatten()
        except Exception:
            Tools.print('Cant process image {}'.format(basename))
            with open("log_img2np_val.txt", "a") as f:
                f.write("Couldn't read: {}\n".format(image_name))
            continue
        arr = np.array(list(r) + list(g) + list(b), dtype=np.uint8)
        images.append(arr)
        labels_list.append(label)
        pass

    data_val = np.row_stack(images)

    d_val = {'data': data_val, 'labels': labels_list}
    Tools.new_dir(out_dir)
    pickle.dump(d_val, open(os.path.join(out_dir, 'val_data'), 'wb'))

    y_test = d_val['labels']
    count = np.zeros([1000])

    for i in y_test:
        count[i-1] += 1
    for i in range(1000):
        Tools.print('%d : %d' % (i, count[i]))
    Tools.print('SUM: %d' % len(y_test))
    pass


def tran_train_img(in_dir, out_dir, alg, size):
    folders, label_dict = get_ordered_folders_and_label_dict()

    data_list_train = []
    labels_list_train = []
    for index, folder in enumerate(folders):
        Tools.print("{}/{}".format(index, len(folders)))

        label = label_dict[folder]
        Tools.print("Processing images from folder %s as label %d" % (folder, label))
        images = []
        for image_name in os.listdir(os.path.join(in_dir, folder)):
            try:
                im = Image.open(os.path.join(in_dir, folder, image_name))
                if im.mode != "RGB":
                    im = im.convert(mode="RGB")
                img = im.resize((size, size), alg)
                img = np.asarray(img)
                r = img[:, :, 0].flatten()
                g = img[:, :, 1].flatten()
                b = img[:, :, 2].flatten()
            except Exception:
                Tools.print('Cant process image %s' % image_name)
                with open("log_img2np.txt", "a") as f:
                    f.write("Couldn't read: {}\n".format(image_name))
                continue
            arr = np.array(list(r) + list(g) + list(b), dtype=np.uint8)
            images.append(arr)
            pass

        data = np.row_stack(images)
        samples_num = data.shape[0]
        labels = [label] * samples_num

        labels_list_train.extend(labels)
        data_list_train.append(data)

        Tools.print('Label: %d: %s has %d samples' % (label, folder, samples_num))
        pass

    x = np.concatenate(data_list_train, axis=0)
    y = labels_list_train
    x_mean = np.mean(x, axis=0)
    train_indices = np.arange(x.shape[0])
    np.random.shuffle(train_indices)
    curr_index = 0
    size = x.shape[0] // 10

    y_test = []
    Tools.new_dir(out_dir)
    for i in range(1, 10):
        d = {'data': x[train_indices[curr_index: (curr_index + size)], :],
             'labels': np.array(y)[train_indices[curr_index: (curr_index + size)]].tolist(), 'mean': x_mean}
        pickle.dump(d, open(os.path.join(out_dir, 'train_data_batch_%d' % i), 'wb'))
        curr_index += size
        y_test.extend(d['labels'])
        pass

    d = {'data': x[train_indices[curr_index:], :],
         'labels': np.array(y)[train_indices[curr_index:]].tolist(), 'mean': x_mean}
    pickle.dump(d, open(os.path.join(out_dir, 'train_data_batch_10'), 'wb'))
    y_test.extend(d['labels'])

    count = np.zeros([1000])
    for i in y_test:
        count[i-1] += 1
    for i in range(1000):
        Tools.print('%d : %d' % (i, count[i]))
    Tools.print('SUM: %d' % len(y_test))
    pass


def tran_image_net(size=16, split="val_new", algorithm="box",
                   root_dir="/media/ubuntu/ALISURE/data/DATASET/ILSVRC2015/Data/CLS-LOC"):
    in_dir = os.path.join(root_dir, split)
    out_dir = os.path.join(root_dir, "{}_{}".format(split, size))
    current_out_dir = os.path.join(out_dir, algorithm)

    _alg_dict = {'lanczos': Image.LANCZOS, 'nearest': Image.NEAREST, 'bilinear': Image.BILINEAR,
                 'bicubic': Image.BICUBIC, 'hamming': Image.HAMMING, 'box': Image.BOX}

    if "train" in split:
        tran_train_img(in_dir=in_dir, out_dir=current_out_dir, alg=_alg_dict[algorithm], size=size)
    else:
        tran_val_img(in_dir=in_dir, out_dir=current_out_dir, alg=_alg_dict[algorithm], size=size)
        pass
    Tools.print("Finished.")
    pass


if __name__ == '__main__':
    # tran_image_net(size=16, split="train", algorithm="box")
    tran_image_net(size=16, split="val_new", algorithm="box")
    pass
