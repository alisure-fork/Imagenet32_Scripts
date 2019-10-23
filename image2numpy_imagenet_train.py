"""
http://stackoverflow.com/questions/35032675/how-to-create-dataset-similar-to-cifar-10/35034287
"""

import os
import utils
import pickle
import imageio
import numpy as np
from alisuretool.Tools import Tools


def process_folder(in_dir, out_dir):
    label_dict = utils.get_label_dict()
    folders = utils.get_ordered_folders()
    data_list_train = []
    labels_list_train = []
    for folder in folders:
        label = label_dict[folder]
        Tools.print("Processing images from folder %s as label %d" % (folder, label))
        images = []
        for image_name in os.listdir(os.path.join(in_dir, folder)):
            try:
                img = imageio.imread(os.path.join(in_dir, folder, image_name))
                r = img[:, :, 0].flatten()
                g = img[:, :, 1].flatten()
                b = img[:, :, 2].flatten()
            except:
                Tools.print('Cant process image %s' % image_name)
                with open("log_img2np.txt", "a") as f:
                    f.write("Couldn't read: {} \n".format(os.path.join(in_dir, folder, image_name)))
                continue
            arr = np.array(list(r) + list(g) + list(b), dtype=np.uint8)
            images.append(arr)

        data = np.row_stack(images)
        samples_num = data.shape[0]
        labels = [label] * samples_num

        labels_list_train.extend(labels)
        data_list_train.append(data)

        Tools.print('Label: %d: %s has %d samples' % (label, folder, samples_num))

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


if __name__ == '__main__':
    size = 64
    algorithm = "box"
    root_dir = "/media/ubuntu/ALISURE/data/DATASET/ILSVRC2015/Data/CLS-LOC"
    in_dir = os.path.join(root_dir, "train_{}".format(size), algorithm)
    out_dir = os.path.join(root_dir, "train_{}_out".format(size), algorithm)

    Tools.print("Start program ...")
    process_folder(in_dir=in_dir, out_dir=out_dir)
    Tools.print("Finished.")
