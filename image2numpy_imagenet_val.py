import os
import glob
import utils
import pickle
import imageio
import numpy as np
from alisuretool.Tools import Tools


def process_folder(all_png, out_dir):
    label_dict = utils.get_label_dict()
    folders = utils.get_ordered_folders()
    val_ground_dict = utils.get_val_ground_dict()

    labels_searched = []
    for folder in folders:
        labels_searched.append(label_dict[folder])
    labels_list = []
    images = []
    for image_index, image_name in enumerate(all_png):
        if image_index % 1000 == 0:
            Tools.print("{} {}".format(image_index, len(all_png)))

        basename = os.path.basename(image_name)
        label = val_ground_dict[basename[:-4]]
        if label not in labels_searched:
            continue
        try:
            img = imageio.imread(image_name)
            r = img[:, :, 0].flatten()
            g = img[:, :, 1].flatten()
            b = img[:, :, 2].flatten()
        except:
            Tools.print('Cant process image {}'.format(basename))
            with open("log_img2np_val.txt", "a") as f:
                f.write("Couldn't read: {}".format(image_name))
            continue
        arr = np.array(list(r) + list(g) + list(b), dtype=np.uint8)
        images.append(arr)
        labels_list.append(label)
        pass

    data_val = np.row_stack(images)

    # Can add some kind of data splitting
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


if __name__ == '__main__':
    size = 64
    algorithm = "box"
    root_dir = "D:\\data\\DATASET\\ILSVRC2015\\Data\\CLS-LOC"
    out_dir = os.path.join(root_dir, "val_new_{}_out".format(size), algorithm)
    Tools.print("Start program ...")

    all_png = glob.glob(os.path.join(root_dir, "val_new_{}".format(size), algorithm, "*/*.png"))
    process_folder(all_png=all_png, out_dir=out_dir)

    Tools.print("Finished.")
