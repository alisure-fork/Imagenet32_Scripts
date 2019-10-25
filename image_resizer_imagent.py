import os
from PIL import Image
import multiprocessing
from alisuretool.Tools import Tools

alg_dict = {
    'lanczos': Image.LANCZOS,
    'nearest': Image.NEAREST,
    'bilinear': Image.BILINEAR,
    'bicubic': Image.BICUBIC,
    'hamming': Image.HAMMING,
    'box': Image.BOX
}


def resize_img_folder(i, in_dir, out_dir, alg, size):
    Tools.print('Folder {} {}'.format(i, in_dir))
    Tools.new_dir(out_dir)
    for filename in os.listdir(in_dir):
        try:
            im = Image.open(os.path.join(in_dir, filename))
            if im.mode != "RGB":
                im = im.convert(mode="RGB")
            im_resized = im.resize((size, size), alg)
            filename = os.path.splitext(filename)[0]
            im_resized.save(os.path.join(out_dir, filename + '.png'))
        except OSError as err:
            Tools.print("This file couldn't be read as an image")
            with open("log.txt", "a") as f:
                f.write("Couldn't resize: %s" % os.path.join(in_dir, filename))
            pass
        pass

    pass


if __name__ == '__main__':
    size = 16
    root_dir = "/media/ubuntu/ALISURE/data/DATASET/ILSVRC2015/Data/CLS-LOC"
    in_dir = os.path.join(root_dir, "train")
    out_dir = os.path.join(root_dir, "train_{}".format(size))
    # in_dir = os.path.join(root_dir, "val_new")
    # out_dir = os.path.join(root_dir, "val_new_{}".format(size))
    algorithm = "box"
    recurrent = True
    current_out_dir = os.path.join(out_dir, algorithm)
    if recurrent:
        folders = [_dir for _dir in sorted(os.listdir(in_dir)) if os.path.isdir(os.path.join(in_dir, _dir))]
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
        for i, folder in enumerate(folders):
            r = pool.apply_async(func=resize_img_folder, args=[
                i, os.path.join(in_dir, folder), os.path.join(current_out_dir, folder), alg_dict[algorithm], size])
            pass
        pool.close()
        pool.join()
    else:
        Tools.print('For folder %s' % in_dir)
        resize_img_folder(i=0, in_dir=in_dir, out_dir=current_out_dir, alg=alg_dict[algorithm], size=size)
        pass

    Tools.print("Finished.")
