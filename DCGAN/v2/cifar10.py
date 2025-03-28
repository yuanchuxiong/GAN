# -*- coding: utf-8 -*-
import numpy as np
import pickle
import imageio
import os

# 解压缩，返回解压后的字典
def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict


def extract_train_data(input_base_path, output_path):

    if os.path.exists(input_base_path) and os.path.exists(output_path):

        current_image_index = 0

        for file_index in range(5):

            file_seq_no = file_index + 1
            # data/cifar-10-python/cifar-10-batches-py/test_batch
            target_file = os.path.join(input_base_path, "data_batch_{}".format(file_seq_no))
            print("process target file: {}".format(target_file))

            # 保存测试集图片
            train = unpickle(target_file)
            for i in range(10000):
                current_image_index += 1
                img = np.reshape(train['data'][i], (3, 32, 32))
                img = img.transpose(1, 2, 0)
                label_output_path = os.path.join(output_path, str(train['labels'][i]))
                if os.path.exists(label_output_path):
                    pass
                else:
                    os.mkdir(label_output_path)
                picName = os.path.join(label_output_path, "{}.jpg".format(current_image_index))
                #, dpi=(600.0,600.0))
                imageio.imsave(picName, img)



def extract_test_data(input_base_path, output_path):

    if os.path.exists(input_base_path) and os.path.exists(output_path):

        # "data/cifar-10-python/cifar-10-batches-py/test_batch"
        target_file = os.path.join(input_base_path, "test_batch")
        # output_path = 'data/cifar-10'

        # 保存测试集图片
        test = unpickle(target_file)

        for i in range(1, 10000):

            img = np.reshape(test['data'][i], (3, 32, 32))
            img = img.transpose(1, 2, 0)
            picName = os.path.join(output_path, str(test['labels'][i]) + '_' + str(i) + '.jpg')
            #, dpi=(600.0,600.0))
            imageio.imsave(picName, img)


if __name__ == '__main__':

    extract_train_data("data/cifar-10-python/cifar-10-batches-py", "data/cifar-10/train")
    print("data_batch loaded.")

    extract_test_data("data/cifar-10-python/cifar-10-batches-py", "data/cifar-10/test")
    print("test_batch loaded.")

