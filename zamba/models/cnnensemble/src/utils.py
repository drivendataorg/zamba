import math
import pickle
import time
import random
from contextlib import contextmanager
import concurrent.futures
from queue import Queue

import skimage.io
import skimage.transform
from skimage.transform import SimilarityTransform, AffineTransform
import numpy as np
import matplotlib.pyplot as plt


def crop_edge(img, x, y, w, h, mode='edge'):
    img_w = img.shape[1]
    img_h = img.shape[0]

    if x >= 0 and y >= 0 and x + w <= img_w and y + h < img_h:
        return img[int(y):int(y + h), int(x):int(x + w)].astype('float32') / 255.0

    tform = SimilarityTransform(translation=(x, y))
    return skimage.transform.warp(img, tform, mode=mode, output_shape=(h, w))


def preprocessed_input_to_img_resnet(x):
    # Zero-center by mean pixel
    x = x.copy()
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # 'BGR' -> RGB
    img = x.copy()
    img[:, :, 0] = x[:, :, 2]
    img[:, :, 1] = x[:, :, 1]
    img[:, :, 2] = x[:, :, 0]
    return img / 255.0


@contextmanager
def timeit_context(name):
    startTime = time.time()
    yield
    elapsedTime = time.time() - startTime
    print('[{}] finished in {} ms'.format(name, int(elapsedTime * 1000)))


def chunks(l, n, add_empty=False):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l) // n * n + n - 1, n):
        if len(l[i:i + n]):
            yield l[i:i + n]
    if add_empty:
        yield []


def load_data(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data


def save_data(data, file_name):
    pickle.dump(data, open(file_name, 'wb'))


def lock_layers_until(model, first_trainable_layer, verbose=False):
    found_first_layer = False
    for layer in model.layers:
        if layer.name == first_trainable_layer:
            found_first_layer = True

        if verbose and found_first_layer and not layer.trainable:
            print('Make layer trainable:', layer.name)
            layer.trainable = True

        layer.trainable = found_first_layer


def rand_or_05():
    if random.random() > 0.5:
        return random.random()
    return 0.5


def rand_scale_log_normal(mean_scale, one_sigma_at_scale):
    """
    Generate a distribution of value at log  scale around mean_scale

    :param mean_scale:  
    :param one_sigma_at_scale: 67% of values between  mean_scale/one_sigma_at_scale .. mean_scale*one_sigma_at_scale
    :return: 
    """

    log_sigma = math.log(one_sigma_at_scale)
    return mean_scale*math.exp(random.normalvariate(0.0, log_sigma))


def print_stats(title, array):
    print('{} shape:{} dtype:{} min:{} max:{} mean:{} median:{}'.format(
        title,
        array.shape,
        array.dtype,
        np.min(array),
        np.max(array),
        np.mean(array),
        np.median(array)
    ))


def limit_mem(K):
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))


def parallel_generator(orig_gen, executor):
    queue = Queue(maxsize=8)

    def bg_task():
        for i in orig_gen:
            # print('bg_task', i)
            queue.put(i)
        # print('bg_task', None)
        queue.put(None)

    task = executor.submit(bg_task)
    while True:
        value = queue.get()
        if value is not None:
            yield value
            queue.task_done()
        else:
            queue.task_done()
            break
    task.result()


def parallel_generator_nthread(orig_gen, executor):
    queue = Queue(maxsize=8)

    def bg_task():
        for i in orig_gen:
            # print('bg_task', i)
            queue.put(i)
        # print('bg_task', None)
        queue.put(None)

    task = executor.submit(bg_task)
    while True:
        value = queue.get()
        if value is not None:
            yield value
            queue.task_done()
        else:
            queue.task_done()
            break
    task.result()


def test_parallel_generator():
    def task(i):
        time.sleep(0.1)
        print('task', i)
        return i

    def orig_gen(n):
        for i in range(n):
            yield task(i)

    res_orig = []
    with timeit_context('orig gen'):
        for i in orig_gen(5):
            time.sleep(0.1)
            res_orig.append(i)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    res_par = []
    with timeit_context('parallel gen'):
        for i in parallel_generator(orig_gen(5), executor):
            time.sleep(0.1)
            res_par.append(i)

    assert res_orig == res_par


if __name__ == '__main__':
    pass
    test_parallel_generator()
    # test_chunks()
    #
    # img = skimage.io.imread(Path(__file__).parent.parent / 'train/ALB/img_00003.jpg')
    # print(img.shape)
    #
    # with timeit_context('Generate crops'):
    #     crop_edge(img, 10, 10, 400, 400)
    #
    # import matplotlib.pyplot as plt
    #
    # plt.figure(1)
    # plt.subplot(221)
    # plt.imshow(img)
    # plt.subplot(222)
    # plt.imshow(crop_edge(img, 1280-200, 720-200, 400, 400, mode='edge'))
    # plt.subplot(223)
    # plt.imshow(crop_edge(img, 1280 - 200, 720 - 200, 400, 400, mode='wrap'))
    # plt.subplot(224)
    # plt.imshow(crop_edge(img, 1280 - 200, 720 - 200, 400, 400, mode='reflect'))
    #
    # plt.show()