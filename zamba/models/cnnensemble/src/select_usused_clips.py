import numpy as np
import pandas as pd
import os
import shutil
import config
import matplotlib.pyplot as plt

predictions = [
    'resnet50_avg_2.csv',
    'inception_v3_1.csv',
    'inception_v3_2.csv',
    'xception_avg_1.csv',
    'xception_avg_2.csv'
]

data_dir = config.MODEL_DIR / 'output/prediction_unused_frames/'

classes = ['bird', 'blank', 'cattle', 'chimpanzee', 'elephant', 'forest buffalo',
           'gorilla', 'hippopotamus', 'human', 'hyena', 'large ungulate',
           'leopard', 'lion', 'other (non-primate)', 'other (primate)', 'pangolin',
           'porcupine', 'reptile', 'rodent', 'small antelope', 'small cat',
           'wild dog', 'duiker', 'hog']

classes_compatible = ['bird', 'blank', 'cattle', 'chimpanzee', 'elephant', 'forest buffalo',
                      'gorilla', 'hippopotamus', 'human', 'hyena', 'large ungulate',
                      'leopard', 'lion', 'other non-primate', 'other primate', 'pangolin',
                      'porcupine', 'reptile', 'rodent', 'small antelope', 'small cat',
                      'wild dog', 'duiker', 'hog']


def find_problematic_clips():
    datasets = [pd.read_csv(os.path.join(data_dir, fn)).sort_values(by=['filename']) for fn in predictions]

    ds0 = pd.merge(datasets[0], datasets[1], on='filename', suffixes=['0', '1'])
    ds1 = pd.merge(datasets[2], datasets[3], on='filename', suffixes=['2', '3'])
    ds2 = pd.merge(ds0, ds1, on='filename')
    ds = pd.merge(ds2, datasets[4], on='filename')

    print(datasets[0].shape, datasets[1].shape, ds.shape)

    combined = np.array(
        [ds.as_matrix(columns=['{}{}'.format(cls, suffix) for cls in classes]) for suffix in ['0', '1', '2', '3', '']])
    err = np.square(combined - np.mean(combined, axis=0, keepdims=True))
    err = np.sum(np.sum(err, axis=0), axis=1)
    max_values = np.max(combined, axis=0)
    mean_values = np.mean(combined, axis=0)

    ds['error'] = err
    for i, cls in enumerate(classes):
        ds['max_' + cls] = max_values[:, i]
        ds['mean_' + cls] = mean_values[:, i]

    print(np.sum(ds.error < 0.1))

    ds_matching = ds[ds.error < 0.1]
    ds_matching.to_csv(Path(__file__).parent.parent / 'output/unused_matching.csv',
                       index=False, float_format='%.7f',
                       columns=['filename'] + ['mean_' + cls for cls in classes],
                       header=['filename'] + classes)
    return
    ds_sorted = ds.sort_values(by=['error'], ascending=False)
    dest_dir = config.MODEL_DIR / 'output/to_label/'

    for cls in classes_compatible:
        os.makedirs(dest_dir+'res/'+cls, exist_ok=True)

    for i, (_, row) in enumerate(ds_sorted.iterrows()):
        if i > 2000:
            break
        mean_categories = row.as_matrix(['mean_' + cls for cls in classes])
        top_categories_idx = np.argsort(mean_categories)[::-1]

        prefixes = ''
        for cat_idx in top_categories_idx[:5]:
            prob = mean_categories[cat_idx]
            if prob < 0.05:
                break
            prefixes = prefixes + '{} {:.2f}_'.format(classes_compatible[cat_idx], prob)

        fn = '{:02} {}{}'.format(i, prefixes, row.filename)
        # print(i, row.error, row.filename, fn)
        cur_dest_dir = os.path.join(dest_dir, '{:02}'.format(i // 100))
        os.makedirs(cur_dest_dir, exist_ok=True)
        print(os.path.join(config.UNUSED_VIDEO_DIR, row.filename),
              os.path.join(cur_dest_dir, fn))
        shutil.copy(os.path.join(config.UNUSED_VIDEO_DIR, row.filename),
                    os.path.join(cur_dest_dir, fn))


def generate_labeled():
    data_dir = config.MODEL_DIR / 'input/extra_data'
    labels = {}  # video_id -> class_id
    for category_id, dir_name in enumerate(classes_compatible):
        for fn in os.listdir(os.path.join(data_dir, dir_name)):
            if fn.endswith('mp4'):
                labels[fn.split('_')[-1]] = category_id

    data = np.zeros((len(labels), len(classes)))
    file_names = []
    for fn, cls_id in labels.items():
        data[len(file_names), cls_id] = 1.0
        file_names.append(fn)

    res = pd.DataFrame(
        data={'filename': file_names}
    )
    for col, cls in enumerate(classes):
        res[cls] = data[:, col]

    res.to_csv(Path(__file__).parent.parent / 'output/unused_labeled.csv',
               index=False, float_format='%.7f', header=True)


find_problematic_clips()
generate_labeled()
