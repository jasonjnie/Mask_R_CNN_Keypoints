import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as sio
import json

from config import Config
import utils as utils
import model as modellib
import visualize
from model import log

mpii_class_names_ = ['right_ankle', 'right_knee', 'right_hip', 'left_hip', 'left_knee',  'left_ankle', 'pelvis',
                     'thorax', 'upper_neck', 'head_top', 'right_wrist', 'right_elbow', 'right_shoulder',
                     'left_shoulder', 'left_elbow', 'left_wrist']

mpii_class_names = ['person']

KEYS = ['isVal', 'joints', 'height', 'width', 'numPeople', 'path', 'annlist_index']

NUM_JOINTS = 16


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


class mpii_data_set(utils.Dataset):
    """ Load the key points dataset (MPII, JHMDB)
        The dataset consists of Human Skeletal System Keypoints.
        For each salient human figure in the dataset,
        we labeled it with 16 human skeletal keypoints.

                Table 1: The numerical orders of human skeletal keypoints
        0 - r ankle,        1 - r knee,         2 - r hip,      3 - l hip,      4 - l knee,     5 - l ankle,
        6 - pelvis,         7 - thorax,         8 - upper neck, 9 - head top,   10 - r wrist,   11 - r elbow,
        12 - r shoulder,    13 - l shoulder,    14 - l elbow,   15 - l wrist
    """

    def _loadmat(filename):
        '''
        this function should be called instead of direct spio.loadmat
        as it cures the problem of not properly recovering python dictionaries
        from mat files. It calls the function check keys to cure all entries
        which are still mat-objects
        '''

        def _check_keys(d):
            '''
            checks if entries in dictionary are mat-objects. If yes
            todict is called to change them to nested dictionaries
            '''
            for key in d:
                if isinstance(d[key], sio.matlab.mio5_params.mat_struct):
                    d[key] = _todict(d[key])
            return d

        def _todict(matobj):
            '''
            A recursive function which constructs from matobjects nested dictionaries
            '''
            d = {}
            for strg in matobj._fieldnames:
                elem = matobj.__dict__[strg]
                if isinstance(elem, sio.matlab.mio5_params.mat_struct):
                    d[strg] = _todict(elem)
                elif isinstance(elem, np.ndarray):
                    d[strg] = _tolist(elem)
                else:
                    d[strg] = elem
            return d

        def _tolist(ndarray):
            '''
            A recursive function which constructs lists from cellarrays
            (which are loaded as numpy ndarrays), recursing into the elements
            if they contain matobjects.
            '''
            elem_list = []
            for sub_elem in ndarray:
                if isinstance(sub_elem, sio.matlab.mio5_params.mat_struct):
                    elem_list.append(_todict(sub_elem))
                elif isinstance(sub_elem, np.ndarray):
                    elem_list.append(_tolist(sub_elem))
                else:
                    elem_list.append(sub_elem)
            return elem_list

        data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
        return _check_keys(data)

    def load_mpii(self, data_dir, subset):
        """
        Load MPII / JHMDB dataset.
        data_dir: The root directory of the data-set.
        subset: What to load (train, test, val).

        return_data: If True, returns the data object.
        """

        # Path
        # path_train = 'ai_challenger_keypoint_train_20170909/keypoint_train_images_20170902'
        # path_valid = 'ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911'
        # image_dir = os.path.join(data_dir, path_train if subset == "train" else path_valid)
        image_dir = data_dir

        # load Anotation file
        '''
        json_path_dict = {
            "train": "ai_challenger_keypoint_train_20170909/keypoint_train_annotations_modif.json",
            "val": "ai_challenger_keypoint_validation_20170911/keypoint_validation_annotations_modif.json",
        }
        json_file = pd.read_json(os.path.join(data_dir, json_path_dict[subset]))
        '''
        # annotation_file_path = os.path.join(data_dir, "mpii_human_pose.mat")
        # annotation_file = self._loadmat("mpii_human_pose.mat")['RELEASE']
        # N = len(annotation_file['img_train'])


        json_file = 'res_.json'
        with open(json_file) as f:
            orig_annotation = json.load(f)   # a list

        num_all = len(orig_annotation)  # 16485
        print('num of all imgs:', num_all)
        keys = orig_annotation[0].keys()  # ['isVal', 'joints', 'height', 'width', 'numPeople', 'path', 'annlist_index']
        #print(keys)
        #temp = orig_annotation[0]
        #print(temp)
        annotation = []
        for i in range(num_all):
            if subset == 'train' and orig_annotation[i]['isVal'] == 0 or subset == 'val' and orig_annotation[i]['isVal'] == 1:
                annotation.append(orig_annotation[i])
        N = len(annotation)
        print('Num of {} = {}'.format(subset, N))

        # All classes
        # The class IDs start from 1, since the 0 belongs to the background class
        class_ids = list(range(1, 2))   # [1]

        # Add classes
        for i in class_ids:
            self.add_class("MPII", i, mpii_class_names[i-1])    # 'person'

        # Add images
        for i in range(N):
            #annotations = annotation_file.iloc[i]
            im_path = os.path.join(image_dir, annotation[i]['path'])
            self.add_image("MPII", image_id=i, path=im_path,
                           width=annotation[i]['width'], height=annotation[i]['height'],
                           annotations=annotation[i])

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not an MPII image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "MPII":
            print("Main 182: wrong source")
            return super(self.__class__).load_mask(image_id)

        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to if ist a person or not.
        human_nums = annotations['numPeople']
        m = np.zeros([human_nums, int(annotations['width']), int(annotations['height']), NUM_JOINTS])
        class_mask = np.zeros([human_nums, NUM_JOINTS])
        # For every human annotations.
        for human_num in range(human_nums):
            annotation = np.reshape(annotations['joints'][human_num], (NUM_JOINTS, 3))
            for part_num, bp in enumerate(annotation):      # bp = [width, height, 0/1]
                if bp[2] < 2:
                    m[human_num, bp[0], bp[1], part_num] = 1
                class_mask[human_num, part_num] = bp[2]
            class_ids.append(1)

        # Pack instance masks into an array
        if class_ids:
            mask = m
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids, class_mask
            # return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(self.__class__).load_mask(image_id)

    # def image_reference(self, ):


def plot_mask_points(dataset, config, model, filter=True, image_id=None):

    if not image_id:
        image_id = random.choice(dataset.image_ids)
    original_image, image_meta, gt_bbox, gt_mask = modellib.load_image_gt(dataset,
                                                                          config,
                                                                          image_id, use_mini_mask=False)

    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    mrcnn = model.run_graph([original_image], [
        ("detections", model.keras_model.get_layer("mrcnn_detection").output),
        ("masks", model.keras_model.get_layer("mrcnn_mask").output),
        ("mask_classes", model.keras_model.get_layer("mrcnn_class_mask").output),
    ])
    det_ix = mrcnn['detections'][0, :, 4]
    det_count = np.where(det_ix == 0)[0][0]
    det_masks = mrcnn['masks'][0, :det_count, :, :, :]
    det_boxes = mrcnn['detections'][0, :det_count, :]
    det_mask_classes = np.argmax(mrcnn['mask_classes'][0, :det_count, :, :], axis=2)
    det_mask_classes = np.where(det_mask_classes == 0, np.ones_like(det_mask_classes), np.zeros_like(det_mask_classes))

    visualize.draw_mpii_boxes(original_image, refined_boxes=det_boxes[:, :4], bp=mpii_class_names_)
    _, ax = plt.subplots(3, 5)
    for i in range(5):
        ax[0, i].set_title(mpii_class_names_[i])
        if filter:
            m = np.where(det_masks[0, :, :, i] > 0.8, det_masks[0, :, :, i], 0)
            m = np.where(m == m.max(), 1, 0)
            m = m * det_mask_classes[0, i]
        else:
            m = det_masks[0, :, :, i] * det_mask_classes[0, i]
        ax[0, i].imshow(m, interpolation='none')
    for i in range(5):
        ax[1, i].set_title(mpii_class_names_[5 + i])
        if filter:
            m = np.where(det_masks[0, :, :, 5 + i] > 0.8, det_masks[0, :, :, 5 + i], 0)
            m = np.where(m == m.max(), 1, 0)
            m = m * det_mask_classes[0, 5 + i]
        else:
            m = det_masks[0, :, :, 5 + i] * det_mask_classes[0, 5 + i]
        ax[1, i].imshow(m, interpolation='none')
    for i in range(4):
        ax[2, i].set_title(mpii_class_names_[10 + i])
        if filter:
            m = np.where(det_masks[0, :, :, 10 + i] > 0.8, det_masks[0, :, :, 10 + i], 0)
            m = np.where(m == m.max(), 1, 0)
            m = m * det_mask_classes[0, 10 + i]
        else:
            m = det_masks[0, :, :, 10 + i] * det_mask_classes[0, 10 + i]
        ax[2, i].imshow(m, interpolation='none')
    ax[2, 4].set_title('Real image')
    visualize.draw_mpii_boxes(original_image, refined_boxes=det_boxes[:1, :4], ax=ax[2, 4], bp=mpii_class_names_)

    # Plot the gt mask points
    _, axx = plt.subplots(3, 5)
    axx[2, 4].set_title('Real image')
    visualize.draw_mpii_boxes(original_image, refined_boxes=gt_bbox[:1, :4], masks=gt_mask,
                            ax=axx[2, 4], bp=mpii_class_names_)
    original_image, image_meta, gt_bbox, gt_mask = modellib.load_image_gt(dataset,
                                                                          config,
                                                                          image_id, use_mini_mask=True)
    for i in range(5):
        axx[0, i].set_title(mpii_class_names_[i])
        axx[0, i].imshow(gt_mask[0, :, :, i], interpolation='none')
    for i in range(5):
        axx[1, i].set_title(mpii_class_names_[5 + i])
        axx[1, i].imshow(gt_mask[0, :, :, 5 + i], interpolation='none')
    for i in range(4):
        axx[2, i].set_title(mpii_class_names_[10 + i])
        axx[2, i].imshow(gt_mask[0, :, :, 10 + i], interpolation='none')
    #plt.show()
    plt.imsave('output.png')


def main():

    # Root directory of the project
    ROOT_DIR = os.getcwd()

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    # MODEL_DIR = '/media/rodrigo/c1d7e9c9-c8cb-402e-b241-9090925389b3/IA_Challenger/save_log'
    # MODEL_DIR = os.path.join(MODEL_DIR, "logs")

    # Path to COCO trained weights
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

    # Path to MPII data set
    # TODO: 
    #MPII_DATA_PATH = '/media/rodrigo/c1d7e9c9-c8cb-402e-b241-9090925389b3/IA_Challenger'
    MPII_DATA_PATH = os.path.join(ROOT_DIR, 'dataset')

    class mpii_data_config(Config):
        """Configuration for training on the dataset.
        Derives from the base Config class and overrides values specific
        to training dataset.
        """
        # Give the configuration a recognizable name
        NAME = "MPII"

        # Train on 1 GPU and 1 images per GPU.
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

        # Number of classes (including background)
        NUM_CLASSES = 1 + 1  # background + 1 mask

        RPN_TRAIN_ANCHORS_PER_IMAGE = 150

        # Number of validation steps to run at the end of every training epoch.
        # A bigger number improves accuracy of validation stats, but slows
        # down the training.
        VALIDATION_STPES = 100
        STEPS_PER_EPOCH = 5
        MINI_MASK_SHAPE = (56, 56)

        # Pooled ROIs
        POOL_SIZE = 7
        MASK_POOL_SIZE = 14
        MASK_SHAPE = [28, 28]

        # Maximum number of ground truth instances to use in one image
        MAX_GT_INSTANCES = 128

    config = mpii_data_config()

    # GPU for training.
    DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

    TEST_MODE = "train"

    # Training dataset
    dataset_train = mpii_data_set()
    dataset_train.load_mpii(MPII_DATA_PATH, TEST_MODE)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = mpii_data_set()
    dataset_val.load_mpii(MPII_DATA_PATH, 'val')
    dataset_val.prepare()
    print("Classes: {}.\n".format(dataset_train.class_names))
    print("Train Images: {}.\n".format(len(dataset_train.image_ids)))
    print("Valid Images: {}".format(len(dataset_val.image_ids)))

    image_id = np.random.choice(dataset_val.image_ids)
    image, image_meta, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, config, image_id,
                                                                 use_mini_mask=False, augment=False)

    """
    # visualize.draw_boxes(image, refined_boxes=gt_bbox[:, :4], masks=gt_mask)
    visualize.draw_mpii_boxes(image, refined_boxes=gt_bbox[:, :4], masks=gt_mask, bp=mpii_class_names_)
    plt.imsave('boxes.png')

    #################################
    # LOAD MODEL
    #################################

    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode='training', config=config, model_dir=MODEL_DIR)

    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    # model.load_weights(COCO_MODEL_PATH, by_name=True,
    #                    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask", "mrcnn_mask_deconv"])
    # model.load_weights('/media/rodrigo/c1d7e9c9-c8cb-402e-b241-9090925389b3/IA_Challenger/save_log/logs/mask_rcnn_ai.h5',
    #                    by_name=True, exclude=["mrcnn_mask"])
    # path_save = '/media/rodrigo/c1d7e9c9-c8cb-402e-b241-9090925389b3/IA_Challenger/save_log/logs/mask_rcnn_ai.h5'
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=320,
                layers='heads')
    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=960,
                layers="all")

    # Save weights
    # Typically not needed because callbacks save after every epoch
    # Uncomment to save manually
    # model_path = os.path.join(MODEL_DIR, "mask_rcnn_keypoints.h5")
    # model.keras_model.save_weights(model_path)

    class InferenceConfig(data_config):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    model_path = model.find_last()[1]

    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    # Test on a random image
    plot_mask_points(dataset_val, inference_config, model)

    print('finished')
    """

if __name__ == '__main__':
    main()
