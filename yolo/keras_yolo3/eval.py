"""
Compute the accuracy on the train dataset and the test dataset
"""
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import argparse


from eval_utils import YOLOPlus, load_data, load_classes, \
    detect_img, prediction_not_ok, plot_bootstrap_curve

def _main():
    # PARSER
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', '-w', type=str,
                        help="path of the network weights")
    parser.add_argument('--train', '-tr', type=str, default=None,
                        help="path of the training text file")
    parser.add_argument('--n_train', '-ntr', type=int, default=None,
                        help="number of elements of the train set to consider"
                             "(default: all)")
    parser.add_argument('--test', '-te', type=str,
                        help="path of the testing text file")
    parser.add_argument('--n_test', '-nte', type=int, default=None,
                        help="number of elements of the test set to consider"
                             "(default: all)")
    parser.add_argument('--tqdm', action="store_true",
                        help="whether to use tqdm or no")
    parser.add_argument('--gray_scale', action="store_true",
                        help="whether to put the images in gray_scale before feeding them to YOLO")
    args = parser.parse_args()


    # MAIN
    model_path = args.weights
    classes_path = "model_data/lisa_classes.txt"
    yolo = YOLOPlus(model_path=model_path, classes_path=classes_path,
                    gray_scale=args.gray_scale)


    # COMPUTE ACCURACY ON TRAIN SET
    if args.train is not None:
        train_dict, train_imgs = load_data(args.train)

        n_examples = args.n_train or len(train_imgs)
        indices = np.random.choice(len(train_imgs), n_examples, replace=False)

        n_good_predictions = 0
        wrong_images_paths = []
        for input_path in tqdm(np.array(train_imgs)[indices], ascii=True,
                               disable=not args.tqdm):
            r_image, labels, scores, boxes = detect_img(input_path, yolo, score_threshold=0.4)

            true_infos = train_dict[input_path]
            true_boxes = [info[0:4] for info in true_infos]
            true_labels = [info[4] for info in true_infos]

            whole_pred_ok = True
            for label, score, box in zip(labels, scores, boxes):
                local_pred_ok = False
                for true_box, true_label in zip(true_boxes, true_labels):
                    local_pred_ok = local_pred_ok or \
                                    not prediction_not_ok(score, true_label, true_box, label, box)
                whole_pred_ok = whole_pred_ok and local_pred_ok
            n_good_predictions += whole_pred_ok

            if not whole_pred_ok:
                # *** Uncomment this line to display the wrong predictions ***
                # r_image.show()
                wrong_images_paths.append(input_path)

        train_accuracy = round((n_good_predictions * 100) / n_examples, 2)
        print("average accuracy (train): ", train_accuracy, "%")


    # COMPUTE ACCURACY ON TEST SET
    test_dict, test_imgs = load_data(args.test)

    n_examples = args.n_test or len(test_imgs)
    indices = np.random.choice(len(test_imgs), n_examples, replace=False)

    n_good_predictions = 0
    wrong_images_paths_test = []
    for input_path in tqdm(np.array(test_imgs)[indices], ascii=True,
                           disable=not args.tqdm):
        r_image, labels, scores, boxes = detect_img(input_path, yolo, score_threshold=0.4)

        true_infos = test_dict[input_path]
        true_boxes = [info[0:4] for info in true_infos]
        true_labels = [info[4] for info in true_infos]

        whole_pred_ok = True
        for label, score, box in zip(labels, scores, boxes):
            local_pred_ok = False
            for true_box, true_label in zip(true_boxes, true_labels):
                local_pred_ok = local_pred_ok or \
                                not prediction_not_ok(score, true_label, true_box, label, box)
            whole_pred_ok = whole_pred_ok and local_pred_ok
        n_good_predictions += whole_pred_ok

        if not whole_pred_ok:
            # *** Uncomment this line to display the wrong predictions ***
            # r_image.show()
            wrong_images_paths_test.append(input_path)

    test_accuracy = round((n_good_predictions * 100) / n_examples, 2)
    if args.train is not None:
        print("average accuracy (train): ", train_accuracy, "%")
    print("average accuracy (test): ", test_accuracy, "%")

if __name__ == "__main__":
    _main()
