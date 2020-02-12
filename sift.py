import cv2
import os
import glob
import matplotlib.pyplot as plt
import numpy as np


DATA_TRAIN_DIRECTORY = "data_sift/train"
DATA_TEST_SINGLE_DIRECTORY = "data_sift/test/single"
DATA_TEST_MULTIPLE_DIRECTORY = "data_sift/test/multiple"

sift = cv2.xfeatures2d.SIFT_create()

CLASSES = ["2c", "2d", "2h", "2s",
           "3c", "3d", "3h", "3s",
           "4c", "4d", "4h", "4s",
           "5c", "5d", "5h", "5s",
           "6c", "6d", "6h", "6s",
           "7c", "7d", "7h", "7s",
           "8c", "8d", "8h", "8s",
           "9c", "9d", "9h", "9s",
           "10c", "10d", "10h", "10s",
           "Ac", "Ad", "Ah", "As",
           "Jc", "Jd", "Jh", "Js",
           "Kc", "Kd", "Kh", "Ks",
           "Qc", "Qd", "Qh", "Qs"]


def to_gray_scale(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img_gray


def preprocess_image(image):
    image = cv2.bilateralFilter(image, d=7, sigmaSpace=75, sigmaColor=75)
    img_gray = to_gray_scale(image)
    a = img_gray.max()
    _, thresh = cv2.threshold(img_gray, a / 2 + 60, a, cv2.THRESH_BINARY)
    return thresh


def get_contours(image):
    binary_image = preprocess_image(image)
    _, contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy


def crop_card(image):
    contours, hierarchy = get_contours(image)
    for cnt in contours:
        accuracy = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, accuracy, True)
        x, y, w, h = cv2.boundingRect(approx)
        if h/w < 1.4:
            continue
        img_crop = image[y:y+h, x:x+w]
    return img_crop


def get_card_name(f):
    first, second = str(f).split('\\')
    name, extension = str(second).split('.')
    if str(name[1]).isdigit():
        return name[:3]
    else:
        return name[:2]


def load_train_images():
    card_label = []
    img_dir = DATA_TRAIN_DIRECTORY
    data_path = os.path.join(img_dir, '*g')
    files = glob.glob(data_path)
    for f1 in files:
        class_name = get_card_name(f1)
        class_num = CLASSES.index(class_name)
        img = cv2.imread(f1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        card = crop_card(img)
        card_label.append((card, class_num))
    return card_label


def crop_and_rotate_card(image):
    contours, hierarchy = get_contours(image)
    croped_cards = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
        width = int(rect[1][0])
        height = int(rect[1][1])
        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height - 1],
                            [0, 0],
                            [width - 1, 0],
                            [width - 1, height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (width, height))
        dimensions = warped.shape
        if dimensions[0] < dimensions[1]:
            warped = np.rot90(warped)
        if len(contours) > 1:
            croped_cards.append(warped)
        else:
            return warped
    return croped_cards


def load_test_single_cards():
    test_img_card = []
    img_dir = DATA_TEST_SINGLE_DIRECTORY
    data_path = os.path.join(img_dir, '*g')
    files = glob.glob(data_path)
    for f1 in files:
        class_name = get_card_name(f1)
        class_num = CLASSES.index(class_name)
        img = cv2.imread(f1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        card = crop_and_rotate_card(img)
        new_tuple = (img, (card, class_num))
        test_img_card.append(new_tuple)
    return test_img_card


def get_training_desc():
    card_label = load_train_images()
    training_tuples = []
    for element in card_label:
        card = element[0]
        img_gray = to_gray_scale(card)
        training_k, training_d = sift.detectAndCompute(img_gray, None)
        my_tuple = (element, training_k, training_d)
        training_tuples.append(my_tuple)
    return training_tuples


def get_test_desc():
    test_img_card = load_test_single_cards()
    test_img_card_multiple = load_test_multiple_cards()
    test_images = test_img_card + test_img_card_multiple
    test_tuples = []
    for element in test_images:
        card = element[1][0]
        gray_test_card = to_gray_scale(card)
        test_k, test_d = sift.detectAndCompute(gray_test_card, None)
        new_tuple = (element[0], element[1], test_k, test_d)
        test_tuples.append(new_tuple)
    return test_tuples


def get_score(train_k, train_d, test_k, test_d):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(train_d, test_d, k=2)

    # FLANN_INDEX_KDITREE = 0
    # flannParam = dict(algorithm=FLANN_INDEX_KDITREE, tree=5)
    # flann = cv2.FlannBasedMatcher(flannParam, {})
    #
    # matches = flann.knnMatch(train_d, test_d, k=2)

    # Apply ratio test
    good_points = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_points.append([m])

    if len(train_k) <= len(test_k):
        number_keypoints = len(train_k)
    else:
        number_keypoints = len(test_k)

    percentage_score = len(good_points) / number_keypoints * 100
    return percentage_score


def get_prediction():
    global prediction
    prediction_labels = []
    show_results_data = []
    training_tuples = get_training_desc()
    test_tuples = get_test_desc()
    for test_tuple in test_tuples:
        max_score = 0
        for train_tuple in training_tuples:
            train_k = train_tuple[1]
            train_d = train_tuple[2]
            test_k = test_tuple[2]
            test_d = test_tuple[3]
            score = get_score(train_k, train_d, test_k, test_d)
            if score > max_score:
                max_score = score
                prediction = train_tuple[0][1]
                new_tuple = (train_tuple[0][0], test_tuple[0], test_tuple[1][0])
        prediction_labels.append(prediction)
        show_results_data.append(new_tuple)
    return prediction_labels, show_results_data


def get_true_labels(test_tuples):
    true_labels = []
    for test_tuple in test_tuples:
        true_labels.append(test_tuple[1][1])
    return true_labels


def success_percentage():
    test_tuples = get_test_desc()
    number_of_test_images = len(test_tuples)
    true_labels = get_true_labels(test_tuples)
    results, _ = get_prediction()
    correct_predictions = 0
    for i in range(len(true_labels)):
        if true_labels[i] == results[i]:
            correct_predictions += 1

    success_score = round(correct_predictions/number_of_test_images * 100, 2)
    print(str(success_score) + "%")


def show_results():
    _, show_results_data = get_prediction()
    for result in show_results_data:
        f, axarr = plt.subplots(nrows=1, ncols=3)
        plt.sca(axarr[0])
        plt.imshow(result[0])
        plt.title('Prediction card')
        plt.sca(axarr[1]);
        plt.imshow(result[1]);
        plt.title('Test image')
        plt.sca(axarr[2]);
        plt.imshow(result[2]);
        plt.title('Test card')
        plt.show()


def extract_data(line):
    image, labels = str(line).split(':')
    labels = str(labels).split("|")
    labels.pop()

    return image, labels


def load_test_multiple_cards():
    test_img_card_multiple = []
    f = open('data_sift/multiple_cards.txt')
    line = f.readline()
    while line:
        image, labels = extract_data(line)
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cards = crop_and_rotate_card(img)
        for i in range(len(cards)):
            class_num = CLASSES.index(labels[i])
            new_tuple = (img, (cards[i], class_num))
            test_img_card_multiple.append(new_tuple)
        line = f.readline()
    f.close()
    return test_img_card_multiple


success_percentage()
show_results()
