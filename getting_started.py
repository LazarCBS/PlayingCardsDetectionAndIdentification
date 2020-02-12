import cv2
import os
import glob
import matplotlib.pyplot as plt
import numpy as np


training_images = []
test_images = []
crop_training_cards = []
test_tuples = list()


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
        # rect = cv2.minAreaRect(cnt)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # img = cv2.drawContours(image, contours, -1, (0, 255, 0), 5)
        # plt.imshow(img)
        # plt.show()
        accuracy = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, accuracy, True)
        x, y, w, h = cv2.boundingRect(approx)
        if h/w < 1.4:
            continue
        img_crop = image[y:y+h, x:x+w]
    return img_crop


def get_training_images():
    img_dir = "E:\Fakultet, 4. godina\/7. semestar\Soft kompjuting\BlackBackground\BlackTraining"
    data_path = os.path.join(img_dir, '*g')
    files = glob.glob(data_path)
    for f1 in files:
        img = cv2.imread(f1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        training_images.append(img)
        img_crop = crop_card(img)
        # plt.imshow(img_crop)
        # plt.show()
        crop_training_cards.append(img_crop)


def get_test_images():
    img_dir = "E:\Fakultet, 4. godina\/7. semestar\Soft kompjuting\BlackBackground\BlackTest"
    data_path = os.path.join(img_dir, '*g')
    files = glob.glob(data_path)
    for f1 in files:
        img = cv2.imread(f1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        test_images.append(img)
        croped_cards = crop_and_rotate_card(img)
        test_tuple = (img, croped_cards)
        test_tuples.append(test_tuple)


def to_gray_scale(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img_gray


def crop_and_rotate_card(image):
    contours, hierarchy = get_contours(image)
    croped_cards = []
    for cnt in contours:
        # accuracy = 0.01 * cv2.arcLength(cnt, True)
        # approx = cv2.approxPolyDP(cnt, accuracy, True)
        # x, y, w, h = cv2.boundingRect(approx)
        # if h / w < 1.2:
        #     continue
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
        croped_cards.append(warped)
    return croped_cards


sift = cv2.xfeatures2d.SIFT_create()
get_training_images()
get_test_images()

training_tuples = list()


def get_training_desc(images):
    for training_card in crop_training_cards:
        img_gray = to_gray_scale(training_card)
        training_k, training_d = sift.detectAndCompute(img_gray, None)
        my_tuple = (training_card, training_k, training_d)
        training_tuples.append(my_tuple)


get_training_desc(training_images)


def get_score(train_tuple, test_keypoints, test_descriptor):
    # bf = cv2.BFMatcher(cv2.NORM_L2)
    # matches = bf.knnMatch(train_tuple[2], test_descriptor, k=2)

    FLANN_INDEX_KDITREE = 0
    flannParam = dict(algorithm=FLANN_INDEX_KDITREE, tree=5)
    flann = cv2.FlannBasedMatcher(flannParam, {})

    matches = flann.knnMatch(train_tuple[2], test_descriptor, k=2)

    # Apply ratio test
    good_points = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_points.append([m])

    if len(train_tuple[1]) <= len(test_keypoints):
        number_keypoints = len(train_tuple[1])
    else:
        number_keypoints = len(test_keypoints)

    percentage_score = len(good_points) / number_keypoints * 100
    return percentage_score


results = list()


for test_tuple in test_tuples:
    for test_card in test_tuple[1]:
        test_image_gray = to_gray_scale(test_card)
        test_k, test_d = sift.detectAndCompute(test_image_gray, None)
        max_score = 0
        for training_tuple in training_tuples:
            score = get_score(training_tuple, test_k, test_d)
            if score > max_score:
                max_score = score
                new_tuple = (training_tuple[0], test_tuple[0], test_card, max_score)
        results.append(new_tuple)

#
# f = plt.figure()
#     subplot1 = f.add_subplot(1, 2, 1)
#     subplot1.title.set_text('Test image')
#     plt.imshow(result[0])
#     subplot2 = f.add_subplot(2, 2, 1)
#     subplot2.title.set_text('Training card')
#     plt.imshow(result[1])
#     subplot3 = f.add_subplot(3, 2, 1)
#     subplot3.title.set_text('Test card, Score: ' + str(round(result[3], 2)) + "%")
#     plt.imshow(result[2])
#     plt.show(block=True)
#     print("Score: " + str(result[3]))


for result in results:
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

