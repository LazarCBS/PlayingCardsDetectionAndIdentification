import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random

from keras import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, Dropout
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from sklearn.metrics import accuracy_score

DATA_TRAIN_DIRECTORY = "data/train"
DATA_TEST_ONE_SIMPLE_CARD = "data/test/one_simple_card"
DATA_TEST_ROTATED_CARD = "data/test/one_rotated_card"

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

training_data = []
test_one_simple_card = []


def convert_output(classes):
    nn_outputs = []
    for index in range(len(classes)):
        row = np.zeros(len(classes))
        row[index] = 1
        nn_outputs.append(row)
    return np.array(nn_outputs)


def resize_to_square(img):
    desired_size = 128
    old_size = img.shape[:2]

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    im = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    return new_im


def resize_image(img):
    scale_percent = 10  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return resized


def load_train_images():
    images_train = []
    labels_train = []
    for rank_suite in CLASSES:
        path = os.path.join(DATA_TRAIN_DIRECTORY, rank_suite)
        class_num = CLASSES.index(rank_suite)
        outputs = convert_output(CLASSES)
        for img in os.listdir(path):
            try:
                img_train = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                img_train = cv2.cvtColor(img_train, cv2.COLOR_BGR2RGB)
                resized = resize_image(img_train)
                training_data.append([resized, outputs[class_num]])
            except Exception as e:
                pass
    random.shuffle(training_data)
    for img_train, label in training_data:
        images_train.append(img_train)
        labels_train.append(label)
    return images_train, labels_train


def load_test_one_card():
    test_images = []
    test_labels = []
    for rank_suite in CLASSES:
        path = os.path.join(DATA_TEST_ONE_SIMPLE_CARD, rank_suite)
        class_num = CLASSES.index(rank_suite)
        for img in os.listdir(path):
            try:
                img_test = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)
                resized = resize_image(img_test)
                test_images.append(resized)
                test_labels.append(class_num)
            except Exception as e:
                pass
    return test_images, test_labels


def load_test_rotated_card():
    test_images = []
    test_labels = []
    for rank_suite in CLASSES:
        path = os.path.join(DATA_TEST_ROTATED_CARD, rank_suite)
        class_num = CLASSES.index(rank_suite)
        for img in os.listdir(path):
            try:
                img_test = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)
                resized = resize_image(img_test)
                test_images.append(resized)
                test_labels.append(class_num)
            except Exception as e:
                pass
    return test_images, test_labels


def save_data(images_train, labels_train):
    x_train_array = np.array(images_train)
    y_train_array = np.array(labels_train)
    np.save('data_saved/images_train_data', x_train_array)
    np.save('data_saved/labels_train_data', y_train_array)


def load_data():
    try:
        x_train = np.load('data_saved/images_train_data.npy')
        y_train = np.load('data_saved/labels_train_data.npy')
        return x_train, y_train
    except IOError:
        save_data()
        print("Data has been saved. Now you can load it.")


def prepare_data(x, y):
    x_prepared = np.array(x)
    y_prepared = np.array(y)
    return x_prepared, y_prepared


def define_train_generator():
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=90,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       zoom_range=0.3
                                       )
    return train_datagen


def define_validation_generator():
    validation_datagen = ImageDataGenerator(rescale=1. / 255,
                                            rotation_range=90,
                                            width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            zoom_range=0.3
                                            )

    valid_generator = validation_datagen.flow_from_directory('data/train', target_size=(101, 49), batch_size=32,
                                                             class_mode='categorical')

    return valid_generator


def define_cnn_model():
    custom_vgg = Sequential()
    custom_vgg.add(Conv2D(32, (3, 3), strides=1, padding="same", activation="relu", input_shape=(101, 49, 3)))
    custom_vgg.add(Dropout(0.4))
    custom_vgg.add(Conv2D(32, (3, 3), strides=1, padding="same", activation="relu"))
    custom_vgg.add(Dropout(0.4))
    custom_vgg.add(MaxPooling2D((2, 2)))

    custom_vgg.add(Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"))
    custom_vgg.add(Dropout(0.4))
    custom_vgg.add(Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"))
    custom_vgg.add(Dropout(0.4))
    custom_vgg.add(MaxPooling2D((2, 2)))

    custom_vgg.add(Conv2D(128, (3, 3), strides=1, padding="same", activation="relu"))
    custom_vgg.add(Dropout(0.4))
    custom_vgg.add(Conv2D(128, (3, 3), strides=1, padding="same", activation="relu"))
    custom_vgg.add(Dropout(0.4))
    custom_vgg.add(MaxPooling2D((2, 2)))

    custom_vgg.add(Flatten())
    custom_vgg.add(Dense(52      , activation="softmax"))

    custom_vgg.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return custom_vgg


def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]


def predict(trained_model, x_test):
    label_array = []
    results = trained_model.predict(np.array(x_test, np.float))
    for result in results:
        win = CLASSES[winner(result)]
        label_array.append(win)
    return label_array


def show_images(images):
    for img in images:
        print(str(img.shape))
        plt.imshow(img)
        plt.show()


def load_trained_model():
    try:
        json_file = open('trained_model/trained_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        cnn = model_from_json(loaded_model_json)
        cnn.load_weights("data_saved/best_model.hdf5")
        print("Istrenirani model uspeno ucitan.")
        return cnn
    except Exception as e:
        return None


def save_model(model):
    model_json = model.to_json()
    with open("trained_model/trained_model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("trained_model/trained_model.h5")


def train_save_model():
    images_train, labels_train = load_train_images()
    X_train, y_train = prepare_data(images_train, labels_train)
    train_generator = define_train_generator()
    valid_generator = define_validation_generator()
    model = define_cnn_model()
    checkpoint = ModelCheckpoint("data_saved/best_model.hdf5", monitor='loss', verbose=1,
                             save_best_only=True, mode='auto', period=1)
#
    model.fit_generator(train_generator.flow(X_train, y_train, batch_size=64), validation_data=valid_generator,
                    steps_per_epoch=len(X_train) // 64, epochs=1000,
                    verbose=1, callbacks=[checkpoint])

    print("Treniranje zavrseno")
    save_model(model)

    return model


def get_accuracy(real_labels, predicted_labels):
    predicted_index = []
    for label in predicted_labels:
        predicted_index.append(CLASSES.index(label))
    score_accuracy = accuracy_score(real_labels, predicted_index)
    print("Accuracy: " + str(round(score_accuracy * 100, 2)) + "%")


def predict_one_card_data():
    test_images, test_labels = load_test_one_card()
    model = load_trained_model()
    X_test, y_test = prepare_data(test_images, test_labels)
    predicted_labels = predict(model, X_test)
    get_accuracy(y_test, predicted_labels)


def predict_one_rotated_card():
    test_images, test_labels = load_test_rotated_card()
    model = load_trained_model()
    X_test, y_test = prepare_data(test_images, test_labels)
    predicted_labels = predict(model, X_test)
    get_accuracy(y_test, predicted_labels)


predict_one_rotated_card()




