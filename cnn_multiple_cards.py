from sklearn.metrics import accuracy_score

from multiple_cards import *
from cnn import prepare_data, predict, load_trained_model, show_images

DIRECTORY_MULTIPLE_IMAGES = "data/test/multiple"


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


def extract_data(line):
    image, labels = str(line).split(':')
    labels = str(labels).split("|")
    labels.pop()

    return image, labels


def load_test_multiple_cards():
    test_cards = []
    test_labels = []
    f = open('data/test/multiple_cards.txt')
    line = f.readline()
    while line:
        image, labels = extract_data(line)
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cards = crop_and_rotate_card(img)
        for i in range(len(cards)):
            class_num = CLASSES.index(labels[i])
            test_cards.append(cards[i])
            test_labels.append(class_num)
        line = f.readline()
    f.close()
    return test_cards, test_labels


def show_images(images):
    for img in images:
        print(str(img.shape))
        plt.imshow(img)
        plt.show()


def resize_cards(test_images):
    resized_images = []
    for img in test_images:
        width = 49
        height = 101
        dim = (width, height)

        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        resized_images.append(resized)
    return resized_images


test_images, test_labels = load_test_multiple_cards()
model = load_trained_model()
resized_cards = resize_cards(test_images)
X_test = np.array(resized_cards)

predicted_labels = predict(model, X_test)

predicted_index = []
for label in predicted_labels:
    predicted_index.append(CLASSES.index(label))

score_accuracy = accuracy_score(test_labels, predicted_index)

print(str(score_accuracy))
