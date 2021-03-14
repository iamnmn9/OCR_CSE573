"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os
import glob
import cv2
import numpy as np
import random
#list_images = []
characters_path = "data/characters/*.jpg"
test_img_path = "C:/Users/Naman Pundir/Desktop/Project1/data/test_img.jpg"


def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def ocr(test_img_path, characters_path):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # TODO Add your code here. Do not modify the return and input arguments

    enrollment(characters_path)

    detection(test_img_path)
    
    recognition(characters_path,detection(test_img_path))

    #raise NotImplementedError



def enrollment(characters_path):
    import os
    #os.mkdir("Sifttt_char")

    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 1 : Your Enrollment code should go here.
    #pip install opencv - contrib - python == 4.4.0.44
    # OPENCV VERSION 4.4.0.44 ONLY ONLY ONLY
    import cv2
    import numpy as np
    #from google.colab.patches import cv2_imshow
    path = characters_path
    print('path1' + path)
    for file in glob.glob(path):
        print(file)
        #image_resized = misc.imresize(image, (64, 64))
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        sift = cv2.xfeatures2d.SIFT_create()
        kp = sift.detect(img, None)

        img = cv2.drawKeypoints(img, kp, None)
        val = file.split()[-1][:-4][-1]
        if val == 't':
            val = 'dot'
        cv2.imwrite('Sifttt_char//'+val+'_sift.jpg', img)

    #raise NotImplementedError

def detection(test_image_path):
    """
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 2 : Your Detection code should go here.

    #raise NotImplementedError

    label = []
    label_dict = {}

    def getLabel(neighbours):


        if all(x == 0 for x in neighbours):
            if len(label) == 0:
                label.append(1)
                return max(label)
            else:
                label.append(max(label) + 1)
                return max(label)




        else:
            max_label = 0
            min_label = 0

            neighbours = [x for x in neighbours if x != 0]
            neighbours.sort()

            min_label = neighbours[0]
            max_label = neighbours[len(neighbours) - 1]

            if max_label == min_label:
                return min_label
            else:
                label_dict[max_label] = min_label
                return min_label

    img = cv2.imread(test_image_path, 0)

    ret, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    row, column = binary_img.shape

    v = 255
    new_img = np.array(binary_img)

    # First Pass
    for i in range(row):
        for j in range(column):


            if new_img[i, j] == v:


                if i == 0 and j == 0:
                    new_img[i, j] = getLabel([])

                elif i == 0 and j > 0:
                    new_img[i, j] = getLabel([new_img[i, j - 1]])

                elif i > 0 and j == 0:
                    new_img[i, j] = getLabel([new_img[i - 1, j], new_img[i - 1, j + 1]])

                elif i > 0 and j == (column - 1):
                    new_img[i, j] = getLabel([new_img[i - 1, j - 1], new_img[i - 1, j], new_img[i, j - 1]])

                elif i > 0 and j > 0:
                    new_img[i, j] = getLabel(
                        [new_img[i - 1, j - 1], new_img[i - 1, j], new_img[i - 1, j + 1], new_img[i, j - 1]])

    # Second Pass
    for k in range(len(label_dict)):
        for i in range(row):
            for j in range(column):
                if new_img[i][j] in label_dict:
                    new_img[i][j] = label_dict[new_img[i][j]]


    output_img = np.zeros((row, column, 3), int)
    labelColor = {0: (0, 0, 0)}
    for i in range(row):
        for j in range(column):
            label = new_img[i, j]
            if label not in labelColor:
                labelColor[label] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            output_img[i, j, :] = labelColor[label]
    # naman_image=cv2.cvtColor(output_img,cv2.COLOR_BGR2GRAY)

    cv2.imwrite('output_img.png', output_img)
    return label_dict






def recognition(characters_path,labels_dict):
    import os
    #os.mkdir("bbox_bbox")
    """
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.

    #raise NotImplementedError

    import numpy as np

    list_images = []

    image = cv2.imread('output_img.png')
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


    ROI_number = 0
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        dict_images = {}
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
        ROI = original[y:y + h, x:x + w]
        dict_images

        cv2.imwrite('bbox_bbox/bbox_{}.png'.format(ROI_number), ROI)
        list_images.append({'bbox': [x, y, w, h], "name": "UNKNOWN"})
        ROI_number += 1

    #cv2_imshow(image)
    #cv2.waitKey()

    #list_images

    #import cv2

    path = characters_path
    labels_dict = {}
    for file in glob.glob(path):
        print(file)
        val = file.split()[-1][:-4][-1]
        if val == 't':
            val = 'dot'
        a = cv2.imread(file)
        img_x = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints_x, descriptors_x = sift.detectAndCompute(img_x, None)
        if keypoints_x != None:
            print(val)

        labels_dict[val] = [keypoints_x, descriptors_x, img_x]

    import matplotlib.pyplot as plt

    image_CCL = cv2.imread('output_img.png')
    original = image_CCL.copy()
    sift = cv2.xfeatures2d.SIFT_create()
    original = image.copy()
    for key, value in labels_dict.items():
        for i in list_images:
            x = i['bbox'][0]
            y = i['bbox'][1]
            w = i['bbox'][2]
            h = i['bbox'][3]
            temp_orig = original[y:y + h, x:x + w]
            img_temp = cv2.cvtColor(temp_orig, cv2.COLOR_BGR2GRAY)
            sift = cv2.xfeatures2d.SIFT_create()
            keypoints_temp, descriptors_temp = sift.detectAndCompute(img_temp, None)
            bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)



            matches = bf.match(descriptors_temp, labels_dict[key][1])
            matches = sorted(matches, key=lambda x: x.distance)
            good = []

            mean = 100000
            for m in matches:
                good.append(int(m.distance))
            # print(good)
            if len(good) > 3:

                mean = sum(sorted(good)[0:3]) / 3
            elif len(good) == 2:

                mean = sum(sorted(good)) / 2
            elif len(good) == 1:

                mean = sum(sorted(good)) / 1
            if mean < 2000:

                i['name'] = key
                print(key)

                #img3_3 = cv2.drawMatches(img_temp, keypoints_temp, labels_dict[key][2], labels_dict[key][0],
                                        # matches[:50], labels_dict[key][2], flags=2)
                # plt.imshow(img3_3),plt.show()

    return list_images







def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = recognition(characters_path,detection(test_img_path))
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()
    
    characters = []

    all_character_imgs = glob.glob(args.character_folder_path+ "/*")
    
    for each_character in all_character_imgs :
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])

    test_img = read_image(args.test_img)

    results = ocr(test_img_path, characters_path)

    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()
