import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import imutils
import glob
import json
import sys
from sklearn.cluster import AgglomerativeClustering
import shutil
import os
import random


def get_white_proportion(edges):
    return cv.sumElems(edges)[0] / (255 * len(edges[0]) * len(edges))


# resolution: 1280 x 720

# Hadamard (bitwise) product of 2 matrices: A.mul(B)

def decompose_image(filename, width_jumps=1280, height_jumps=720):
    print(filename)
    basename = filename.split('/')[-1][:-4]
    img = cv.imread(filename)
    height = len(img)
    width = len(img[0])
    for i in range(0, height, height_jumps):
        if i + height_jumps > height:
            break
        for j in range(0, width, width_jumps):
            if j + width_jumps > width:
                break
            sub_img = img[i:i+height_jumps, j:j+width_jumps]
            cv.imwrite("decomposed_images/{}_{}_{}.tif".format(basename, i, j), sub_img)


def get_edges(filename, target_proportion=0.01):
    print(filename)
    img = cv.imread(filename)
    best_params = (None, None)
    best_prop = None
    best_edges = None
    lowest_error = float('inf')
    for lower in [5, 10, 20, 40, 80, 160]:
        for ratio in [1.5, 2, 3, 5]:
            edges = cv.Canny(img, lower, lower*ratio)
            prop = get_white_proportion(edges)
            error = abs(target_proportion-prop) / target_proportion
            if error < lowest_error:
                best_prop = prop
                lowest_error = error
                best_params = (lower, ratio)
                best_edges = edges
    cv.imwrite("edges/{}".format(filename.split('/')[-1]), best_edges)


def get_black_white(filename, target_proportion=0.5):
    print(filename)
    img = cv.imread(filename)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    best_thresh = None
    best_prop = None
    best_black_white = None
    lowest_error = float('inf')
    for thresh in range(255):
        dum, black_white = cv.threshold(img, thresh, 255, cv.THRESH_BINARY)
        prop = get_white_proportion(black_white)
        error = abs(target_proportion - prop) / target_proportion
        if error < lowest_error:
            best_prop = prop
            lowest_error = error
            best_thresh = thresh
            best_black_white = black_white
    print(best_prop, lowest_error, best_thresh)
    cv.imwrite("black_white/{}".format(filename.split('/')[-1]), best_black_white)





def make_binary(edges):
    binary = np.zeros((len(edges), len(edges[0])))
    for i in range(len(edges)):
        for j in range(len(edges[0])):
            if edges[i][j] > 0:
                binary[i][j] = 1
    return binary


def get_edges_similarity(edges_1, edges_2):
    max_value = min(np.sum(edges_1), np.sum(edges_2))/255
    similarity = np.sum(np.multiply(edges_1, edges_2)) / max_value
    return similarity


def get_closest(dist_mat, labels, start_i, n_imgs=100):
    taken = {start_i}
    imgs = [labels[start_i]]
    current_i = start_i
    for i in range(1, n_imgs):
        min_dist = float('inf')
        min_j = None
        for j in range(len(dist_mat)):
            if j in taken:
                continue
            elem = dist_mat[current_i][j]
            if elem < min_dist:
                min_dist = elem
                min_j = j
        print(min_dist)
        imgs.append(labels[min_j])
        taken.add(min_j)
    return imgs








if __name__ == "__main__":
    # étape 1, décomposer les images
    # images = glob.glob("raw_images/*.tif")
    # [decompose_image(image) for image in images]

    # étape 2, "edge detection" en visant environ 1% de blanc
    # decomposed_imgs = glob.glob("decomposed_images/*.tif")
    # [get_edges(img) for img in decomposed_imgs]

    # étape 2B, images noir blanc
    decomposed_imgs = glob.glob("decomposed_images/*.tif")
    random_imgs = set()
    for i in range(10):
        random_imgs.add(random.choice(decomposed_imgs))
    [get_black_white(filename) for filename in random_imgs]

    # étape 3, construction de la matrice de distance
    # edges_files = glob.glob("edges/*.tif")
    # edges = [cv.imread(filename, 0) for filename in edges_files]
    # n = len(edges)
    # dist_mat = np.zeros((n, n))
    # for i in range(n):
    #     for j in range(i, n):
    #         similarity = get_edges_similarity(edges[i], edges[j])
    #         dist_mat[i][j] = dist_mat[j][i] = 1 - similarity
    #     print(i)
    # with open("labels.json", "w") as f:
    #     json.dump(edges_files, f)
    # np.savetxt("distmat.txt", dist_mat)

    # étape 3B, construction de la matrice
    bw_files = glob.glob("black_white/*.tif")
    bw = [cv.imread(filename, 0) for filename in bw_files]



    # dist_mat = np.loadtxt("distmat.txt")
    # with open("labels.json") as f:
    #     labels = json.load(f)

    # étape 4, exploration

    # min_sum = float('inf')
    # min_i = None
    # for i in range(len(dist_mat)):
    #     for elem in dist_mat[i]:
    #         if elem > 0.1 and elem < min_sum:
    #             min_sum = elem
    #             min_i = i
    # print(min_sum, labels[min_i])
    # min_i = None
    # for i, l in enumerate(labels):
    #     if l.endswith("heic0707a_4320_10240.tif"):
    #         min_i = i
    #         print("bingo")
    # closest_100 = get_closest(dist_mat, labels, min_i)
    # for i, img in enumerate(closest_100):
    #     name = img.split('/')[-1]
    #     shutil.copyfile("decomposed_images/{}".format(name), "output/{:03d}.tif".format(i))

    # étape 5, clustering
    # n_clusters = 10
    # clustering = AgglomerativeClustering(n_clusters, affinity="precomputed", linkage='complete')
    # clustering.fit(dist_mat)
    # clust_labels = np.array(clustering.labels_)
    # for i in range(n_clusters):
    #     os.mkdir("clusters/{}".format(i))
    # for i, label in enumerate(labels):
    #     name = label.split('/')[-1]
    #     shutil.copyfile("decomposed_images/{}".format(name), "clusters/{}/{}".format(clust_labels[i], name))












# img = cv.imread('heic1509a.jpg', 0)
# rotated = imutils.rotate_bound(img, 45)
# img1 = cv.imread('sub_image1.png', 0)
# img2 = cv.imread('sub_image2.png', 0)
# edges1 = cv.Canny(img1, 10, 50)
# print(get_white_proportion(edges1))
# edges2 = cv.Canny(img2, 40, 80)
# print(get_white_proportion(edges2))
#
# plt.subplot(121), plt.imshow(img2, cmap='gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(edges2, cmap='gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()
