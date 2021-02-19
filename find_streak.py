import numpy as np
import cv2 as cv
import imutils
import glob
import json
import sys
import shutil
import os
import random
from image_container import ImageContainer
from multiprocessing import Pool
from itertools import repeat
from stringfish import strexe


def get_sub_rectangle(img, x, y, x_length=720, y_length=1280, angle=0):
    if angle==0:
        return img[x:x+x_length, y:y+y_length]


def make_rotated(img):
    rotated = []
    for rot in [cv.ROTATE_90_CLOCKWISE, cv.ROTATE_180, cv.ROTATE_90_COUNTERCLOCKWISE]:
        rotated.append(cv.rotate(img.copy(), rot))
        print("just rotated")
    return rotated


def make_grid(img_container, step, x_length=720, y_length=1280):
    width = img_container.width
    height = img_container.height
    offset = int(np.sqrt(x_length**2 + y_length**2)/2)
    grid = []
    for i in range(offset, height-offset, step):
        for j in range(offset, width-offset, step):
            grid.append([i, j])
    return grid



def scan_image(target_image, img_container, remove_rectangle=False,
               x_length=720, y_length=1280, step=100):
    grid = make_grid(img_container, step, x_length, y_length)
    results = []
    for angle in np.arange(0, np.pi, np.pi / 8):
        for flip in [True, False]:
            results.append(scan_image_helper(target_image, img_container, grid, angle, flip,
                                             x_length, y_length))
    return get_best_result(results)


def scan_image_helper(target_image, img_container, grid, angle, flip,
                      x_length=720, y_length=1280):
    """ scan_coords = [x_start, x_end, y_start, y_end]
    """
    min_dist = float('inf')
    best_match = None
    best_params = []
    for point in grid:
        sub_img = img_container.get_sub_img_rotated_rect(point, angle, flip,
                                                         height=x_length, width=y_length)
        if sub_img is not None:
            dist = get_distance(target_image, sub_img)
            if dist < min_dist:
                min_dist = dist
                best_match = sub_img
                best_params = [point, angle, flip]
    return min_dist, best_match, best_params


def get_distance(img1, img2):
    diff = img1-img2
    dist = np.multiply(diff, diff)
    return np.sum(dist)/(dist.size*255)


def find_closest(n_closest, starting_image, img_containers, remove_rectangle=False, step=300):
    closest = [starting_image]
    dists = [0]
    current_image = starting_image
    for i in range(1, n_closest):
        results = []
        for img_container in img_containers:
            results.append(scan_image(current_image, img_container, remove_rectangle=remove_rectangle, step=step))
        # results = p.starmap(scan_image, zip(repeat(current_image), img_containers))
        min_dist, best_img, best_params, best_j = get_best_result(results)
        img_containers[best_j].add_params(best_params)
        closest.append(best_img)
        dists.append(min_dist)
        print(i, min_dist)
        current_image = best_img
    return closest, dists


def get_best_result(results):
    min_dist = float('inf')
    best_img = None
    best_coords = []
    best_j = None
    for j, r in enumerate(results):
        if r[0] < min_dist:
            min_dist = r[0]
            best_img = r[1]
            best_coords = r[2]
            best_j = j
    return min_dist, best_img, best_coords, best_j


def write_cut_images(filename, target_size=3000):
    img = cv.imread(filename)
    name = filename.split('/')[-1][:-4]
    height = img.shape[0]
    width = img.shape[1]
    if height < target_size or width < target_size:
        return
    counter = 0
    for i in range(0, height-target_size, target_size):
        for j in range(0, width-target_size, target_size):
            cv.imwrite("cut_images/{}_{}.tif".format(name, counter), img[i:i+target_size, j:j+target_size])
            print(counter)
            counter += 1


def move_limit_images_back():
    cut_filenames = glob.glob("cut_images/*.tif")
    n_dict = dict()
    for cf in cut_filenames:
        key = cf.split('/')[-1].split('_')[0]
        if key in n_dict:
            n_dict[key] += 1
        else:
            n_dict[key] = 1
    for key in n_dict:
        if n_dict[key] < 5:
            strexe("rm cut_images/{}*.tif".format(key))
            filenames = glob.glob("raw_images/{}*.tif".format(key))
            assert len(filenames) == 1
            shutil.copyfile(filenames[0], "cut_images/{}_0.tif".format(key))








if __name__ == "__main__":
    # filenames = glob.glob("raw_images/*.tif")
    # for filename in filenames:
    #     write_cut_images(filename)
    move_limit_images_back()

    # img_containers = [ImageContainer(img, param_type='point')]







    # closest_10, dists = find_closest(10, main_sub, img_containers, step=600)
    # dirpath = "color_distance"
    # for i, closest in enumerate(closest_10):
    #     cv.imwrite("{}/img{:03d}.tif".format(dirpath, i), closest)





