import numpy as np
import cv2 as cv

class ImageContainer:

    def __init__(self, img):
        self.img = img
        self.height = len(self.img)
        self.width = len(self.img[0])
        self.rectangles = []

    def add_rectangle(self, x1, y1, x2, y2):
        self.rectangles.append([x1, y1, x2, y2])

    def get_sub_img(self, x1, y1, x2, y2):
        for rect in self.rectangles:
            if rectangles_overlap(x1, y1, x2, y2, rect[0], rect[1], rect[2], rect[3]):
                return None
        return self.img[x1:x2, y1:y2]

    def get_sub_img_rotated_rect(self, center, angle, flipped=False, height=720, width=1280):
        corners = get_rotated_corners(center, angle, height, width)
        if not self.contain(corners):
            return None
        [bottom_left, top_left, top_right, bottom_right] = corners

        cnt = np.array([
            [int(bottom_left[0]), int(bottom_left[1])],
            [int(top_left[0]), int(top_left[1])],
            [int(top_right[0]), int(top_right[1])],
            [int(bottom_right[0]), int(bottom_right[1])]
        ])
        print(cnt)
        rect = cv.minAreaRect(cnt)
        # return self.crop_minAreaRect(rect)
        box = cv.boxPoints(rect)
        box = np.intp(box)

        r_width = int(rect[1][0])
        r_height = int(rect[1][1])
        print(r_height, r_width)

        src_points = box
        src_points = np.array(box, dtype="float32")

        dst_points = np.array([[0, r_height-1],
                               [0, 0],
                               [r_width-1, 0],
                               [r_width-1, r_height-1]], dtype="float32")

        M = cv.getPerspectiveTransform(src_points, dst_points)
        warped = cv.warpPerspective(self.img, M, (r_width, r_height))
        if len(warped) > len(warped[0]):
            # rot = cv.ROTATE_90_CLOCKWISE
            # if angle > np.pi:
            #     rot = cv.ROTATE_90_COUNTERCLOCKWISE
            warped = cv.rotate(warped.copy(), cv.ROTATE_90_CLOCKWISE)
        if flipped:
            warped = cv.rotate(warped.copy(), cv.ROTATE_180)
        warped = warped[:720, :1280]
        return warped

    def crop_minAreaRect(self, rect):
        img = self.img
        angle = rect[2]
        rows, cols = img.shape[0], img.shape[1]
        M = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        img_rot = cv.warpAffine(img, M, (cols, rows))

        # rotate bounding box
        rect0 = (rect[0], rect[1], 0.0)
        box = cv.boxPoints(rect0)
        pts = np.intp(cv.transform(np.array([box]), M))[0]
        pts[pts < 0] = 0

        # crop
        img_crop = img_rot[pts[1][1]:pts[0][1],
                   pts[1][0]:pts[2][0]]

        return img_crop

    def contain(self, corners):
        for c in corners:
            if c[0] < 0 or c[1] < 0:
                return False
            if c[0] >= self.height or c[1] >= self.width:
                return False
        return True




def rectangles_overlap(l1x, l1y, r1x, r1y, l2x, l2y, r2x, r2y):
    # If one rectangle is on left side of other
    if l1x >= r2x or l2x >= r1x:
        return False
    # If one rectangle is above other
    if l1y >= r2y or l2y >= r1y:
        return False
    return True


def get_rotated_corners(center, angle, height, width, offset=2):
    cx, cy = center[0], center[1]
    h_over2 = int(height/2 +offset)
    w_over2 = int(width/2 + offset)
    bottom_left = [cx + h_over2, cy - w_over2]
    top_left = [cx - h_over2, cy - w_over2]
    top_right = [cx - h_over2, cy + w_over2]
    bottom_right = [cx + h_over2, cy + w_over2]
    corners = [bottom_left, top_left, top_right, bottom_right]
    print("corners", corners)
    for i, c in enumerate(corners):
        corners[i] = rotate(corners[i], center, angle)
    print("rotated corners", corners)
    return corners


def rotate(point, center, angle):
    tempx = point[0] - center[0]
    tempy = point[1] - center[1]

    rotx = tempx*np.cos(angle) - tempy*np.sin(angle)
    roty = tempx*np.sin(angle) + tempy*np.cos(angle)

    return [rotx + center[0], roty + center[1]]


