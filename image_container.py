
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


def rectangles_overlap(l1x, l1y, r1x, r1y, l2x, l2y, r2x, r2y):
    # If one rectangle is on left side of other
    if l1x >= r2x or l2x >= r1x:
        return False
    # If one rectangle is above other
    if l1y >= r2y or l2y >= r1y:
        return False
    return True


