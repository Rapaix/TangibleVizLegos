import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def first_nonzero(arr, axis, invalid_val=0):
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def last_nonzero(arr, axis, invalid_val=0):
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)

# returns the indexes indicating the bounding box of lego area
def getsubmatrix(mat):
    print mat

def getinrange(mat,min,max):
    mask = cv.inRange(mat, min, max)
    newmat = cv.bitwise_and(mat, mat, mask=mask)
    return newmat

def getcontours(img):
    gray = cv.cvtColor(img, cv.COLOR_HSV2RGB)
    gray = cv.cvtColor(gray, cv.COLOR_RGB2GRAY)
    ret, thresh = cv.threshold(gray, 127, 255, 0)
    im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return im2, contours, hierarchy

def getcontoursapprox(cnts):
    ap = []
    for cnt in cnts:
        approx = cv.approxPolyDP(cnt, 0.01*cv.arcLength(cnt, True), True)
        ap.append(approx)
    return ap

def getwhitebbox(img):
    white = img == 255
    black = np.bitwise_not(white)


if __name__ == '__main__':
    img = cv.imread('./images/original/thick.jpg', 1)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    #passar o blur aqui
    adjusted = cv.GaussianBlur(img_hsv,(5,5),0)
    bright = adjusted.copy()

    # bright = bright + np.array((0,0,10), dtype=np.int16)
    # bright = cv.convertScaleAbs(bright)

    # cortar cantos
    thresh = bright.copy()[200:-100, 400:-200]

    min_white = np.array([0, 0, 0], dtype=np.uint8)
    max_white = np.array([180, 150, 150], dtype=np.uint8)
    thresh = getinrange(thresh, min_white, max_white)
    #bright = cp.copy()
    thresh[:, :] = np.where(thresh[:, :] != (0, 0, 0), (128, 0, 128), (0, 0, 0))

    cp = thresh.copy()
    cp = cv.cvtColor(thresh, cv.COLOR_HSV2RGB)
    cp = cv.cvtColor(cp, cv.COLOR_RGB2GRAY)

    left, right = first_nonzero(cp, 1), last_nonzero(cp, 1)
    top, bottom = first_nonzero(cp, 0), last_nonzero(cp, 0)

    il, it = first_nonzero(left, 0), first_nonzero(top,0)
    l,t = left[il+10], top[it+100] # mediana ou media

    ir, ib = first_nonzero(right, 0), first_nonzero(bottom,0)
    r,b = right[cp.shape[0] - ir], bottom[cp.shape[0] - ib]

    test_cp = thresh.copy()

    cv.circle(test_cp,(l,t),4,(0,255,0),100)
    cv.circle(test_cp, (r, b), 4, (0, 255, 255), 100)

    clrcp = cv.cvtColor(thresh, cv.COLOR_HSV2RGB)
    bright = cv.cvtColor(bright, cv.COLOR_HSV2RGB)

    fig, axes = plt.subplots(1, 3)
    for ax in axes:
        ax.clear()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    axes[2].imshow(clrcp,cmap='gray')
    axes[1].imshow(test_cp)
    axes[0].imshow(img)
    plt.tight_layout()
    plt.show()



