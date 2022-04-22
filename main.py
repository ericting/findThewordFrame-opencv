import cv2 as cv
from cv2 import imshow
from cv2 import blur
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('./demo.jpg',1)

def method(image):
    """
    去除图像噪声然后二值化图像
    """
    blurred = cv.pyrMeanShiftFiltering(image, 10, 100)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    show("gray",gray)
    t, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    show("binary",binary)
    return binary

def findOutline(binary,img):
    """
    找到轮廓并在，并删除较小的轮廓
    """
    # 筛选后的区域
    # region=[]
    contours, hierarchy = cv.findContours(binary,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    # # 遍历轮廓删除较小的点
    # for i in range(len(contours)):
    #     cnt=contours[i]
    #     if(cnt>1000):
    #         region.append(cnt)
    # show("region",region)
    
    #在img中画出轮廓
    cv.drawContours(img,contours,-1,(0,0,255),3)
    show("contours",contours)
    return img


def show(titles,images):
    """
    show me
    """
    plt.subplot(1,1,1),plt.imshow(images,'gray')
    plt.title(titles)
    plt.xticks([]),plt.yticks([])
    plt.show()

def gain(image):
    """
    获取黑色区域的
    """
    blurred = cv.pyrMeanShiftFiltering(image, 10, 100)
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
 
    # HSV中黑色范围
    lower_blue = np.array([0,0,0]) 
    upper_blue = np.array([180,255,46]) 
 
    # 获得黑色区域的mask
    mask = cv.inRange(hsv, lower_blue, upper_blue)

    return mask
    
 
def main(image):
    show("Original Image",image)
    binary=method(image)
    contours=findOutline(binary,image)
    cv.imwrite("result.jpg",contours)


if __name__ == '__main__':
    main(img)



