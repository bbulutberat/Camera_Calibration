import cv2 as cv

img_right = cv.imread("image_right.png")
img_left = cv.imread("image_left.png")

stereo = cv.StereoBM_create(numDisparities = 16, 
                            blockSize = 15)

dis = stereo.compute(img_left, img_right)
norm_dis = cv.normalize(dis, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

cv.imshow("Dis", norm_dis)

cv.waitKey(0)
cv.destroyAllWindows()


