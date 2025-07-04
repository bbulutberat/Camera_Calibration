import cv2 as cv
import numpy as np
import glob 

class CamCal():

    def __init__(self):

        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.objp = np.zeros((6*7, 3), np.float32)
        self.objp[:,:2] = np.mgrid[0:7, 0:6].T.reshape(-1,2)

        self.objpoint = []
        self.imgpoint = []

    def points(self): 
        images = glob.glob("Dataset/*.jpg")
        
        for fname in images:
            print(f"İşleniyor {fname}")
            img = cv.imread(fname)
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            ret, corners = cv.findChessboardCorners(img_gray, (7,6), None)

            if ret == True:
                print("Köşeler bulundu")
                self.objpoint.append(self.objp)

                corners2 = cv.cornerSubPix(img_gray, corners, (11,11), (-1,-1), self.criteria)
                self.imgpoint.append(corners2)

            else:
                print("Köşe bulunamadı")
        self.calibration(img)

    def calibration(self, img):
        h, w = img.shape[:2]
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(self.objpoint, self.imgpoint,(w,h), None, None)
        newcammatrix, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

        mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcammatrix, (w,h), 5)
        dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        cv.imwrite("output.jpg", dst)
        print("Düzenlenmiş görüntü output.jpg olarak kaydedildi.")
        print(f"Kamera matrisi: \n{mtx}")

if __name__ == "__main__":
    run = CamCal()
    run.points()


