## Kamera Kalibrasyonu ve Distorsiyon Giderme
- Satranç tahtası desenleri kullanarak bir kameranın içsel parametrelerini hesaplayan ve lens bozulmalarını düzelten Python/OpenCV uygulaması.

# Kamera Kalibrasyonu
- Görüntüler, 3B nesneleri gerçek dünyadan 2B görüntülere dönüştürmek için lensler kullanan kameralar tarafından yakalanır. Bununla birlikte, dönüştürme sırasında lensler kullanıldığından, resimlerde bazı bozulmalar da ortaya çıkar. Bozuk görüntülerin yakalanmasını önlemek için, gerçek dünyadaki bir 3B noktayı görüntüdeki eşleşen 2B projeksiyonu (piksel) ile doğru bir şekilde ilişkilendirmek için kameranın kalibre edilmesi gerekir. Bu nedenle, kamera kalibrasyonu, Opencv'deki calibrateCamera() işlevi tarafından gerçekleştirilen bozulmamış bir görüntü yakalamak için kameranın parametrelerinin belirlenmesi anlamına gelir.

# Adımlar
- Öncelikle gerçek dünya için 3d noktalar hazırlanır. Düz bir düzlem üzerinde kamera kalibrasyonu yapılacak ise z değeri 0 verilir. Örneğin bir satranç tahtasında 1. iç kenar için (0,0) ardından (0,1), (0,2)...
- Ardından görsel üzerindeki köşeler gerekli işlemler ile tespit edilir.
- Bu iki koordinat noktaları iki ayrı dizide tutulur. 1. resim için örneğin gerçek dünya koordinatları a[0] dizisinde ise resim koordinat noktaları da b[0] dizisinde olmalıdır.
- Bu işlemlerden sonra artık kamera kalibrasyonu işlemi için kamera matrisini, bozulma katsayılarını, döndürme ve öteleme vektörlerini vb. döndüren ```cv.calibrateCamera()``` fonksiyonu kullanılır.
- Ardından ```cv.getOptimalNewCameraMatrix()``` fonksiyonu ile yeni optimize edilmiş kamera matrisi ve düzeltme işlemi yapılacak alanın roi’si alınır. Bu fonksiyon mtx, dist, görüntü boyutu, alpha değeri ve tekrardan görüntü boyutunu parametre olarak alır. Burada alpha değeri eğer 0 olursa, keskin bir düzeltme yapılır, kenarları kırpabilir ama istenmeyen (siyah/bozuk) pikselleri ortadan kaldırır. Eğer 1 olursa tüm pikselleri korur ama kenarlarda siyah boşluklar kalabilir.
- İlk yöntem: ```cv.undistort()``` fonskiyonu kullanımı ve aldığı parametreler şu şekildedir; ```cv.undistort(img, mtx, dist, None, newcameramtx)```
- İkinci yöntem olan remapping yöntemi biraz daha gelişmiş bir yöntemdir. ```Cv.initUndistortRectifyMap``` fonksiyonu kullanılır. Eski bozuk görüntüdeki her pikselin yeni görüntüde nereye denk geldiğini hesaplar. Mapx ve mapy adlı iki değişken döndürür. Sonrasında``` cv.remap(img, mapx, mapy, cv.INTER_LINEAR)``` fonksiyonu ile bu mapler kullanılarak görüntü yeniden şekillendirilir. 

## KOD AÇIKLAMASI ## 
``` 
import cv2 as cv
import numpy as np
import glob 

class CamCal():

    def __init__(self):

        # satranç tahtasında köşe bulma algoritması için parametreler; iterasyonun ne zaman biteceğini belirler.
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Gerçek dünya koordinat sistemi tanımlaması için satranç tahtasının boyutuna göre diziler oluşturulur.
        self.objp = np.zeros((6*7, 3), np.float32)
        self.objp[:,:2] = np.mgrid[0:7, 0:6].T.reshape(-1,2)

        # Gerçek dünya ve görüntü koorditanlarını kalibrasyonda kullanmak için diziler oluşturulur.
        self.objpoint = []
        self.imgpoint = []

    def points(self): 
        # Dataset klasöründeki ".jpg" uzantılı dosyalar okunur
        images = glob.glob("Dataset/*.jpg")
        
        for fname in images:
            print(f"İşleniyor {fname}")
            # Görüntü opencv tarafından okunup gri görüntüye çevirilir.
            img = cv.imread(fname)
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # 2D Düzlem koordinatları belirlenir.
            ret, corners = cv.findChessboardCorners(img_gray, (7,6), None)

            # Eğer koordinatlar bulunduysa
            if ret == True:
                print("Köşeler bulundu")
                # Gerçek dünya koordinatlarını objpoint listesine ekle.
                self.objpoint.append(self.objp)

                # findchessboardCorners fonkisyonu ile tespit edilen köşeleri daha hassas bir şekilde yeniden hesaplar.
                corners2 = cv.cornerSubPix(img_gray, corners, (11,11), (-1,-1), self.criteria)
                # 2D koordinatları image listesine ekler
                self.imgpoint.append(corners2)

            else:
                print("Köşe bulunamadı")
        self.calibration(img)

    def calibration(self, img):
        #görüntünün boyutu alınır.
        h, w = img.shape[:2]
        # Kalibrasyon hesaplamaları yapılır.
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(self.objpoint, self.imgpoint,(w,h), None, None)
        # optimum kamera matrisi hesaplaması tekrardan yapılır.
        newcammatrix, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

        # Düzeltme işlemleri yapılır ve map olarak alınır.
        mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcammatrix, (w,h), 5)
        # Map koordinatları çevirilir.
        dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

        # ilgilenilen bölüm kırpılır.
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        cv.imwrite("output.jpg", dst)
        print("Düzenlenmiş görüntü output.jpg olarak kaydedildi.")
        print(f"Kamera matrisi: \n{mtx}")

if __name__ == "__main__":
    run = CamCal()
    run.points()


``` 