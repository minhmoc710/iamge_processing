import Detector
import cv2

imgTest = cv2.imread('ImgTest.jpg')
newImg = Detector.detect(imgTest)
cv2.imshow('Result', newImg)
cv2.waitKey(5000)
cv2.destroyAllWindows()