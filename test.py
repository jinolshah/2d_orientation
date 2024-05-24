import cv2
from rembg import remove

img = cv2.imread(r"template_images/template_1.jpg")
gray = cv2.cvtColor(remove(img), cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), cv2.BORDER_DEFAULT)
blur = cv2.GaussianBlur(blur, (5,5), cv2.BORDER_DEFAULT)

# _, thresh = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)

img2 = cv2.Canny(blur, 60, 100)

contours, _ = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

final_img = cv2.resize(img, (0, 0), fx = 0.75, fy = 0.75)
cv2.imshow('test', final_img)

cv2.waitKey(0)