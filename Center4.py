import cv2
import numpy as np
from glob import glob

#指定フォルダ内のBMP画像のpathを取得する
dir_name = '20211207/'
img_path_all = glob(f'{dir_name}*.bmp')
#print(img_path_all)
#BMP画像を1枚ずつ読み込み処理する

cntt = 0
area = 112*4

for img_path in img_path_all:
	imgo = cv2.imread(img_path,0)	#画像読込 1280*1080 640*540 320*270
	img = cv2.medianBlur(imgo,5)
	#img = cv2.medianBlur(img,5)
	#img = cv2.medianBlur(img,5)
	#img = cv2.medianBlur(img,5)
	img = cv2.medianBlur(img,3)
	ret2, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)	#二値化

	img = img[300:780, 420:880]
	img = cv2.bitwise_not(img)

	mu = cv2.moments(img, False)
	y,x= int(mu["m10"]/mu["m00"]) , int(mu["m01"]/mu["m00"])

	img = cv2.circle(img,(y,x),150,(0,0,255), thickness=1)

	# cv2.imwrite(img_path[:-4]+'.png', img)	#保存する

	cntt=cntt+1
	xx=300+x
	yy=420+y

	img = imgo[xx-area : xx+area, yy-area : yy+area]	#円中心から指定範囲512*512を切り抜く
	img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)	#切り抜いた画像を縮小する
	cv2.imwrite(img_path[:-4]+'.jpg', img)	#保存する

	print(cntt, xx, yy)



