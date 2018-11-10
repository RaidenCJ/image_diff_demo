import sys
import cv2
import numpy as np

#for test only
def matchAB(fileA, fileB):

    imgA = cv2.imread(fileA)
    imgB = cv2.imread(fileB)


    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    cv2.imshow("grayA", grayA)
    cv2.imshow("grayB", grayB)
    cv2.waitKey(0)

    height, width = grayA.shape


    result_window = np.zeros((height, width), dtype=imgA.dtype)
    for start_y in range(0, height-100, 10):
        for start_x in range(0, width-100, 10):
            window = grayA[start_y:start_y+100, start_x:start_x+100]
            match = cv2.matchTemplate(grayB, window, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(match)
            matched_window = grayB[max_loc[1]:max_loc[1]+100, max_loc[0]:max_loc[0]+100]
            result = cv2.absdiff(window, matched_window)
            result_window[start_y:start_y+100, start_x:start_x+100] = result


    _, result_window_bin = cv2.threshold(result_window, 30, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(result_window_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imgC = imgA.copy()
    for contour in contours:
        min = np.nanmin(contour, 0)
        max = np.nanmax(contour, 0)
        loc1 = (min[0][0], min[0][1])
        loc2 = (max[0][0], max[0][1])
        cv2.rectangle(imgC, loc1, loc2, 255, 2)

    #plt.subplot(1, 3, 1), plt.imshow(cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)), plt.title('A'), plt.xticks([]), plt.yticks([])
    #plt.subplot(1, 3, 2), plt.imshow(cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)), plt.title('B'), plt.xticks([]), plt.yticks([])
    #plt.subplot(1, 3, 3), plt.imshow(cv2.cvtColor(imgC, cv2.COLOR_BGR2RGB)), plt.title('Answer'), plt.xticks([]), plt.yticks([])
    #plt.show()
    cv2.imshow("result", imgC)
    cv2.waitKey(0)
    
#for test only
def diffAB(fileA, fileB):
    img1 = cv2.imread(fileA)
    img2 = cv2.imread(fileB)
    diff = cv2.absdiff(img1, img2)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    th = 1
    imask =  mask>th

    canvas = np.zeros_like(img2, np.uint8)
    canvas[imask] = img2[imask]

    #cv2.imwrite("result.png", canvas)
    cv2.imshow("result", canvas)
    cv2.waitKey(0)

def testAB(fileA, fileB):

    imgA = cv2.imread(fileA)
    imgB = cv2.imread(fileB)
    diff = cv2.absdiff(imgA, imgB)
    diff_lwpCV = cv2.GaussianBlur(diff, (15, 15), 0)
    cv2.imshow("diff_lwpCV", diff_lwpCV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10, 10))
    diffde = cv2.dilate(diff_lwpCV, kernel)
    diffde = cv2.erode(diffde, kernel)
    grayDiff = cv2.cvtColor(diffde, cv2.COLOR_BGR2GRAY)
    
    ret, binary = cv2.threshold(grayDiff,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow("binary", binary)
    _, contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
    for cnt in contours :        
        if cv2.contourArea(cnt) > 1000 :
            cv2.drawContours(imgB, [cnt], 0, (255,0,0), 3)
    cv2.imshow("imgB", imgB)
    cv2.imwrite("result.png", imgB)
    #cv2.imshow("Thresh", thresh)
    cv2.waitKey(0)

testAB(sys.argv[1], sys.argv[2])
