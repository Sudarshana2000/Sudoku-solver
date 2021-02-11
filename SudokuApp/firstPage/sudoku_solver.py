'''Main function to extract and solve Sudoku'''

import cv2
import numpy as np
from skimage.segmentation import clear_border
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from .SolveSudoku import main

model=load_model("F:/models/digit_classifier.h5")


def four_point_transform(image,points):
    (tr,tl,bl,br)=points
    rect=np.asarray([tl,tr,br,bl])
    width1=np.sqrt(((br[0]-bl[0])**2)+((br[1]-bl[1])**2))
    width2=np.sqrt(((tr[0]-tl[0])**2)+((tr[1]-tl[1])**2))
    width=min(int(width1),int(width2))
    height1=np.sqrt(((tr[0]-br[0])**2)+((tr[1]-br[1])**2))
    height2=np.sqrt(((tl[0]-bl[0])**2)+((tl[1]-bl[1])**2))
    height=min(int(height1),int(height2))
    dest=np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],dtype="float32")
    M=cv2.getPerspectiveTransform(rect,dest)
    warped=cv2.warpPerspective(image,M,(width,height))
    return warped


def find_puzzle(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(7,7),0)
    thresh=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    contours,hierarchy=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours=sorted(contours,key=cv2.contourArea,reverse=True)
    border=None
    for c in contours:
        peri=cv2.arcLength(c,True)
        approx=cv2.approxPolyDP(c,0.02*peri,True)
        if len(approx)==4:
            border=approx
            break
    if border is None:
        print("Image Not Clear!")
        puzzle=None
    else:
        # res=image.copy()
        # res=cv2.drawContours(res,[border],-1,(0,255,0),2)
        # cv2.imshow("image",res)
        # cv2.waitKey(0)
        border=border.astype(np.float32)
        puzzle=four_point_transform(image, border.reshape(4,2))
    return puzzle


def preprocess_digit(cell):
    gray=cv2.cvtColor(cell,cv2.COLOR_BGR2GRAY)
    roi=cv2.medianBlur(gray,5)
    thresh=cv2.threshold(roi,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
    thresh=clear_border(thresh)
    contours,hierarchy=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)==0:
        return None
    c=max(contours,key=cv2.contourArea)
    res=np.zeros(thresh.shape,dtype="uint8")
    res=cv2.drawContours(res,[c],-1,(255,255,255),-1)
    h,w=thresh.shape[:2]
    if cv2.countNonZero(res)/float(h*w) < 0.03:
        return None
    digit=cv2.bitwise_and(thresh,thresh,mask=res)
    return digit


def extract_digits(puzzle):
    h,w=puzzle.shape[:2]
    h=h//9
    w=w//9
    board=np.zeros((9,9),dtype="int")
    for i in range(9):
        for j in range(9):
            cell=puzzle[i*h:(i+1)*h,j*w:(j+1)*w]
            cell=preprocess_digit(cell)
            if cell is not None:
                cell=cv2.resize(cell,(28,28))
                # cv2.imshow("roi",cell)
                # cv2.waitKey(0)
                cell=cell.astype("float")/255.0
                cell=img_to_array(cell)
                cell=np.expand_dims(cell, axis=0)
                board[i,j]=model.predict(cell).argmax(axis=1)[0]
    return board


def display(image,board):
    h,w=image.shape[:2]
    h=h//9
    w=w//9
    for i in range(9):
        for j in range(9):
            output=cv2.putText(image, str(board[i,j]), (int((j+0.33)*w), int((i+0.8)*h)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return output


def process(image):
    #image=cv2.resize(image,(600,int(600*(image.shape[0]/image.shape[1]))))
    puzzle=find_puzzle(image)
    puzzle=cv2.resize(puzzle,(450,450))
    grid=extract_digits(puzzle)
    #SolveSudoku.display_sudoku(grid)
    grid=main(grid)
    #display(puzzle,grid)
    return display(puzzle,grid)
