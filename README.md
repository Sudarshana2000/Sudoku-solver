# Sudoku-solver

Being a Sudoku lover, I dedicate this Web-App for all Sudoku learners and pros who can easily check for solutions...


## Specification

- HTML, CSS, Javascript in Frontend
- OpenCV
- Python 3.8
- TensorFlow-Keras
- Django in backend
- MNIST Dataset (Available with Keras datasets)


## Overview

The main tasks to do in the pipeline are:
- Creating the algorithm for solving Sudoku
- Extracting Sudoku from the image
- Solving Sudoku and displaying results
- Developing the Web-App


### Algorithm for solving Sudoku

The most obvious way to solve a partially-filled sudoku grid is to use backtracking algorithm - just check for a suitable number in each cell that doesn't repeat in its row, column and 3X3 subgrid, until all cells are successfully filled up.

The algorithm is implemented [here](SudokuApp/firstPage/SolveSudoku.py).


### Extracting Sudoku from the source image

This is implemented using OpenCV, Python and TensorFlow-Keras as follows:

1. Create a model for digit-recognition by training neural network over MNIST dataset containing 60,000 images of digits from 0 to 9.
2. Preprocess the image to detect sudoku grid.
3. Extract the sudoku grid and warp the image to perspective view.
4. Extract each cell by slicing the grid sequentially.
5. Using the saved model, detect digit for every cell, otherwise mark it empty.
6. Save the values of the sudoku grid in a matrix and return it.


### Solving Sudoku and displaying results

1. Get the sudoku matrix.
2. Solve the sudoku using the algorithm.
3. Print the solution over the source image and return output.


### Integrating functionalities and develop the Web-App

The Web-App is designed as such:

1. **The user needs to upload the image.**
<img src="https://github.com/Sudarshana2000/Sudoku-solver/blob/master/images/IMG1.JPG" />
<br />
2. **Once uploaded, press Solve.**
<img src="https://github.com/Sudarshana2000/Sudoku-solver/blob/master/images/IMG2.JPG" />
<br />
3. **The solution to the given Sudoku is provided.**
<img src="https://github.com/Sudarshana2000/Sudoku-solver/blob/master/images/IMG3.JPG" />
<br />