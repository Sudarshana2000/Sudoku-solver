''' Solving 9 X 9 sudoku puzzle using Backtracking method '''

def check_emptycell(puzzle):
    for i in range(9):
        for j in range(9):
            if puzzle[i,j] == 0:
                return [i,j]
    return None


def check_row(puzzle,row,col,n):
    for j in range(9):
        if puzzle[row,j] == n:
            return False
    return True


def check_col(puzzle,row,col,n):
    for i in range(9):
        if puzzle[i,col] == n:
            return False
    return True


def check_box(puzzle,row,col,n):
    for i in range(row - row%3, row - row%3 + 3):
        for j in range(col - col%3, col - col%3 + 3):
            if puzzle[i,j] == n:
                return False
    return True


def check_safe_cell(puzzle,row,col,n):
    return (check_box(puzzle, row, col, n) and check_row(puzzle, row, col, n) and check_col(puzzle, row, col, n))


def solve_sudoku(grid):
    loc=check_emptycell(grid)
    if loc is None:
        return True
    for n in range(1,10):
        if check_safe_cell(grid, loc[0], loc[1], n):
            grid[loc[0],loc[1]]=n
            if solve_sudoku(grid):
                return True
            grid[loc[0],loc[1]]=0
    return False


def display_sudoku(board):
    print("\n+---------------+---------------+---------------+")
    for i in range(9):
        print('|',end='\t')
        for j in range(9):
            if board[i,j]!=0:
                print(board[i,j],end="\t")
            else:
                print(" ",end="\t")
            if((j+1)%3==0):
                print('|',end='\t')
        if (i+1)%3==0:
            print("\n+---------------+---------------+---------------+")
        else:
            print()


def main(sudoku):
    if solve_sudoku(sudoku):
        print("Solved successfully!")
    else:
        print("No solution exist...")
    display_sudoku(sudoku)
    return sudoku