import numpy as np
import pandas as pd
from datetime import datetime

sudoku = pd.read_csv("./sudoku.csv") # Loading puzzles from csv
sample = sudoku.loc[2020] # row 2020
print(sample)

from ortools.linear_solver import pywraplp


def solve_with_ip(grid: np.ndarray) -> (np.ndarray, float):
    '''Solve Sudoku instance (np.matrix) with IP modeling. Returns a tuple with the resulting matrix and the execution time in seconds.'''
    assert grid.shape == (9, 9)

    grid_size = 9
    cell_size = 3  # np.sqrt(grid_size).astype(np.int)
    solver = pywraplp.Solver('Sudoku Solver', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)  # Step 1

    # Begin of Step2: Create variables.
    x = {}
    for i in range(grid_size):
        for j in range(grid_size):
            # Initial values.
            for k in range(grid_size):
                x[i, j, k] = solver.BoolVar('x[%i,%i,%i]' % (i, j, k))
    # End of Step2

    # Begin of Step3: Initialize variables in case of known (defined) values.
    for i in range(grid_size):
        for j in range(grid_size):
            defined = grid[i, j] != 0
            if defined:
                solver.Add(x[i, j, grid[i, j] - 1] == 1)
    # End of Step3

    # Begin of Step4: Initialize variables in case of known (defined) values.
    # All bins of a cell must have sum equals to 1
    for i in range(grid_size):
        for j in range(grid_size):
            solver.Add(solver.Sum([x[i, j, k] for k in range(grid_size)]) == 1)
    # End of Step4

    # Begin of Step5: Create variables.
    for k in range(grid_size):
        # AllDifferent on rows.
        for i in range(grid_size):
            solver.Add(solver.Sum([x[i, j, k] for j in range(grid_size)]) == 1)

        # AllDifferent on columns.
        for j in range(grid_size):
            solver.Add(solver.Sum([x[i, j, k] for i in range(grid_size)]) == 1)

        # AllDifferent on regions.
        for row_idx in range(0, grid_size, cell_size):
            for col_idx in range(0, grid_size, cell_size):
                solver.Add(solver.Sum([x[row_idx + i, j, k] for j in range(col_idx, (col_idx + cell_size)) for i in
                                       range(cell_size)]) == 1)
    # End of Step5

    # Solve and print out the solution.
    start = datetime.now()
    status = solver.Solve()  # Step 6
    exec_time = datetime.now() - start
    statusdict = {0: 'OPTIMAL', 1: 'FEASIBLE', 2: 'INFEASIBLE', 3: 'UNBOUNDED',
                  4: 'ABNORMAL', 5: 'MODEL_INVALID', 6: 'NOT_SOLVED'}

    result = np.zeros((grid_size, grid_size)).astype(np.int)
    if status == pywraplp.Solver.OPTIMAL:
        for i in range(grid_size):
            for j in range(grid_size):
                result[i, j] = sum((k + 1) * int(x[i, j, k].solution_value()) for k in range(grid_size))
    else:
        raise Exception('Unfeasible Sudoku: {}'.format(statusdict[status]))

    return result, exec_time.total_seconds()


res, _ = solve_with_ip(decoded_puzzle)
ip_solution = encode_sudoku(res)

assert ip_solution == sample['solution']  # must show the same solution for the puzzle found on the dataset
res