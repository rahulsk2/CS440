# -*- coding: utf-8 -*-
import numpy as np
import argparse
from solve import solve


def runs(rowcol):
    """
    Returns the set of nonzero runs for a given row or column
    For example, the row or column [1,1,0,2,2,0,1,2,1] returns
    [[2,1],[2,2],[1,1],[1,2],[1,1]]
    """
    run = []
    curr_run = [1, rowcol[0]]
    rowcol.append(0)
    for i in range(1, len(rowcol)):
        if rowcol[i] != curr_run[1]:
            if curr_run[1] != 0:
                run.append(curr_run)
            curr_run = [1, rowcol[i]]
        else:
            curr_run[0] += 1
    return run


class Nonogram:
    # Initializes the Nonogram object by reading the constraints from a file
    def __init__(self, filename):
        self.__filename = filename
        self.constraints = np.load(self.__filename)
        self.dim = len(self.constraints)

    def isValidSolution(self, solution):
        """Returns True if solution fits constraints, False otherwise"""
        solution = np.array(solution)
        dim0 = len(self.constraints[0])
        dim1 = len(self.constraints[1])
        if solution.shape != (dim0, dim1):
            return False
        for i in range(dim0):
            constraints = self.constraints[0][i]
            rowcol = []
            for j in range(dim1):
                rowcol.append(solution[i][j])
            if not (runs(rowcol) == constraints):
                return False

        for j in range(dim1):
            constraints = self.constraints[1][j]
            rowcol = []
            for i in range(dim0):
                rowcol.append(solution[i][j])
            if not (runs(rowcol) == constraints):
                return False
        return True


def main():
    parser = argparse.ArgumentParser(description='Nonogram runner')
    parser.add_argument('constraints_file', type=str,
                        help='.npy file containing constraints')
    args = parser.parse_args()
    nono = Nonogram(args.constraints_file)
    solution = solve(nono.constraints)
    if nono.isValidSolution(solution):
        print('\x1b[6;30;42m' + 'Success!' + '\x1b[0m')
    else:
        print('\x1b[3;30;41m' + 'Failure.' + '\x1b[0m')


if __name__ == "__main__":
    main()