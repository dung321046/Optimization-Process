from ortools.linear_solver import pywraplp


def LinearProgrammingExample():
    """Linear programming sample."""
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        return
    X = []
    for i in range(2):
        X.append(solver.NumVar(0, solver.infinity(), 'x' + str(i)))
    print('Number of variables =', solver.NumVariables())
    solver.Add(4 *X[0] + 2 * X[1] <= 60.0)
    solver.Add(2 *X[0] + 4 * X[1] <= 48.0)

    print('Number of constraints =', solver.NumConstraints())
    solver.Maximize(8 * X[0] + 6 * X[1])
    # [END objective]

    # Solve the system.
    # [START solve]
    status = solver.Solve()
    # [END solve]

    # [START print_solution]
    if status == pywraplp.Solver.OPTIMAL:
        print('Solution:')
        print('Objective value =', solver.Objective().Value())
        print('X_0 =', X[0].solution_value())
        print('X_1 =', X[1].solution_value())
    else:
        print('The problem does not have an optimal solution.')
    # [END print_solution]

    # [START advanced]
    print('\nAdvanced usage:')
    print('Problem solved in %f milliseconds' % solver.wall_time())
    print('Problem solved in %d iterations' % solver.iterations())
    # [END advanced]


LinearProgrammingExample()