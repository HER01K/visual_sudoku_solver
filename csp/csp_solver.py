"""
CSP Sudoku Solver
Milestone 1: Samuel Brooke

Conventions:
  - Grid = flat list of 81 ints, row-major, 0 = empty, 1-9 = digit.
    index = 9 * row + col
  - Class 0 = "empty", classes 1-9 = digits.

I/O:
  Input : probabilities, a list of 81 arrays, 
          each length 10 (probs[i][c] = P(cell i is class c))
  Output: list of 81 (int class, bool is_input) pairs
          is_input = True  -> cell is a given clue
          is_input = False -> cell was filled in by the solver
"""

from typing import Optional

# 1. Problem structure, basic rules of sudoku

def _build_units() -> list[list[int]]:
    """Return the 27 units (9 rows, 9 cols, 9 boxes)"""
    lst = []

    for i in range(9):
        sublst = []
        for j in range(9):
            sublst.append(9 * i + j)
        lst.append(sublst)
    for i in range (9):
        sublst = []
        for j in range(9):
            sublst.append(9 * j + i)
        lst.append(sublst)
    for i in range (3):
        for j in range (3):
            sublst = []
            for k in range (3):
                for m in range (3):
                    sublst.append(i*27 + j*3 + k*9 + m)
            lst.append(sublst)

    return lst

def _build_peers(units: list[list[int]]) -> list[set[int]]:
    """Return PEERS, where PEERS[i] is the set of the 20 cells sharing a unit with cell i."""
    
    peers = []
    
    for i in range(81):
        cell_peers = set()
        for unit in units:
            if i in unit:
                cell_peers.update(unit)
        cell_peers.discard(i)  # a cell isn't its own peer
        peers.append(cell_peers)
    
    return peers

# Universal UNITS and PEERS
UNITS = _build_units()
PEERS = _build_peers(UNITS)

# 2. Reading the classifier output

def probabilities_to_grid(probs) -> list[int]:
    """Most likely reading of each cell (argmax over the 10 classes)."""

    lst = []

    for row in probs:
        lst.append(max(range(10), key=lambda c: row[c]))
    
    return lst

# 3. CSP core: domains, consistency check, propagation

def initial_domains(grid: list[int]) -> Optional[list[set[int]]]:
    """
    Given cells get domain {d}; empty cells get {1..9}.
    Return None immediately if two clues already conflict (same digit in one unit).
    """
    
    lst = []

    for i in range(len(grid)):
        if grid[i] == 0:
            lst.append(set(range(1, 10)))
        else:
            lst.append({grid[i]})
            for j in PEERS[i]:
                if grid[j] == grid[i]:
                    return None

    return lst

def obtain_singleton_cells(domains: list[set[int]]) -> list[int]:

    lst = []

    for i in range(len(domains)):
        if len(domains[i]) == 1:
            lst.append(i)
    
    return lst


def propagate(domains: list[set[int]]) -> bool:
    """
    Constraint propagation, modifying domains in place.
    When a cell has one value left, remove it from its peers.
    Return False if any domain becomes empty (i.e. contradiction).
    
    TODO Milestone 2: hidden singles and AC-3 methods for slight efficiency bump
    """
    queue = obtain_singleton_cells(domains)

    while len(queue) != 0:
        i = queue.pop(0)
        v = next(iter(domains[i]))
        for j in PEERS[i]:
            if v in domains[j]:
                domains[j].remove(v)
                if len(domains[j]) == 0:
                    return False
                if len(domains[j]) == 1:
                    queue.append(j)
    
    return True


# 4. Backtracking search (counts solutions, stops at `limit`)

def obtain_smallest_domain_cell(domains: list[set[int]]) -> int:
    # returns the INDEX of the smallest domain-size cell
    best = -1

    for i in range(len(domains)):
        size = len(domains[i])
        if size > 1 and (best == -1 or size < len(domains[best])):
            best = i

    return best


def search(domains: list[set[int]], solutions: list[list[int]], limit: int = 2) -> None:
    """
    Backtracking with propagation.
      - pick an unassigned cell (start with smallest domain > 1)
      - recursively try each value on a COPY of the domains, propagate
      - when all domains are size 1, append the grid to `solutions`
      - stop once len(solutions) >= limit
    """
    if len(solutions) >= limit:
        return
    
    i = obtain_smallest_domain_cell(domains)
    if i == -1:
        solutions.append([next(iter(d)) for d in domains])
        return 
    
    for v in domains[i]:
        domains_copy = [set(d) for d in domains]
        domains_copy[i] = {v}
        if propagate(domains_copy):
            search(domains_copy, solutions, limit)
        if len(solutions) >= limit:
            return


def solve_grid(grid: list[int], limit: int = 2) -> list[list[int]]:
    """Solve a plain int grid. Returns up to `limit` solutions (0, 1, or 2+ matters)."""
    
    domains = initial_domains(grid)
    
    if domains is None:
        return []
    if not propagate(domains):
        return []
    
    solutions = []
    search(domains, solutions, limit)
    return solutions


# 5. Error correction (Milestone 2): use CNN confidences

def correct_and_solve(probs, grid: list[int]) -> Optional[tuple[list[int], list[int]]]:
    """
    Called when the argmax grid has 0 or 2+ solutions.
    Idea: find alternative readings that give exactly one solution, preferring
    the ones that change the least-confident cells.
      - rank cells by confidence of their chosen class
      - try changing 1 cell (to its next most likely class), then 2, ...
      - score each candidate by total log-probability of the changes
    Return (corrected_grid, solution) or None.
    """
    # TODO: leave for Milestone 2
    raise NotImplemented


# 6. Public entry point

def solve_sudoku(probs) -> Optional[list[tuple[int, bool]]]:
    """
    Pipeline entry point.
    Returns 81 (class, is_input) pairs, or None if no solution could be found.
    """
    grid = probabilities_to_grid(probs)
    solutions = solve_grid(grid)

    if len(solutions) == 1:
        clues = grid
        solution = solutions[0]
    else:
        # TODO (Milestone 2): fall back to correct_and_solve(probs, grid)
        return None

    # is_input is True where the (possibly corrected) reading was a nonzero clue
    return [(solution[i], clues[i] != 0) for i in range(81)]


# 7. Tests: write these first

if __name__ == "__main__":
    # TODO: test with a known easy puzzle, a hard puzzle, an unsolvable one
    # (conflicting clues), and one with multiple solutions (very few clues).
    pass