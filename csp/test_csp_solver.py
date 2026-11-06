"""
Tests for csp_solver.py
"""

import random

from csp_solver import (
    UNITS,
    PEERS,
    probabilities_to_grid,
    initial_domains,
    propagate,
    obtain_singleton_cells,
    obtain_smallest_domain_cell,
    search,
    solve_grid,
    solve_sudoku,
)

def parse(s: str) -> list[int]:
    """Turn an 81-character string ('.' or '0' = empty) into a flat int grid."""
    s = s.replace("\n", "").replace(" ", "")
    assert len(s) == 81
    return [0 if ch in ".0" else int(ch) for ch in s]


# Easy: basic puzzle yoinked from wikipedia
EASY = parse("""
    530070000
    600195000
    098000060
    800060003
    400803001
    700020006
    060000280
    000419005
    000080079
""")
EASY_SOLUTION = parse("""
    534678912
    672195348
    198342567
    859761423
    426853791
    713924856
    961537284
    287419635
    345286179
""")

# Hard: By Arto Inkala's, allegedly the "world's hardest sudoku"
HARD = parse("""
    8........
    ..36.....
    .7..9.2..
    .5...7...
    ....457..
    ...1...3.
    ..1....68
    ..85...1.
    .9....4..
""")

EMPTY = [0] * 81


# Helpers

def is_valid_solution(sol: list[int], puzzle: list[int]) -> bool:
    """True if sol is a complete valid sudoku that keeps every clue of puzzle."""
    if len(sol) != 81:
        return False
    if any(not (1 <= v <= 9) for v in sol):
        return False
    if any(puzzle[i] != 0 and puzzle[i] != sol[i] for i in range(81)):
        return False
    for unit in UNITS:
        if sorted(sol[i] for i in unit) != list(range(1, 10)):
            return False
    return True


def fake_probs(grid: list[int], confidence: float = 0.91) -> list[list[float]]:
    """made up CNN output"""
    other = (1.0 - confidence) / 9
    probs = []
    for d in grid:
        row = [other] * 10
        row[d] = confidence
        probs.append(row)
    return probs


def noisy_probs(grid: list[int], seed: int = 0) -> list[list[float]]:
    """
    Varying fake CNN output: the true class always wins, but with a different
    confidence (0.55 - 0.97) for each cell, and random leftover probability
    spread over the other 9 classes. Every row sums to 1.
    """
    rng = random.Random(seed)
    probs = []
    for d in grid:
        conf = rng.uniform(0.55, 0.97)
        weights = [rng.random() for _ in range(9)]
        total = sum(weights)
        leftover = [(1.0 - conf) * w / total for w in weights]
        row = []
        k = 0
        for c in range(10):
            if c == d:
                row.append(conf)
            else:
                row.append(leftover[k])
                k += 1
        probs.append(row)
    return probs


# 1. Problem structure

def test_units_structure():
    assert len(UNITS) == 27
    assert all(len(u) == 9 for u in UNITS)
    assert all(len(set(u)) == 9 for u in UNITS)
    # every cell is in exactly 3 units (its row, column, and box)
    assert all(sum(i in u for u in UNITS) == 3 for i in range(81))
    # every index is valid
    assert all(0 <= i <= 80 for u in UNITS for i in u)


def test_units_known_values():
    assert UNITS[0] == list(range(9))
    assert UNITS[8] == [72, 73, 74, 75, 76, 77, 78, 79, 80]    # row 8
    assert UNITS[9] == [0, 9, 18, 27, 36, 45, 54, 63, 72]      # column 0
    assert UNITS[18] == [0, 1, 2, 9, 10, 11, 18, 19, 20]       # top-left box
    assert UNITS[19] == [3, 4, 5, 12, 13, 14, 21, 22, 23]      # top-middle box
    assert UNITS[26] == [60, 61, 62, 69, 70, 71, 78, 79, 80]   # bottom-right box


def test_peers_structure():
    assert len(PEERS) == 81
    assert all(len(p) == 20 for p in PEERS)
    assert all(i not in PEERS[i] for i in range(81))
    # peer relation is symmetric
    assert all(i in PEERS[j] for i in range(81) for j in PEERS[i])


def test_peers_known_values():
    expected = set(range(1, 9)) | {9, 18, 27, 36, 45, 54, 63, 72} | {10, 11, 19, 20}
    assert PEERS[0] == expected
    assert 0 not in PEERS[40] and 80 not in PEERS[40]

# 2. probs_to_grid

def test_probs_to_grid_basic():
    probs = [[0.0] * 10 for _ in range(81)]
    probs[0][7] = 1.0
    probs[1][0] = 1.0
    probs[2][3] = 1.0
    grid = probabilities_to_grid(probs)
    assert len(grid) == 81
    assert grid[:3] == [7, 0, 3]


def test_probs_to_grid_roundtrip():
    assert probabilities_to_grid(fake_probs(EASY)) == EASY
    assert probabilities_to_grid(fake_probs(HARD)) == HARD
    assert probabilities_to_grid(noisy_probs(EASY, seed=1)) == EASY


# 3. initial_domains

def test_initial_domains_empty_grid():
    d = initial_domains(EMPTY)
    assert d is not None and len(d) == 81
    assert all(x == set(range(1, 10)) for x in d)


def test_initial_domains_clue_and_empty():
    g = [0] * 81
    g[0] = 5
    d = initial_domains(g)
    assert d[0] == {5}
    assert d[1] == set(range(1, 10))


def test_initial_domains_fresh_sets():
    # editing one empty cell's domain must not change any other cell's domain
    d = initial_domains(EMPTY)
    d[0].remove(1)
    assert 1 in d[1] and 1 in d[80]


def test_initial_domains_conflicts_return_none():
    row = [0] * 81
    row[0] = row[1] = 5                  # same row
    assert initial_domains(row) is None

    col = [0] * 81
    col[0] = col[9] = 3                  # same column
    assert initial_domains(col) is None

    box = [0] * 81
    box[0] = box[10] = 7                 # same box, different row and column
    assert initial_domains(box) is None


def test_initial_domains_no_false_conflict():
    # same digit in different rows, columns, and boxes is fine
    g = [0] * 81
    g[0] = 5
    g[40] = 5
    g[80] = 5
    assert initial_domains(g) is not None
    # real puzzles are conflict-free
    assert initial_domains(EASY) is not None
    assert initial_domains(HARD) is not None


# 4. propagate and helpers

def test_obtain_singleton_cells():
    d = [{1}, {1, 2, 3}, {4, 5}, {1}]
    assert obtain_singleton_cells(d) == [0, 3]
    assert obtain_singleton_cells([{1, 2}, {3, 4}]) == []


def test_obtain_smallest_domain_cell():
    d = [{1}, {1, 2, 3}, {4, 5}, {1}]
    assert obtain_smallest_domain_cell(d) == 2      # size 2 beats size 3; singletons skipped
    assert obtain_smallest_domain_cell([{1}, {2}]) == -1


def test_propagate_forces_last_digit():
    # row 0 has 1-8 filled in, so cell 8 must be 9
    g = [0] * 81
    for c in range(8):
        g[c] = c + 1
    d = initial_domains(g)
    assert propagate(d) is True
    assert d[8] == {9}


def test_propagate_detects_contradiction():
    # same row as above, but a 9 sits in cell 8's column, so cell 8 has no legal value
    g = [0] * 81
    for c in range(8):
        g[c] = c + 1
    g[17] = 9
    d = initial_domains(g)
    assert d is not None            # clues alone do not conflict
    assert propagate(d) is False


def test_propagate_solves_easy_puzzle_mostly():
    d = initial_domains(EASY)
    assert propagate(d) is True
    # propagation must never remove the true answer from any domain
    assert all(EASY_SOLUTION[i] in d[i] for i in range(81))
    # and it should make real progress
    assert len(obtain_singleton_cells(d)) > sum(1 for v in EASY if v != 0)


# 5. search / solve_grid

def test_solve_easy():
    sols = solve_grid(EASY)
    assert len(sols) == 1
    assert sols[0] == EASY_SOLUTION
    assert is_valid_solution(sols[0], EASY)


def test_solve_hard():
    sols = solve_grid(HARD)
    assert len(sols) == 1
    assert is_valid_solution(sols[0], HARD)


def test_solve_does_not_modify_input():
    before = list(EASY)
    solve_grid(EASY)
    assert EASY == before


def test_solve_empty_grid_has_many_solutions():
    sols = solve_grid(EMPTY)
    assert len(sols) == 2 
    assert all(is_valid_solution(s, EMPTY) for s in sols)
    assert sols[0] != sols[1]


def test_solve_limit_parameter():
    assert len(solve_grid(EMPTY, limit=1)) == 1
    assert len(solve_grid(EMPTY, limit=5)) == 5


def test_solve_conflicting_clues():
    g = [0] * 81
    g[0] = g[1] = 5
    assert solve_grid(g) == []


def test_solve_conflict_found_by_propagation():
    # clues don't repeat a digit in any unit, but the puzzle is still impossible
    g = [0] * 81
    for c in range(8):
        g[c] = c + 1
    g[17] = 9
    assert solve_grid(g) == []


def test_solve_multiple_solutions():
    # remove clues from the easy puzzle until the solution stops being unique
    g = list(EASY_SOLUTION)
    for i in range(0, 81, 2): # every cell blank
        g[i] = 0
    sols = solve_grid(g)
    assert len(sols) == 2
    assert all(is_valid_solution(s, g) for s in sols)


# 6. solve_sudoku (full I/O: probabilities in, (class, is_input) pairs out)

def check_solve_sudoku_output(result, puzzle):
    assert result is not None
    assert len(result) == 81
    classes = [c for c, _ in result]
    assert is_valid_solution(classes, puzzle)
    for i, (cls, is_input) in enumerate(result):
        assert isinstance(cls, int)
        assert isinstance(is_input, bool)
        assert is_input == (puzzle[i] != 0)   # True exactly on the original clues


def test_solve_sudoku_clean_probs_easy():
    result = solve_sudoku(fake_probs(EASY))
    check_solve_sudoku_output(result, EASY)
    assert [c for c, _ in result] == EASY_SOLUTION
    assert result[0] == (5, True)
    assert result[2] == (4, False)


def test_solve_sudoku_clean_probs_hard():
    check_solve_sudoku_output(solve_sudoku(fake_probs(HARD)), HARD)


def test_solve_sudoku_varying_probs():
    # the true class always has the highest probability
    for seed in range(5):
        probs = noisy_probs(EASY, seed=seed)
        assert all(abs(sum(row) - 1.0) < 1e-9 for row in probs)
        check_solve_sudoku_output(solve_sudoku(probs), EASY)
    check_solve_sudoku_output(solve_sudoku(noisy_probs(HARD, seed=42)), HARD)


def test_solve_sudoku_unsolvable_reading():
    g = [0] * 81
    g[0] = g[1] = 5
    # for milestone 1, no error correction yet, so an impossible reading gives None.
    assert solve_sudoku(fake_probs(g)) is None

# Runner (so `python test_csp_solver.py` works without pytest)
# NOTE: Claude helped me figure out how to do this bit so we can skip pytest

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
