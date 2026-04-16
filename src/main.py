"""
Entry point — runs all bearing-triangulation examples in sequence.

Individual examples live in src/examples/:
    ex1_minimal.py        2 stations, exact bearings, no sigma
    ex2_realistic.py      3 stations, noise, uncertainty ellipse
    ex3_overdetermined.py 6 heterogeneous sensors, poor-geometry warning
    ex4_long_range.py     4 stations 100-300 km away

Run:
    python src/main.py
"""

from examples.ex1_minimal import main as ex1
from examples.ex2_realistic import main as ex2
from examples.ex3_overdetermined import main as ex3
from examples.ex4_long_range import main as ex4

if __name__ == "__main__":
    ex1()
    ex2()
    ex3()
    ex4()
