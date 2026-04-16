"""
Example 1 -- Minimum viable: 2 stations, exact bearings, no sigma.

The simplest possible call. Two stations and computed exact bearings — no noise,
no sigma. This is the bare minimum to get a fix. No ellipse is returned because
no uncertainty information was provided.

Run:
    python -m examples.ex1_minimal
"""

from dfx import Triangulator
from examples._helpers import bearing, print_result


def main() -> None:
    print("=" * 60)
    print("Example 1 -- Minimum viable: 2 stations, exact bearings")
    print("=" * 60)

    TARGET = (52.0000, 0.0000)  # somewhere in East Anglia

    observations = [
        (51.80, -0.30, bearing(51.80, -0.30, *TARGET)),  # ~28 km SW
        (51.85,  0.35, bearing(51.85,  0.35, *TARGET)),  # ~22 km SE
    ]

    print(f"  Truth:     {TARGET[0]:.6f} N  {TARGET[1]:.6f} E")
    result = Triangulator().locate(observations)
    print_result(result, *TARGET)
    print("  (no ellipse: no sigma values supplied)")
    print()


if __name__ == "__main__":
    main()
