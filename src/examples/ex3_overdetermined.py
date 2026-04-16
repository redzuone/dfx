"""
Example 3 -- Overdetermined: 6 heterogeneous sensors, mixed accuracy and geometry.

Six sensors of different types (SIGINT, radar DF, ELINT) surround the target.
Two sensors (Radar-N and Radar-NE) have nearly parallel bearings — the solver
warns via the condition number. Despite the poor pair, the four remaining sensors
produce a well-constrained fix. The uncertainty ellipse is noticeably smaller
than Example 2 thanks to the additional observations.

Run:
    python -m examples.ex3_overdetermined
"""

from dfx import Triangulator
from examples._helpers import bearing, dist_m, print_result


def main() -> None:
    print("=" * 60)
    print("Example 3 -- Overdetermined: 6 sensors, mixed accuracy + poor-geometry pair")
    print("=" * 60)

    TARGET = (48.8566, 2.3522)  # Paris

    # (station_lat, station_lon, noise_deg, sigma_deg, label)
    stations = [
        (48.60,  2.00, +0.4, 0.5, "SIGINT-A "),   # ~33 km SW
        (48.60,  2.70, -0.3, 0.5, "SIGINT-B "),   # ~30 km SE
        (49.10,  2.20, +0.6, 1.0, "Radar-N  "),   # ~27 km N
        (49.10,  2.45, -0.5, 1.0, "Radar-NE "),   # ~28 km NNE -- near-parallel to Radar-N
        (48.70,  1.50, +0.2, 0.3, "ELINT-W  "),   # ~60 km W, high precision
        (48.50,  3.10, -0.4, 0.8, "ELINT-SE "),   # ~65 km SE
    ]

    observations = [
        (slat, slon, (bearing(slat, slon, *TARGET) + noise) % 360.0, sigma)
        for slat, slon, noise, sigma, _ in stations
    ]

    print(f"  Truth:     {TARGET[0]:.6f} N  {TARGET[1]:.6f} E")
    print("  Sensors:")
    for (slat, slon, _, sigma, label), (_, _, brg, _) in zip(stations, observations):
        rng_km = dist_m(slat, slon, *TARGET) / 1000
        print(f"    {label}  ({slat:.2f},{slon:.2f})  {rng_km:.0f} km  brg={brg:.1f} deg  sigma={sigma} deg")
    print()

    result = Triangulator().locate(observations)
    print_result(result, *TARGET)


if __name__ == "__main__":
    main()
