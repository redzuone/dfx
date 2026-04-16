"""
Example 2 -- Realistic: 3 stations, bearing noise, per-station sigma.

Real sensors have angular measurement error. Each observation includes a 1-sigma
value in degrees. The solver weights each bearing accordingly and returns a
best-estimate fix plus an uncertainty ellipse (semi-major/minor axes in metres,
azimuth from North).

Run:
    python -m examples.ex2_realistic
"""

from dfx import Triangulator
from examples._helpers import bearing, print_result


def main() -> None:
    print("=" * 60)
    print("Example 2 -- Realistic: 3 stations, noise + uncertainty ellipse")
    print("=" * 60)

    TARGET = (52.0000, 0.0000)

    # (station_lat, station_lon, noise_offset_deg, sigma_deg)
    # sigma_deg is the 1-sigma (68%) angular measurement uncertainty
    stations = [
        (51.80, -0.30, +0.30, 0.5),   # ~28 km SW, moderate precision (e.g. SIGINT DF)
        (51.85,  0.35, -0.50, 0.3),   # ~22 km SE, high precision    (e.g. radar DF)
        (52.20,  0.25, +0.20, 1.0),   # ~23 km NE, low precision     (e.g. visual bearing)
    ]

    observations = [
        (slat, slon, (bearing(slat, slon, *TARGET) + noise) % 360.0, sigma)
        for slat, slon, noise, sigma in stations
    ]

    print(f"  Truth:     {TARGET[0]:.6f} N  {TARGET[1]:.6f} E")
    print("  Stations:")
    for (slat, slon, noise, sigma), (_, _, brg, _) in zip(stations, observations):
        print(f"    ({slat:.2f}, {slon:.2f})  brg={brg:.2f} deg  sigma={sigma} deg")
    print()

    result = Triangulator().locate(observations)
    print_result(result, *TARGET)


if __name__ == "__main__":
    main()
