"""
Example 4 -- Long range: stations 100-300 km away.

At ranges beyond ~50 km a flat-earth approximation accumulates hundreds of metres
of systematic error. This example uses the WGS-84 / ECEF algorithm which remains
accurate at any range. Simulates an HF direction-finding network or satellite
intercept stations surrounding a distant target.

Run:
    python -m examples.ex4_long_range
"""

from dfx import Triangulator
from examples._helpers import bearing, dist_m, print_result


def main() -> None:
    print("=" * 60)
    print("Example 4 -- Long range: stations 100-300 km away")
    print("=" * 60)

    TARGET = (35.6892, 51.3890)  # Tehran

    # (station_lat, station_lon, noise_deg, sigma_deg, label)
    stations = [
        (34.00, 50.00, +0.2, 0.5, "Station-SW"),   # ~230 km
        (37.20, 49.50, -0.3, 0.5, "Station-NW"),   # ~210 km
        (36.50, 53.50, +0.4, 0.8, "Station-NE"),   # ~175 km
        (34.50, 53.00, -0.2, 0.6, "Station-SE"),   # ~215 km
    ]

    observations = [
        (slat, slon, (bearing(slat, slon, *TARGET) + noise) % 360.0, sigma)
        for slat, slon, noise, sigma, _ in stations
    ]

    print(f"  Truth:     {TARGET[0]:.6f} N  {TARGET[1]:.6f} E")
    print("  Stations:")
    for (slat, slon, _, sigma, label), (_, _, brg, _) in zip(stations, observations):
        rng_km = dist_m(slat, slon, *TARGET) / 1000
        print(f"    {label}  ({slat:.2f},{slon:.2f})  {rng_km:.0f} km  brg={brg:.1f} deg  sigma={sigma} deg")
    print()

    result = Triangulator().locate(observations)
    print_result(result, *TARGET)


if __name__ == "__main__":
    main()
