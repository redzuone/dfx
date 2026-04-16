"""Shared geodetic helpers for all examples."""

from pyproj import Geod

from dfx import TriangulationResult

_geod = Geod(ellps="WGS84")


def bearing(from_lat: float, from_lon: float, to_lat: float, to_lon: float) -> float:
    """True bearing in degrees (0=N, CW) from one WGS-84 point to another."""
    az, _, _ = _geod.inv(from_lon, from_lat, to_lon, to_lat)
    return float(az) % 360.0


def dist_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Geodetic distance in metres between two WGS-84 points."""
    _, _, d = _geod.inv(lon1, lat1, lon2, lat2)
    return float(d)


def print_result(result: TriangulationResult, truth_lat: float, truth_lon: float) -> None:
    err = dist_m(truth_lat, truth_lon, result.lat, result.lon)
    print(f"  Solved:    {result.lat:.6f} N  {result.lon:.6f} E")
    print(f"  Error:     {err:.1f} m")
    residuals = "  ".join(f"[{i+1}] {r:.4f} deg" for i, r in enumerate(result.residuals))
    print(f"  Residuals: {residuals}")
    if result.ellipse:
        sm, sn, az = result.ellipse
        print(f"  1-sigma ellipse: {sm:.0f} m x {sn:.0f} m  (az={az:.1f} deg from N)")
    if result.condition > 1e6:
        print(f"  WARNING: condition={result.condition:.2e} -- near-parallel bearings, poor geometry")
    print()
