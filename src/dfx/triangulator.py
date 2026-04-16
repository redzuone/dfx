"""
Geodetic bearing triangulation / direction-finding intersection.

Each observation is a (sensor_lat, sensor_lon, true_bearing_deg) tuple, optionally
followed by a 1-sigma angular uncertainty in degrees.  The solver finds the
WGS-84 (lat, lon) point that best satisfies all bearing lines using a weighted
eigenproblem on Earth-centred Earth-fixed (ECEF) great-circle planes — making it
accurate from metres to hundreds of kilometres.

Angle convention: **true bearing**, 0 ° = geographic North, clockwise positive.
                  This is the NATO / STANAG standard used by military sensors.

Usage::

    from dfx import Triangulator

    result = Triangulator().locate([
        (51.5, -0.1, 045.0),           # equal weight (default σ = 1 °)
        (51.6, -0.3, 120.0, 0.5),      # 0.5 ° 1-sigma
        (51.4, -0.5, 080.0, 1.0),
    ])

    print(result.lat, result.lon)
    print(result.ellipse)    # (semi_major_m, semi_minor_m, azimuth_deg) or None
    print(result.residuals)  # per-observation angular residuals in degrees
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, cast

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# (lat, lon, bearing) or (lat, lon, bearing, sigma_deg)
# Using Sequence[float] allows flexible tuple lengths without type: ignore suppressions.
Observation = Sequence[float]

# Semi-major axis (m), semi-minor axis (m), azimuth of semi-major from North (°)
UncertaintyEllipse = Tuple[float, float, float]

# WGS-84 parameters
_A = 6_378_137.0          # semi-major axis (m)
_F = 1 / 298.257_223_563  # flattening
_B = _A * (1 - _F)        # semi-minor axis (m)
_E2 = 2 * _F - _F**2      # first eccentricity squared

_DEFAULT_SIGMA_DEG = 1.0   # assumed 1-sigma when none supplied


# ---------------------------------------------------------------------------
# Low-level ECEF / geodetic helpers
# ---------------------------------------------------------------------------


def _to_ecef(lat_deg: float, lon_deg: float) -> npt.NDArray[np.float64]:
    """
    Convert a surface point (WGS-84) to an ECEF unit vector.

    The magnitude is normalised to 1 so the vector sits on the unit sphere;
    this is sufficient for great-circle plane arithmetic.
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    cos_lat = math.cos(lat)
    x = cos_lat * math.cos(lon)
    y = cos_lat * math.sin(lon)
    z = math.sin(lat)
    return np.array([x, y, z], dtype=np.float64)


def _ecef_unit_to_geodetic(v: npt.NDArray[np.float64]) -> Tuple[float, float]:
    """
    Convert an ECEF *unit* vector back to (lat_deg, lon_deg) on the sphere.

    Because we work with normalised vectors the inversion is trivial.
    """
    v = v / np.linalg.norm(v)
    lat_deg = math.degrees(math.asin(float(np.clip(v[2], -1.0, 1.0))))
    lon_deg = math.degrees(math.atan2(float(v[1]), float(v[0])))
    return lat_deg, lon_deg


def _bearing_to_ecef_tangent(
    lat_deg: float, lon_deg: float, bearing_deg: float
) -> npt.NDArray[np.float64]:
    """
    Return the ECEF unit tangent vector for a true bearing at a station.

    The bearing defines a direction in the local East-North-Up (ENU) frame at
    the station.  We project that into ECEF and normalise.

    ENU axes in ECEF:
        E = [-sin(lon),  cos(lon),       0    ]
        N = [-sin(lat)*cos(lon), -sin(lat)*sin(lon),  cos(lat)]
        U = [ cos(lat)*cos(lon),  cos(lat)*sin(lon),  sin(lat)]

    A true bearing β gives the horizontal ENU direction:
        d_enu = [sin(β), cos(β), 0]   (E, N, U components)
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    beta = math.radians(bearing_deg)

    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    sin_b, cos_b = math.sin(beta), math.cos(beta)

    # East component weight = sin_b, North component weight = cos_b
    e = np.array([-sin_lon, cos_lon, 0.0], dtype=np.float64)
    n = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], dtype=np.float64)

    tangent = sin_b * e + cos_b * n
    tangent /= np.linalg.norm(tangent)
    return tangent


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class TriangulationResult:
    """
    Result of a bearing triangulation.

    Attributes
    ----------
    lat : float
        Best-estimate WGS-84 latitude in decimal degrees.
    lon : float
        Best-estimate WGS-84 longitude in decimal degrees.
    ellipse : UncertaintyEllipse | None
        One-sigma uncertainty ellipse as
        ``(semi_major_m, semi_minor_m, azimuth_deg)``.
        *azimuth_deg* is the clockwise angle from North to the semi-major axis.
        ``None`` when no angular uncertainties (σ) were supplied.
    residuals : list[float]
        Per-observation angular residual in degrees — the angle between the
        solved point and each sensor's bearing line.
    condition : float
        Ratio of the two smallest eigenvalues of the normal matrix.  Values
        close to 1 indicate well-conditioned geometry; very large values warn
        of near-parallel bearing lines (poor geometry / PDOP).
    """

    lat: float
    lon: float
    ellipse: Optional[UncertaintyEllipse]
    residuals: list[float] = field(default_factory=list)
    condition: float = 0.0


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------


class Triangulator:
    """
    Geodetic bearing triangulation solver.

    Accepts 2 or more (station_lat, station_lon, true_bearing_deg [, sigma_deg])
    observations and returns the best-estimate target location with optional
    uncertainty ellipse.

    The algorithm represents each bearing as a great-circle plane normal in
    ECEF and solves the weighted least-squares problem by finding the smallest
    eigenvector of the 3×3 normal matrix  M = Σ wᵢ nᵢ nᵢᵀ.  This is
    non-iterative, globally optimal (given the linearisation), and accurate
    at any range from metres to hundreds of kilometres.
    """

    def locate(
        self,
        observations: Sequence[Observation],
        *,
        default_sigma_deg: float = _DEFAULT_SIGMA_DEG,
    ) -> TriangulationResult:
        """
        Compute the best-estimate target location from bearing observations.

        Parameters
        ----------
        observations :
            Sequence of ``(lat, lon, bearing)`` or ``(lat, lon, bearing, sigma)``
            tuples.  *lat* and *lon* are WGS-84 decimal degrees; *bearing* is a
            true bearing in degrees (0 ° = North, clockwise); *sigma* is the
            1-sigma angular measurement uncertainty in degrees.
        default_sigma_deg :
            Fallback 1-sigma value used for observations that do not include a
            sigma.  Does not affect the fix when all weights are equal (only
            relative weights matter), but *does* affect the ellipse scale.

        Returns
        -------
        TriangulationResult

        Raises
        ------
        ValueError
            If fewer than 2 observations are supplied.
        """
        if len(observations) < 2:
            raise ValueError("At least 2 observations are required for triangulation.")

        normals: list[npt.NDArray[np.float64]] = []
        weights: list[float] = []
        has_explicit_sigma = False

        for obs in observations:
            lat, lon, bearing = float(obs[0]), float(obs[1]), float(obs[2])
            if len(obs) >= 4:
                sigma = float(obs[3])
                has_explicit_sigma = True
            else:
                sigma = default_sigma_deg

            sigma_rad = math.radians(max(sigma, 1e-6))  # guard against zero
            weight = 1.0 / sigma_rad**2

            station = _to_ecef(lat, lon)
            tangent = _bearing_to_ecef_tangent(lat, lon, bearing)
            normal: npt.NDArray[np.float64] = cast(
                npt.NDArray[np.float64], np.cross(station, tangent)
            )
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-12:
                raise ValueError(
                    f"Degenerate observation: station and bearing tangent are "
                    f"collinear at ({lat}, {lon})."
                )
            normal = (normal / norm_len).astype(np.float64)

            normals.append(normal)
            weights.append(weight)

        # Build 3×3 weighted normal matrix  M = Σ wᵢ nᵢ nᵢᵀ
        M = np.zeros((3, 3), dtype=np.float64)
        for w, n in zip(weights, normals):
            M += w * np.outer(n, n)

        # Smallest eigenvector → best-estimate ECEF direction
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        # eigh returns ascending order; smallest eigenvalue is index 0
        solution_ecef = eigenvectors[:, 0]
        # Ensure the solution is on the correct hemisphere (positive Z ≈ upward)
        if solution_ecef[2] < 0:
            solution_ecef = -solution_ecef

        lat_sol, lon_sol = _ecef_unit_to_geodetic(solution_ecef)

        # Condition number: ratio of 2nd-smallest to smallest eigenvalue
        condition = float(eigenvalues[1] / max(eigenvalues[0], 1e-30))

        # Per-observation angular residuals in degrees
        residuals = []
        for n in normals:
            # |n · p| = sin of the angle between the great-circle plane and p
            sin_angle = float(np.clip(abs(float(n @ solution_ecef)), -1.0, 1.0))
            residuals.append(math.degrees(math.asin(sin_angle)))

        # Uncertainty ellipse --------------------------------------------------
        ellipse: Optional[UncertaintyEllipse] = None

        if has_explicit_sigma:
            # Build the Jacobian of the *bearing* (rad) at each station w.r.t.
            # a local East-North displacement (m) of the target point.
            #
            # For station i at range r_i observing with true bearing β_i:
            # A target displacement (dE, dN) changes the bearing by:
            #   dβ ≈ (dE·cos β_i − dN·sin β_i) / r_i   [rad per meter]
            #
            # so J[i] = [cos β_i,  −sin β_i] / r_i
            #
            # C_inv = Σ J_i^T J_i / σ_i²    (2×2, units m⁻²)
            # C     = C_inv⁻¹               (2×2, units m²)

            from pyproj import Geod  # already a project dependency

            _geod = Geod(ellps="WGS84")

            J = np.zeros((len(observations), 2), dtype=np.float64)
            obs_sigmas: list[float] = []

            for i, obs in enumerate(observations):
                slat, slon = float(obs[0]), float(obs[1])
                sigma = float(obs[3]) if len(obs) >= 4 else default_sigma_deg
                obs_sigmas.append(sigma)

                # Geodetic range and forward azimuth from station to solved point
                az_fwd, _az_bk, dist_m = _geod.inv(slon, slat, lon_sol, lat_sol)
                if dist_m < 1.0:
                    dist_m = 1.0  # avoid division by zero for co-located obs
                beta = math.radians(az_fwd % 360.0)

                J[i, 0] = math.cos(beta) / dist_m   # ∂β/∂E
                J[i, 1] = -math.sin(beta) / dist_m  # ∂β/∂N

            # Weight matrix: 1 / σ_i² in rad⁻²
            sigma_rads = np.array([math.radians(s) for s in obs_sigmas], dtype=np.float64)
            W_diag = 1.0 / sigma_rads**2

            C_inv = J.T @ (W_diag[:, None] * J)

            det = float(np.linalg.det(C_inv))
            if det > 1e-30:
                C = np.linalg.inv(C_inv)  # 2×2 ENU covariance in m²

                # Eigen-decompose covariance for ellipse parameters
                evals, evecs = np.linalg.eigh(C)
                # eigh returns ascending order; largest eigenvalue → semi-major
                semi_major = float(math.sqrt(max(evals[1], 0.0)))
                semi_minor = float(math.sqrt(max(evals[0], 0.0)))

                # Azimuth of the semi-major axis from North (clockwise)
                major_enu = evecs[:, 1]  # (East, North) components
                azimuth = math.degrees(math.atan2(float(major_enu[0]), float(major_enu[1])))
                azimuth = azimuth % 360.0

                ellipse = (semi_major, semi_minor, azimuth)

        return TriangulationResult(
            lat=lat_sol,
            lon=lon_sol,
            ellipse=ellipse,
            residuals=residuals,
            condition=condition,
        )
