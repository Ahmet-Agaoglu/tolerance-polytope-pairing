"""Fresh Python implementation; all geometric calculations use scaled coordinates."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import cubature
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.stats import qmc


@dataclass(frozen=True)
class CaseStudy:
    n_contacts: int = 5
    pitch_radius_mm: float = 200.0
    pin_mean_mm: float = 5.992
    pin_std_mm: float = 0.0013
    hole_mean_mm: float = 6.006
    hole_std_mm: float = 0.0020
    location_std_mm: float = 0.0033


def generate_pair(seed: int, settings: CaseStudy):
    """Generate exactly one pair; do not reject or resample infeasible parts."""
    rng = np.random.default_rng(seed)
    n = settings.n_contacts
    angles = np.arange(n) * 2 * np.pi / n
    nominal = settings.pitch_radius_mm * np.column_stack(
        (np.cos(angles), np.sin(angles))
    )
    pin_d = rng.normal(settings.pin_mean_mm, settings.pin_std_mm, n)
    hole_d = rng.normal(settings.hole_mean_mm, settings.hole_std_mm, n)
    pin_x = rng.normal(0, settings.location_std_mm, n)
    pin_y = rng.normal(0, settings.location_std_mm, n)
    hole_x = rng.normal(0, settings.location_std_mm, n)
    hole_y = rng.normal(0, settings.location_std_mm, n)
    pins = np.column_stack((nominal + np.column_stack((pin_x, pin_y)), pin_d))
    holes = np.column_stack((nominal + np.column_stack((hole_x, hole_y)), hole_d))
    return pins, holes, nominal


def pair_constraints(pins, holes, nominal, n_directions=9):
    """Return A q <= b for q=(tx_mm, ty_mm, theta_rad), with CCW rotation.

    The nominal lever arm follows the source paper's first-order model.
    Nine directions plus their opposites give 18 unique support directions,
    matching the old helper's discretization count, not its midpoint lever arm.
    The contact polygons are outer approximations of circular clearances.
    """
    angles = np.arange(n_directions) * 2 * np.pi / n_directions
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    rows, bounds = [], []
    for pin, hole, position in zip(pins, holes, nominal):
        delta = pin[:2] - hole[:2]
        clearance = (hole[2] - pin[2]) / 2
        for normal in normals:
            lever = position[0] * normal[1] - position[1] * normal[0]
            row = np.r_[normal, lever]
            rows.extend((row, -row))
            bounds.extend((clearance - normal @ delta, clearance + normal @ delta))
    return np.array(rows), np.array(bounds)


def functional_constraints(radius_mm, sides=128):
    """Inscribed regular polygon extruded along the rotation coordinate."""
    if radius_mm <= 0 or sides < 3:
        raise ValueError("Positive radius and at least three sides are required.")
    normals_angle = (np.arange(sides) + 0.5) * 2 * np.pi / sides
    A = np.column_stack(
        (np.cos(normals_angle), np.sin(normals_angle), np.zeros(sides))
    )
    return A, np.full(sides, radius_mm * np.cos(np.pi / sides))


@dataclass
class Polytope:
    A: np.ndarray
    b: np.ndarray
    scale: np.ndarray
    status: str
    center: np.ndarray | None = None
    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None
    volume_scaled: float = 0.0

    @property
    def physical_vertices(self):
        return self.vertices * self.scale

    @property
    def volume(self):
        return self.volume_scaled * float(np.prod(self.scale))

    def tetrahedra(self):
        for face in self.faces:
            yield np.vstack((self.center, self.vertices[face]))


def build_polytope(A, b, scale, interior_tolerance=1e-9):
    """Enumerate a bounded 3D polytope without relaxing its constraints.

    Empty and lower-dimensional sets are distinct. Solver errors are never
    converted into geometric infeasibility.
    """
    A, b, scale = np.asarray(A), np.asarray(b), np.asarray(scale)
    scaled = A * scale
    norms = np.linalg.norm(scaled, axis=1)
    if np.any(norms == 0):
        raise ValueError("Zero-normal inequalities are not supported.")
    An, bn = scaled / norms[:, None], b / norms
    feasible = linprog(
        np.zeros(3), A_ub=An, b_ub=bn,
        bounds=[(None, None)] * 3, method="highs",
    )
    if feasible.status == 2:
        return Polytope(A, b, scale, "empty")
    if not feasible.success:
        raise RuntimeError(feasible.message)
    center = linprog(
        [0, 0, 0, -1], A_ub=np.column_stack((An, np.ones(len(b)))),
        b_ub=bn, bounds=[(None, None)] * 3 + [(0, None)], method="highs",
    )
    if not center.success:
        raise RuntimeError(center.message)
    if center.x[3] <= interior_tolerance:
        return Polytope(A, b, scale, "lower_dimensional", center=center.x[:3])
    vertices = HalfspaceIntersection(np.column_stack((An, -bn)), center.x[:3]).intersections
    if np.max(An @ vertices.T - bn[:, None]) > 1e-7:
        raise RuntimeError("Vertex feasibility validation failed.")
    hull = ConvexHull(vertices)
    result = Polytope(
        A, b, scale, "full_dimensional", center.x[:3],
        vertices, hull.simplices, float(hull.volume),
    )
    tetra_volume = sum(
        abs(np.linalg.det((T[1:] - T[0]).T)) / 6 for T in result.tetrahedra()
    )
    if not np.isclose(tetra_volume, result.volume_scaled, rtol=1e-9, atol=1e-12):
        raise RuntimeError("Tetrahedra do not reproduce the hull volume.")
    return result


def gaussian_weight(mean, sigma):
    """Unnormalized Gaussian kernel; parameters are in mm, mm, radians."""
    mean, sigma = np.asarray(mean), np.asarray(sigma)
    if np.any(sigma <= 0):
        raise ValueError("All standard deviations must be positive.")

    def weight(points):
        return np.exp(-0.5 * np.sum(((points - mean) / sigma) ** 2, axis=-1))

    return weight


def integrate_mass(poly: Polytope, weight: Callable, rtol=1e-8, atol=1e-10):
    """Tetrahedral decomposition followed by adaptive cube cubature.

    For u,v,t in [0,1], barycentric coordinates are
    (1-u)(1-v)(1-t), u, (1-u)v, (1-u)(1-v)t.
    The Jacobian is abs(det(B))*(1-u)^2*(1-v).
    Error values are numerical estimates, not certified mathematical bounds.
    """
    if poly.status in ("empty", "lower_dimensional"):
        return {"mass": 0.0, "estimated_error": 0.0, "tetrahedra": 0, "subdivisions": 0}
    value, error, subdivisions = 0.0, 0.0, 0
    for tetra in poly.tetrahedra():
        B = (tetra[1:] - tetra[0]).T
        determinant = abs(np.linalg.det(B))

        def integrand(cube_points):
            u, v, t = cube_points.T
            bary = np.column_stack((u, (1-u)*v, (1-u)*(1-v)*t))
            physical = (tetra[0] + bary @ B.T) * poly.scale
            return weight(physical) * determinant * (1-u)**2 * (1-v)

        integral = cubature(
            integrand, np.zeros(3), np.ones(3), rule="genz-malik",
            rtol=rtol, atol=atol / len(poly.faces), max_subdivisions=10000,
        )
        if integral.status != "converged":
            raise RuntimeError("Cubature did not converge.")
        value += float(integral.estimate)
        error += float(integral.error)
        subdivisions += integral.subdivisions
    jacobian = float(np.prod(poly.scale))
    return {
        "mass": value * jacobian,
        "estimated_error": error * jacobian,
        "tetrahedra": len(poly.faces),
        "subdivisions": subdivisions,
    }


def uniform_sobol_samples(poly, power=15, seed=31415):
    """Independent check using scrambled Sobol points and volume-weighted tetrahedra."""
    tetra = np.array(list(poly.tetrahedra()))
    volumes = np.abs(np.linalg.det(np.transpose(tetra[:, 1:] - tetra[:, :1], (0, 2, 1)))) / 6
    cumulative = np.cumsum(volumes) / volumes.sum()
    cumulative[-1] = 1
    u = qmc.Sobol(4, scramble=True, seed=seed).random_base2(power)
    ids = np.searchsorted(cumulative, u[:, 0])
    cuts = np.sort(u[:, 1:], axis=1)
    bary = np.diff(np.column_stack((np.zeros(len(u)), cuts, np.ones(len(u)))), axis=1)
    return np.einsum("ni,nij->nj", bary, tetra[ids]) * poly.scale
