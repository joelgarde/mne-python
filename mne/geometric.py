from functools import partial
import logging
from mne.bem import _fit_sphere

import numpy as np
from numpy.typing import ArrayLike

vnorm = partial(np.linalg.norm, axis=-1, ord=2)
vcross = partial(np.cross, axis=-1)



def carthesian(a: ArrayLike, ρ=0.1):
    """converts a to carthesian coordiantes."""
    a = np.asarray(a)
    if a.shape[-1] == 2:
        ρ = ρ
        θ, φ = a.T
    else:
        ρ, θ, φ = a.T
    θ = (np.pi /2) - θ
    cos_θ = np.cos(θ)
    cos_φ = np.cos(φ)
    sin_θ = np.sin(θ)
    sin_φ = np.sin(φ)
    x = ρ * sin_θ * cos_φ
    y = ρ * sin_θ * sin_φ
    z = ρ * cos_θ
    return np.c_[x, y, z].squeeze()


def polar(a: ArrayLike):
    """converts x to polar coordinates."""
    a = np.asarray(a)
    x, y, z = a.T
    ρ = vnorm(a)
    rxy = vnorm(a[..., :2])
    θ = (np.pi / 2) - np.arctan2(rxy, z)
    φ = np.arctan2(y, x)
    return np.c_[ρ, θ, φ].squeeze()


def find_tri(X: ArrayLike, rr: ArrayLike, tri: ArrayLike):
    """
    Triangles - polytope intersection using: 10.1145/1198555.1198746.
    nota: may cause issues when X is an exact point in rr.
    Parameters
    ----------
    X: (3,): directions of the rays intersecting the triangle mesh.
    rr: (Np, 3): points coordinates on the spherical mesh.
    tri: (Nt, 3): triangles indices.
    Returns
    -------
    m: intersected triangle index.
    t: (3,) coords of the intersection in in i's barycentric system.
    """

    # Solve the barycentric system.
    X = np.ravel(X)
    A = rr[tri]
    E1 = A[:, 1] - A[:, 0]
    E2 = A[:, 2] - A[:, 0]
    T = -A[:, 0]
    P = np.cross(X, E2)
    Q = np.cross(T, E1)
    det = np.einsum("ij,ij->i", E1, P)
    idet = np.reciprocal(det)
    u = np.einsum("ij,ij,i->i", T, P, idet)
    v = np.einsum("j,ij,i->i", X, Q, idet)
    cull = (v >= 0) & (u >= 0) & (u + v <= 1)
    m = np.flatnonzero(cull).item()
    v = v[m]
    u = u[m]
    t = np.asarray([u, v, 1 - u - v])
    return m, t


class SphereWhite(dict):
    """subclassing dict to be considered as a surface by the rest of mne.
    Points on the sphere are treated using their carthesian coordinates.
    """

    def __init__(self, hemishpere='lh', project=False) -> None:
        from pathlib import Path
        from mne.surface import _read_mri_surface
        from mne.datasets.sample import data_path
        from scipy.sparse import csr_array
        from sklearn.neighbors import KDTree

        self.project = project
        surf_path = Path(data_path()) / 'subjects' / 'sample' / 'surf'
        white = _read_mri_surface(surf_path / f'{hemishpere}.white')
        sphere = _read_mri_surface(surf_path / f'{hemishpere}.sphere')
        self |= white

        radius, origin = _fit_sphere(sphere['rr'])

        logging.info(f'sphere radius, origine: {radius}, {origin}')           
        u = sphere['rr'] - origin
        u /= radius
        self['sph_rr'] = u
        self.radius = 1.0
        self.center = 0.0

        self.tree = KDTree(self['sph_rr'])
        points = self['tris'].flatten()
        tri_idx = np.repeat(np.arange(self['tris'].shape[0]), 3)
        point2tri = csr_array((tri_idx, (points, tri_idx)), shape=(self['rr'].shape[0], self['tris'].shape[0]))
        self.point2tri = point2tri


    def transform(self, rd, on="rr"):
        """ from sphere point to cortical point.
        """
        rd = np.atleast_2d(rd)
        if self.project:
            rd = rd / np.linalg.norm(rd, axis=1).item()
        leaf = self.tree.query(rd, k=25, sort_results=False, return_distance=False)
        tri_leaf = np.unique(self.point2tri[leaf[0]].data)
        intersected_tri, barycentric_coordinates = find_tri(
            rd, self["sph_rr"], self["tris"][tri_leaf])
        idx = tri_leaf[intersected_tri]
        tri = self[on][self["tris"][idx]]
        rd = np.einsum("ij,i->j", tri, barycentric_coordinates)
        print(rd)
        return rd


def toMesh(rr, tri):
    import pyvista
    mesh = pyvista.PolyData(rr, np.c_[np.full(len(tri), 3), tri])
    return mesh


if __name__ == "__main__":
    """this si a demo of the sphere <-> cortical mapping."""
    from pathlib import Path

    import pyvista

    import mne

    np.set_printoptions(precision=2, floatmode="fixed")
    surfaces_dir: Path = (
        mne.datasets.sample.data_path() / "subjects" / "sample" / "surf"
    )
    subject_path: Path = mne.datasets.sample.data_path() / "subjects" / "sample"
    logging.basicConfig(level=logging.INFO)
    sph = SphereWhite()
    random = True
    if random:
        # direct random sampling on the sphere
        from numpy.random import default_rng

        rng = default_rng()
        grid = rng.standard_normal((200, 3))
        inv_norm = sph.radius / np.linalg.norm(grid, axis=1)
        grid = np.einsum("ij,i->ij", grid, inv_norm)
    else:
        # grid on angles points /!\ not uniform on the sphere.
        grid = carthesian(np.mgrid[0.1 : np.pi : 10j, -np.pi : np.pi : 10j].reshape((2, -1)).T)

    assert np.allclose(grid, carthesian(polar(grid)))

    rd = np.vectorize(partial(sph.transform, on="sph_rr"), signature="(m)->(n)")(grid)
    
    sphere_mesh = toMesh(sph["sph_rr"], sph["tris"])
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere_mesh, opacity=0.85, show_edges=True)
    plotter.add_points(
        grid, color="blue", render_points_as_spheres=True, point_size=10.0
    )
    plotter.add_points(rd, color="red", render_points_as_spheres=True, point_size=10.0)
    plotter.add_title(
        "Points sampled on (θ, φ) space, \ntheir carthesian transform (blue) \nand mesh interpolation (red)"
    )
    plotter.show()

    rd = np.vectorize(partial(sph.transform, on="rr"), signature="(m)->(n)")(grid)
    plotter = pyvista.Plotter()
    plotter.add_mesh(toMesh(sph["rr"], sph["tris"]), opacity=0.85, show_edges=True)
    plotter.add_points(rd, color="red", render_points_as_spheres=True, point_size=10.0)
    plotter.add_title("Points sampled on (θ, φ) space and cortical interpolation.")
    plotter.show()