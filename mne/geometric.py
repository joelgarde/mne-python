from functools import partial

import numpy as np
from numpy.typing import ArrayLike

vnorm = partial(np.linalg.norm, axis=-1, ord=2)
vcross = partial(np.cross, axis=-1)

def make_find_tri(rr: ArrayLike, tri: ArrayLike):
    tree = tri_KDTree(rr, tri)
    def find_tri_tree(x):
        """fast point to intersecting triangle using a KDTree."""
        idx = tree(x)
        tri_idx, tri_coo = find_tri(x, rr, tri[idx])
        return idx[tri_idx], tri_coo
    return find_tri_tree

def carthesian(a: ArrayLike, ρ=0.1):
    """converts a to carthesian coordiantes."""
    a = np.asarray(a)
    if a.shape[-1] == 2:
        ρ = ρ
        θ, φ = a.T
    else:
        ρ, θ, φ = a.T
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
    θ = np.arctan2(rxy, z)
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
    i: intersected triangle index.
    t: (3,) coords of the intersection in in i's barycentric system.
    """

    # Solve the barycentric system.
    e = 1e-5
    X = np.asarray(X)
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



def tri_KDTree(rr: ArrayLike, tri: ArrayLike):
    """build a KTD tree to find tri close to a point x."""
    A = rr[tri]
    E1 = np.linalg.norm(A[:, 0] - A[:, 1], axis=-1)
    E2 = np.linalg.norm(A[:, 0] - A[:, 2], axis=-1)
    E3 = np.linalg.norm(A[:, 2] - A[:, 1], axis=-1)
    E = np.c_[E1, E2, E3]
    max_tri_lenght = np.max(E)
    point2tri = np.frompyfunc(list, 0, 1)(np.empty(rr.shape[0], dtype=object))
    for i, (a, b, c) in enumerate(tri):
        point2tri[a].append(i)
        point2tri[b].append(i)
        point2tri[c].append(i)

    from scipy.spatial import KDTree
    tree = KDTree(rr, copy_data=True)
    def find_close(x):
        """

        Parameters
        ----------
        x (3,) point to query

        Returns
        -------
        tri: (N,) close triangles.
        """
        #pids = tree.query_ball_point(np.atleast_2d(x), r=max_tri_lenght)
        dd, pids = tree.query(x, k=10)
        return np.unique([u for i in pids for u in point2tri[i]])

    return find_close





class SphereMappedSources(dict):

    @classmethod
    def from_path(cls, subject_path,  hemisphere="lh",  **setupkwargs):
        from pathlib import Path
        from mne import setup_source_space

        subject_path = Path(subject_path)
        subject = subject_path.name
        subject_dir = subject_path.parent
        setupkwargs = dict(subjects_dir=subject_dir, add_dist=False, spacing="all") | setupkwargs
        setup = partial(setup_source_space, subject, **setupkwargs)

        white = setup(surface="white")
        sphere = setup(surface="sphere")

        idx = 0 if (hemisphere == "lh") else 1
        white = white[idx]
        sphere = sphere[idx]

        sph_rr = sphere['rr']
        mesh_rr = white['rr']
        tri = sphere['tris']

        cull_unused = False
        if cull_unused:
            vertno = sphere['vertno']
            tris = sphere['use_tris'] or sphere['tris']
            inv_vertno = np.full(sph_rr.shape[0], -1, dtype=np.intc)
            inv_vertno[vertno] = np.arange(vertno.shape[0])
            mesh_rr = mesh_rr[vertno]
            sph_rr = sph_rr[vertno]
            tri = inv_vertno[tris]

        tree_fn = tri_KDTree(sph_rr, tri)
        out = cls(id=sphere["id"], coord_frame=white["coord_frame"], rr=mesh_rr, tree_fn=tree_fn, sph_rr=sph_rr, tri=tri)
        return out

    def transform(self, rd, on='rr'):
        """ late binding provided by `from_path`
        (φ,θ) → (x,y,z)
        """
        coords = carthesian(rd)
        idx_leafs = self["tree_fn"](coords)
        correct_leaf, bary_coo = find_tri(coords, self["sph_rr"], self["tri"][idx_leafs])
        idx = idx_leafs[correct_leaf]
        rd = np.einsum("ij,i->j", self[on][self["tri"][idx]], bary_coo)
        return rd


def toMesh(rr, tri):
    import pyvista
    mesh = pyvista.PolyData(rr, np.c_[np.full(len(tri), 3), tri])
    return mesh

if __name__ == "__main__":
    """ this si a demo of the sphere <-> cortical mapping.
    """
    from pathlib import Path
    import pyvista
    import mne
    np.set_printoptions(precision=2, floatmode="fixed")
    surfaces_dir: Path = mne.datasets.sample.data_path() / "subjects" / "sample" / "surf"
    subject_path: Path = mne.datasets.sample.data_path() / "subjects" / "sample"

    sph = SphereMappedSources.from_path(subject_path=subject_path,)

    random = True
    if random:
        #direct random sampling on the sphere
        from numpy.random import default_rng
        rng = default_rng()
        grid = rng.standard_normal((200, 3))
        inv_norm = 0.1 / np.linalg.norm(grid, axis=1)
        grid = polar(np.einsum("ij,i->ij", grid, inv_norm))
    else:
        #grid on angles points /!\ not uniform on the sphere.
        grid = np.mgrid[0.1:np.pi:10j, -np.pi:np.pi:10j].reshape((2, -1)).T

    assert np.allclose(grid, polar(carthesian(grid)))
    rd = np.vectorize(partial(sph.transform, on="sph_rr"), signature="(m)->(n)")(grid)

    sphere_mesh = toMesh(sph['sph_rr'], sph['tri'])
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere_mesh, opacity=0.85, show_edges=True)
    plotter.add_points(carthesian(grid), color="blue", render_points_as_spheres=True, point_size=10.0)
    plotter.add_points(rd, color="red", render_points_as_spheres=True, point_size=10.0)
    plotter.add_title("Points sampled on (θ, φ) space, \ntheir carthesian transform (blue) \nand mesh interpolation (red)")
    plotter.show()

    rd = np.vectorize(partial(sph.transform, on="rr"), signature="(m)->(n)")(grid)
    plotter = pyvista.Plotter()
    plotter.add_mesh(toMesh(sph['rr'], sph['tri']), opacity=0.85, show_edges=True)
    plotter.add_points(rd, color="red", render_points_as_spheres=True, point_size=10.0)
    plotter.add_title(
        "Points sampled on (θ, φ) space and cortical interpolation.")
    plotter.show()






