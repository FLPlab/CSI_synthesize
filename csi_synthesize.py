"""Rebuild the CSI shards of chosen trajectories from a site's CSV file.

    python synthesize_csi.py --site <site> --traj 1-25,251-260 --out <site>_shards <site>_64x8x8_raytracing_time_series.csv
    python synthesize_csi.py --site <site> --per-profile 25 --out <site>_shards <site>_64x8x8_raytracing_time_series.csv
    python synthesize_csi.py --site <site> --all --out <site>_shards <site>_64x8x8_raytracing_time_series.csv

    Site options: (stecath, decarie, ericsson)

Writes to --out (default <site>_shards/):
    traj_%04d.mat  H (N_STEPS x N_SC x N_ANT complex single), valid, los, pos,
                   vel, g (profile), with MATLAB sizes
    meta.mat       settings and per-trajectory diagnostics, as synthesize_csi.m;
                   updated in place by later runs into the same folder

"""

import argparse
import datetime
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio


SITES = {                                   # BS latitude, longitude
    "stecath": (45.49995, -73.57322),
    "decarie": (45.48516, -73.63159),
    "ericsson": (45.49181, -73.72727),
}
FC_CENTER = 2e9                        
SCS = 15e3                             
N_SC = 64
ARRAY_SIZE = (8, 8)                         # BS URA, lambda/2 spacing
N_ANT = ARRAY_SIZE[0] * ARRAY_SIZE[1]
BS_HEIGHT = 25.0                            # m
RX_HEIGHT = 1.5                             # m
TX_POWER = 19.952623149688797               # W (43 dBm)
MAX_REFL = 2                                # ray tracing: reflections,
MAX_DIFF = 0                                # diffractions
ANCHOR_M = 0.5                              # anchor lattice spacing (m)
AOA_SIGN = -1                               # sign of the phase advance
DT = 1e-3                                   # s per step
N_STEPS = 2500
PROFILE_NAMES = ["pedestrian", "cyclist", "car", "fast"]
V_MEANS = [1.4, 5.0, 13.9, 25.0]            # m/s, per profile
N_PER_PROFILE = 250
N_TRAJ = N_PER_PROFILE * len(PROFILE_NAMES)
LIGHT_SPEED = 299792458.0
FC = FC_CENTER + (np.arange(N_SC) - (N_SC - 1) / 2) * SCS

CPLX_SINGLE = np.dtype([("real", "<f4"), ("imag", "<f4")])   # MATLAB's complex single
CORNERS = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
KBLK = 512                                                   # positions per block


def profile_of(traj):
    """1-based profile of trajectory numbers (from 1)."""
    return (np.asarray(traj) - 1) // N_PER_PROFILE + 1


def anchor_key(ij):
    # One exact int64 per lattice point (|i|, |j| < 2^20)
    ij = np.asarray(ij, dtype=np.int64)
    return (ij[..., 0] + 2**20) * 2**21 + (ij[..., 1] + 2**20)


def run_starts(x):
    """Indices where a new run of equal values starts in x."""
    return np.flatnonzero(np.concatenate([[True], x[1:] != x[:-1]]))


class PathDatabase:
    """Anchors and their ray paths, from the path rows of the CSV file."""

    def __init__(self, P):
        keys = anchor_key(P[["i", "j"]].to_numpy())
        starts = run_starts(keys)
        assert len(starts) == len(np.unique(keys)), "the rows of an anchor are not contiguous"
        has_path = P["tau_s"].notna().to_numpy()
        self.npath = np.add.reduceat(has_path.astype(np.int64), starts)
        self.ptr = np.concatenate([[0], np.cumsum(self.npath)])
        self.los = P["los"].to_numpy()[starts] == 1

        anchor_keys = keys[starts]
        self.order = np.argsort(anchor_keys)
        self.sorted_keys = anchor_keys[self.order]

        Q = P[has_path]
        self.tau = Q["tau_s"].to_numpy(np.float64)
        self.uxy = Q[["ux", "uy"]].to_numpy(np.float64)                  # Np x 2
        self.alpha = Q["alpha_re"].to_numpy(np.float64) + 1j * Q["alpha_im"].to_numpy(np.float64)
        self.psi = Q[["psi_1", "psi_2"]].to_numpy(np.float64)            # Np x 2

        n1, n2 = ARRAY_SIZE
        self.m1 = np.tile(np.arange(n1), n2)          # antenna m1 + n1 * m2 (MATLAB order)
        self.m2 = np.repeat(np.arange(n2), n1)

    def lookup(self, ij):
        """Rows of the anchors at lattice indices ij (..., 2); -1 where absent."""
        q = anchor_key(ij)
        loc = np.clip(np.searchsorted(self.sorted_keys, q), 0, len(self.sorted_keys) - 1)
        return np.where(self.sorted_keys[loc] == q, self.order[loc], -1)

    def gains(self, k):
        """Per-antenna complex gains of paths k: len(k) x N_ANT."""
        ph = np.outer(self.psi[k, 0], self.m1) + np.outer(self.psi[k, 1], self.m2)
        return self.alpha[k, None] * np.exp(1j * ph)


def read_site(csv_file):
    """Path database and trajectories {traj: (pos, vel)} of a site's CSV file."""
    D = pd.read_csv(csv_file)
    is_traj = D["traj"].notna()
    db = PathDatabase(D[~is_traj])

    T = D[is_traj]
    start = T[T["x"].notna()]
    steps = T[T["vx"].notna()]
    x0 = dict(zip(start["traj"].astype(int), start[["x", "y"]].to_numpy()))
    traj = steps["traj"].to_numpy().astype(int)
    vel = steps[["vx", "vy"]].to_numpy(np.float64)
    trajectories = {}
    for s0, s1 in zip(run_starts(traj), np.append(run_starts(traj)[1:], len(traj))):
        k = traj[s0]
        assert s1 - s0 == N_STEPS and k not in trajectories, f"trajectory {k}: bad step rows"
        v = vel[s0:s1]
        # pos(s+1) = pos(s) + vel(s) * DT, summed in step order as the generator did
        pos = np.cumsum(np.vstack([x0[k], v[:-1] * DT]), axis=0)
        trajectories[k] = (pos, v)
    return db, trajectories

def anchor_csi(db, a, delta, sgn):
    """CSI of anchor a's paths moved to offsets delta (K x 2, m): K x N_SC x N_ANT."""
    k = np.arange(db.ptr[a], db.ptr[a + 1])
    g = db.gains(k)                                          # Np x N_ANT
    tau = db.tau[k]
    dl = -sgn * (delta @ db.uxy[k].T)                        # K x Np, path-length change
    df = FC - FC_CENTER                                      # N_SC

    H = np.empty((len(delta), N_SC, N_ANT), np.complex128)
    for lo in range(0, len(delta), KBLK):
        d = dl[lo:lo + KBLK]                                 # kb x Np
        # Carrier phase at the centre frequency, folded into the gains; the
        # subcarrier offset sees the delay at the new position, tau + dl/c
        ph = np.exp(-2j * np.pi * FC_CENTER * d / LIGHT_SPEED)
        E = np.exp(-2j * np.pi * df[None, :, None] * (tau + d / LIGHT_SPEED)[:, None, :])
        H[lo:lo + KBLK] = (E * ph[:, None, :]) @ g           # kb x N_SC x N_ANT
    return H


def trajectory_csi(db, pos, every, sgn):
    """CSI along one trajectory: 4-anchor bilinear blend, run by run over lattice squares."""
    n = len(pos)
    u = pos / ANCHOR_M
    i0 = np.floor(u).astype(np.int64)
    f = u - i0
    ij = i0[:, None, :] + CORNERS                            # n x 4 x 2
    w = (np.where(CORNERS[:, 0], f[:, None, 0], 1 - f[:, None, 0])
         * np.where(CORNERS[:, 1], f[:, None, 1], 1 - f[:, None, 1]))   # n x 4

    rows = db.lookup(ij)                                     # n x 4
    if np.any(rows < 0):
        raise ValueError("anchors missing from the path database for this trajectory")

    npath = db.npath[rows]
    valid = np.any((npath > 0) & (w > 0), axis=1)            # some weighted anchor has a ray
    nearest = np.argmax(w, axis=1)
    los = db.los[rows[np.arange(n), nearest]]

    H = np.zeros((n, N_SC, N_ANT), np.complex128)
    new_square = np.concatenate([[True], np.any(np.diff(i0, axis=0) != 0, axis=1)])
    starts = np.flatnonzero(new_square)
    stops = np.append(starts[1:], n)
    for s0, s1 in zip(starts, stops):
        for c in range(4):
            a = rows[s0, c]
            if npath[s0, c] == 0 or np.all(w[s0:s1, c] == 0):
                continue
            Hc = anchor_csi(db, a, pos[s0:s1] - ij[s0, c] * ANCHOR_M, sgn)
            H[s0:s1] += Hc * w[s0:s1, c, None, None]
    vals = []
    for s in range(0, n, every):
        cs = np.flatnonzero(npath[s] > 0)
        if len(cs) < 2:
            continue
        v = []
        for sg in [sgn, -sgn]:
            Hs = [anchor_csi(db, rows[s, c], pos[s:s + 1] - ij[s, c] * ANCHOR_M, sg).ravel()
                  for c in cs]
            cp = [abs(np.vdot(Hs[p], Hs[q])) / (np.linalg.norm(Hs[p]) * np.linalg.norm(Hs[q]))
                  for p in range(len(cs)) for q in range(p + 1, len(cs))]
            v.append(np.mean(cp))
        vals.append(v)
    coh = np.nanmedian(np.array(vals), axis=0) if vals else np.array([np.nan, np.nan])

    return H.astype(np.complex64), valid, los, new_square, coh


def _tag(ds, cls):
    ds.attrs["MATLAB_class"] = np.bytes_(cls)
    if cls == "logical":
        ds.attrs["MATLAB_int_decode"] = np.int32(1)


def write_matlab_header(path):
    # First 128 bytes of the 512-byte user block: text, subsystem offset, version 0x0200, 'IM'
    text = (f"MATLAB 7.3 MAT-file, Platform: GLNXA64, Created on: "
            f"{datetime.datetime.now():%a %b %d %H:%M:%S %Y} HDF5 schema 1.00 .")
    with open(path, "r+b") as fh:
        fh.write(text.ljust(116).encode("ascii") + bytes(8) + b"\x00\x02IM")


def save_shard(path, H, valid, los, pos, vel, g):
    """Shard with the schema of synthesize_csi.m: HDF5 axes are the MATLAB axes reversed."""
    tmp = path.with_name(path.name + ".part")
    with h5py.File(tmp, "w", userblock_size=512) as f:
        Ht = np.ascontiguousarray(H.transpose(2, 1, 0)).view(CPLX_SINGLE)   # N_ANT x N_SC x N
        _tag(f.create_dataset("H", data=Ht), "single")
        _tag(f.create_dataset("valid", data=valid[None, :].astype("u1")), "logical")
        _tag(f.create_dataset("los", data=los[None, :].astype("u1")), "logical")
        _tag(f.create_dataset("pos", data=np.ascontiguousarray(pos.T, "<f8")), "double")
        _tag(f.create_dataset("vel", data=np.ascontiguousarray(vel.T, "<f8")), "double")
        _tag(f.create_dataset("g", data=np.full((1, 1), g, "<f8")), "double")
    write_matlab_header(tmp)
    tmp.replace(path)


PER_TRAJ = ["written", "frac_valid", "coh_true", "coh_flipped", "jump_cross_db",
            "jump_within_db"]


def parse_traj(spec):
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def select(available, args):
    """Trajectory numbers to synthesize."""
    available = sorted(available)
    if args.all:
        return available
    if args.per_profile is not None:
        count, out = {}, []
        for k in available:
            p = int(profile_of(k))
            if count.get(p, 0) < args.per_profile:
                count[p] = count.get(p, 0) + 1
                out.append(k)
        return out
    want = parse_traj(args.traj)
    missing = sorted(set(want) - set(available))
    if missing:
        raise SystemExit(f"trajectories not in the CSV file: {missing[:10]}"
                         f"{' ...' if len(missing) > 10 else ''}")
    return want


def load_meta(path):
    """Per-trajectory entries of an existing meta.mat for the same trajectory set, else fresh."""
    fresh = {k: np.full(N_TRAJ, np.nan) for k in PER_TRAJ}
    fresh["written"][:] = 0
    if not path.is_file():
        return fresh
    old = sio.loadmat(path, squeeze_me=True)
    if int(old.get("N_TRAJ", -1)) != N_TRAJ or np.ndim(old.get("written")) != 1:
        return fresh
    return {k: np.asarray(old[k], float).copy() for k in PER_TRAJ}


def save_meta(path, site, per):
    group = profile_of(np.arange(1, N_TRAJ + 1)).astype(float)
    col = lambda x: np.asarray(x, float).reshape(-1, 1)               # MATLAB column vector
    bs_lat, bs_lon = SITES[site]
    M = dict(N_TRAJ=N_TRAJ, N_STEPS=N_STEPS, DT=DT, FC=FC, FC_CENTER=FC_CENTER, SCS=SCS,
             N_SC=N_SC, N_ANT=N_ANT, ARRAY_SIZE=np.array(ARRAY_SIZE, float),
             BS_LAT=bs_lat, BS_LON=bs_lon, BS_HEIGHT=BS_HEIGHT, RX_HEIGHT=RX_HEIGHT,
             TX_POWER=TX_POWER, MAX_REFL=MAX_REFL, MAX_DIFF=MAX_DIFF, ANCHOR_M=ANCHOR_M,
             AOA_SIGN=AOA_SIGN, V_MEANS=col(V_MEANS),
             PROFILE_NAMES=np.array(PROFILE_NAMES, dtype=object).reshape(-1, 1),
             traj_group=col(group), traj_v_mean=col(np.array(V_MEANS)[group.astype(int) - 1]),
             **{k: col(per[k]) for k in PER_TRAJ})
    sio.savemat(path, {k: (float(v) if np.isscalar(v) else v) for k, v in M.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv_file", type=Path, help="the site's CSV file, e.g. stecath.csv")
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--traj", help="trajectory numbers, e.g. 1-25,251,300-310")
    which.add_argument("--per-profile", type=int, help="first N trajectories of every profile")
    which.add_argument("--all", action="store_true", help="every trajectory of the CSV file")
    ap.add_argument("--site", choices=sorted(SITES), default=None,
                    help="site of the CSV file (default: from its name)")
    ap.add_argument("--out", type=Path, default=None, help="shard folder (default <site>_shards)")
    ap.add_argument("--coherence-every", type=int, default=100,
                    help="steps between anchor-coherence samples (default 100)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="keep shards already in --out instead of rewriting them")
    args = ap.parse_args()

    site = args.site or args.csv_file.name.split(".")[0]
    if site not in SITES:
        raise SystemExit(f"unknown site {site!r}: name the file <site>.csv or pass --site "
                         f"({', '.join(SITES)})")
    out = args.out or Path(f"{site}_shards")
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"Loading {args.csv_file} ...")
    db, trajectories = read_site(args.csv_file)
    sel = select(trajectories, args)
    print(f"{site}: {len(db.npath)} anchors, {len(db.tau)} paths, {len(trajectories)} "
          f"trajectories; {len(sel)} to synthesize ({time.time() - t0:.0f} s)")

    meta_file = out / "meta.mat"
    per = load_meta(meta_file)
    t0 = time.time()
    for n, k in enumerate(sel, 1):
        shard = out / f"traj_{k:04d}.mat"
        if args.skip_existing and shard.is_file():
            continue
        pos, vel = trajectories[k]
        H, valid, los, new_square, coh = trajectory_csi(db, pos, args.coherence_every, AOA_SIGN)

        # Gain as the model will see it: sum over antennas, mean over subcarriers
        with np.errstate(divide="ignore", invalid="ignore"):
            gain_db = 10 * np.log10(np.mean(np.sum(np.abs(H) ** 2, axis=2), axis=1))
            d = np.abs(np.diff(gain_db))
        ok = np.isfinite(d)
        cross = new_square[1:]
        i = k - 1
        per["frac_valid"][i] = valid.mean()
        per["coh_true"][i], per["coh_flipped"][i] = coh
        per["jump_cross_db"][i] = d[ok & cross].mean() if np.any(ok & cross) else np.nan
        per["jump_within_db"][i] = d[ok & ~cross].mean() if np.any(ok & ~cross) else np.nan

        save_shard(shard, H, valid, los, pos, vel, float(profile_of(k)))
        per["written"][i] = 1
        if n % 25 == 0 or n == len(sel):
            print(f"{n:4d} / {len(sel)} shards  ({(time.time() - t0) / 60:.1f} min)")
            save_meta(meta_file, site, per)
    save_meta(meta_file, site, per)

    done = [k - 1 for k in sel if per["written"][k - 1]]
    print(f"\nShards in {out}; meta.mat marks {int(np.nansum(per['written']))} as written")
    print(f"Valid steps: {100 * np.nanmean(per['frac_valid'][done]):.2f} %")


if __name__ == "__main__":
    main()
