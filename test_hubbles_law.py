"""Tests for hubbles_law.py. Run with:  python test_hubbles_law.py   (or: pytest)

None of these need the internet. They check the maths against cases that can be worked
out by hand or against closed-form results, then run the fitting code on synthetic data
where the true answer is known. They test the code, not the astrophysics; the real data
is what tests the astrophysics.
"""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import hubbles_law as hl

C = hl.C_KM_S


# ---------------------------------------------------------------- Hubble's 1929 table


def test_1929_table_shape_and_signs():
    d, v = hl.hubble_1929_arrays()
    assert len(d) == 24 and len(v) == 24
    assert np.all(np.diff(d) >= 0), "the paper lists galaxies in order of distance"
    assert d.min() == 0.032 and d.max() == 2.0
    assert (v < 0).sum() == 5, "five nearby galaxies (M31 and friends) are approaching us"


def test_1929_fit_is_pinned():
    """Pins the transcription of Table 1: any typo in a distance or velocity moves these."""
    d, v = hl.hubble_1929_arrays()
    f0, f1 = hl.fit_through_origin(d, v), hl.fit_with_intercept(d, v)
    assert abs(f0.h0 - 423.9) < 0.1 and abs(f0.h0_err - 42.2) < 0.1
    assert abs(f1.h0 - 454.2) < 0.1 and abs(f1.h0_err - 75.2) < 0.1
    assert abs(f1.intercept - (-40.8)) < 0.2


# ---------------------------------------------------------------- linear fits


def test_through_origin_exact_line_has_zero_error():
    f = hl.fit_through_origin([1, 2, 3], [100, 200, 300])
    assert abs(f.h0 - 100) < 1e-12 and f.h0_err < 1e-9 and f.n == 3


def test_through_origin_hand_calculated_error():
    # d = 1, 1 and v = 90, 110: H = 100, residuals -10/+10, s^2 = 200 / (2 - 1) = 200,
    # error = sqrt(s^2 / sum d^2) = sqrt(200 / 2) = 10.
    f = hl.fit_through_origin([1, 1], [90, 110])
    assert abs(f.h0 - 100) < 1e-12
    assert abs(f.h0_err - 10) < 1e-12
    assert abs(f.scatter - np.sqrt(200)) < 1e-12


def test_intercept_fit_recovers_line_and_matches_polyfit_errors():
    d = np.array([0.3, 0.8, 1.1, 1.6, 2.0, 2.4])
    v = np.array([120.0, 380.0, 410.0, 700.0, 760.0, 1010.0])
    f = hl.fit_with_intercept(d, v)
    coef, cov = np.polyfit(d, v, 1, cov=True)  # an independent implementation
    assert abs(f.h0 - coef[0]) < 1e-9 and abs(f.intercept - coef[1]) < 1e-9
    assert abs(f.h0_err - np.sqrt(cov[0, 0])) < 1e-9
    assert abs(f.intercept_err - np.sqrt(cov[1, 1])) < 1e-9
    exact = hl.fit_with_intercept([1, 2, 3, 4], [450, 850, 1250, 1650])  # v = 400 d + 50
    assert abs(exact.h0 - 400) < 1e-9 and abs(exact.intercept - 50) < 1e-9


def test_bootstrap_is_reproducible_and_brackets_the_fit():
    d, v = hl.hubble_1929_arrays()
    a = hl.bootstrap_slope(d, v, True, n_boot=4000, seed=5)
    b = hl.bootstrap_slope(d, v, True, n_boot=4000, seed=5)
    assert np.array_equal(a, b)
    fit = hl.fit_through_origin(d, v)
    assert abs(np.median(a) - fit.h0) < 0.15 * fit.h0_err
    assert 0.9 < np.std(a) / fit.h0_err < 1.1, "bootstrap and formula should agree on the error"
    c = hl.bootstrap_slope(d, v, False, n_boot=4000, seed=5)
    assert np.all(np.isfinite(c))


def test_hubble_time():
    assert abs(hl.hubble_time_gyr(70.0) - 13.968) < 0.005
    assert abs(hl.hubble_time_gyr(100.0) - 9.778) < 0.005  # the famous 9.78 Gyr / h


def test_distance_modulus_hand_cases():
    assert abs(hl.distance_modulus(1e-5)) < 1e-12  # 10 pc: mu = 0
    assert abs(hl.distance_modulus(1.0) - 25.0) < 1e-12  # 1 Mpc: mu = 25
    assert abs(hl.distance_modulus(100.0) - 35.0) < 1e-12
    x = np.array([0.5, 12.0, 830.0])
    assert np.allclose(hl.distance_from_modulus(hl.distance_modulus(x)), x)


# ---------------------------------------------------------------- supernova Hubble flow


def synthetic_supernovae(n=400, h0=70.0, sigma=0.15, seed=3):
    """Fake supernovae that follow the low-redshift Hubble law exactly, plus Gaussian noise."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.02, 0.2, n)
    d_eff = C * z * (1 + 0.5 * (1 - hl.Q0_LOW_Z) * z)  # km/s
    mu = 25 + 5 * np.log10(d_eff / h0) + rng.normal(0, sigma, n)
    return z, mu, np.full(n, sigma)


def test_hubble_flow_fit_exact_and_noisy():
    rng = np.random.default_rng(7)
    z = rng.uniform(0.023, 0.15, 300)
    d_eff = C * z * (1 + 0.5 * (1 - hl.Q0_LOW_Z) * z)  # km/s
    mu_exact = 25 + 5 * np.log10(d_eff / 70.0)
    err = np.full(300, 0.12)
    exact = hl.fit_hubble_flow(z, mu_exact, err)
    assert abs(exact["h0"] - 70.0) < 1e-9 and exact["chi2"] < 1e-12
    noisy = hl.fit_hubble_flow(z, mu_exact + rng.normal(0, 0.12, 300), err)
    assert abs(noisy["h0"] - 70.0) < 4 * noisy["h0_err"]
    boot = hl.bootstrap_hubble_flow(z, mu_exact + rng.normal(0, 0.12, 300), err, n_boot=3000, seed=1)
    assert abs(np.std(boot) / noisy["h0_err"] - 1) < 0.2


def test_hubble_flow_error_matches_hand_formula():
    # One supernova: mu error 0.2 mag -> sigma_log10(H0) = 0.2 / 5, sigma_H0 = H0 ln10 * 0.04.
    z = np.array([0.05])
    mu = np.array([36.8])
    f = hl.fit_hubble_flow(z, mu, np.array([0.2]))
    assert abs(f["h0_err"] / f["h0"] - np.log(10) * 0.04) < 1e-12


# ---------------------------------------------------------------- data handling

FAKE_TABLE = """CID IDSURVEY zHD MU_SH0ES MU_SH0ES_ERR_DIAG IS_CALIBRATOR USED_IN_SH0ES_HF EXTRA
sn_a 1 0.030 35.5 0.30 0 1 x
sn_a 2 0.030 35.6 0.20 0 0 x
sn_b 1 0.001 29.0 0.20 1 0 x
sn_c 1 0.005 32.0 0.20 0 0 x
sn_d 1 0.400 42.0 0.25 0 0 x
sn_e 1 0.080 37.9 0.25 0 1 x
"""


def _write_fake(tmp: Path) -> Path:
    path = tmp / "fake.dat"
    path.write_text(FAKE_TABLE)
    return path


def test_read_pantheon_keeps_only_needed_columns():
    with tempfile.TemporaryDirectory() as t:
        df = hl.read_pantheon(_write_fake(Path(t)))
    assert list(df.columns) == hl.PANTHEON_COLUMNS and len(df) == 6


def test_read_pantheon_rejects_the_wrong_file():
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "bad.dat"
        path.write_text("a b c\n1 2 3\n")
        try:
            hl.read_pantheon(path)
        except ValueError as exc:
            assert "Pantheon" in str(exc)
            return
    raise AssertionError("expected ValueError")


def test_sample_selection():
    with tempfile.TemporaryDirectory() as t:
        df = hl.read_pantheon(_write_fake(Path(t)))
    flow = hl.hubble_flow_sample(df)
    assert list(flow["CID"]) == ["sn_a", "sn_e"]


class FakeResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def read(self):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_load_pantheon_downloads_once_then_uses_cache():
    calls = []

    def opener(url, timeout=None):
        calls.append(url)
        return FakeResponse(FAKE_TABLE.encode())

    with tempfile.TemporaryDirectory() as t:
        first = hl.load_pantheon(t, opener=opener)
        second = hl.load_pantheon(t, opener=opener)
        assert len(calls) == 1 and calls[0] == hl.PANTHEON_URL
        assert first.equals(second)
        assert not list(Path(t).glob("*.part")), "no half-written files left behind"
        hl.load_pantheon(t, refresh=True, opener=opener)
        assert len(calls) == 2


def test_failed_download_leaves_no_cache():
    def opener(url, timeout=None):
        raise OSError("network down")

    with tempfile.TemporaryDirectory() as t:
        try:
            hl.load_pantheon(t, opener=opener)
        except OSError:
            pass
        else:
            raise AssertionError("expected OSError")
        assert not (Path(t) / hl.PANTHEON_CACHE_NAME).exists()


def test_local_data_option_skips_download():
    def opener(url, timeout=None):
        raise AssertionError("should not download")

    with tempfile.TemporaryDirectory() as t:
        df = hl.load_pantheon(t, path=_write_fake(Path(t)), opener=opener)
    assert len(df) == 6


# ---------------------------------------------------------------- expansion simulation


def test_without_peculiar_velocities_every_observer_sees_exactly_h():
    pos, pec = hl.make_universe(n=200, sigma_pec=0.0, seed=2)
    for t in (0.0, 4.0, 9.0):
        x, v = hl.universe_at(pos, pec, t)
        h_t = hl.H0_SIM / hl.scale_factor(t)
        for k in (0, 17, 150):
            dist, v_r = hl.observer_view(x, v, k)
            assert np.allclose(v_r, h_t * dist, rtol=1e-9), (t, k)


def test_any_observer_recovers_the_input_hubble_constant_with_noise():
    pos, pec = hl.make_universe(n=300, seed=4)
    x, v = hl.universe_at(pos, pec, 0.0)
    for k in (0, 40, 120, 250):
        f = hl.fit_through_origin(*hl.observer_view(x, v, k))
        assert abs(f.h0 - hl.H0_SIM) < 4 * f.h0_err + 3.0, (k, f.h0)


def test_hubble_flow_slows_as_the_universe_grows():
    assert hl.scale_factor(0.0) == 1.0
    a = hl.scale_factor(9.0)
    assert abs(a - (1 + 70 * hl.KMS_TO_MPC_PER_GYR * 9.0)) < 1e-12
    assert 1.6 < a < 1.7


def test_choose_observers_are_distinct_and_at_different_radii():
    pos, _ = hl.make_universe(seed=1)
    a, b = hl.choose_observers(pos)
    assert a != b
    assert np.linalg.norm(pos[a]) < 0.2 * np.linalg.norm(pos[b])


# ---------------------------------------------------------------- figures and CLI


def test_pipeline_writes_real_figures_and_results():
    old = os.getcwd()
    with tempfile.TemporaryDirectory() as t:
        os.chdir(t)
        try:
            out = Path("figs")
            rc = hl.main(["--no-supernovae", "--no-animation", "--out-dir", str(out), "--results", "r.json"])
            assert rc == 0
            for name in ("fig1_hubble_1929.png", "fig2_hubble_1929_analysis.png", "fig5_two_observers.png"):
                assert (out / name).stat().st_size > 20_000, name
            data = json.loads(Path("r.json").read_text())
            assert abs(data["hubble_1929"]["through_origin"]["h0"] - 423.9) < 0.1
        finally:
            os.chdir(old)


def test_supernova_figures_from_synthetic_data():
    z, mu, err = synthetic_supernovae(n=400)
    df = pd.DataFrame({"CID": [f"s{i}" for i in range(len(z))], "zHD": z, "MU_SH0ES": mu, "MU_SH0ES_ERR_DIAG": err,
                       "IS_CALIBRATOR": 0, "USED_IN_SH0ES_HF": ((z > 0.023) & (z < 0.15)).astype(int)})
    hl.apply_style()
    with tempfile.TemporaryDirectory() as t:
        results, rows = hl.report_supernovae(df, Path(t))
        hl.plot_h0_summary(rows, Path(t) / "summary.png")
        for name in ("fig3_hubble_flow.png", "summary.png"):
            assert (Path(t) / name).stat().st_size > 20_000, name
    assert abs(results["hubble_flow"]["h0"] - 70.0) < 4 * results["hubble_flow"]["h0_err"] + 0.5


def test_animation_writes_a_multi_frame_gif():
    from PIL import Image

    hl.apply_style()
    with tempfile.TemporaryDirectory() as t:
        out = Path(t) / "e.gif"
        hl.animate_expansion(out, frames=4)
        with Image.open(out) as im:
            assert im.n_frames == 4


def test_missing_supernova_data_gives_a_clear_error():
    with tempfile.TemporaryDirectory() as t:
        rc = hl.main(["--out-dir", t, "--data", str(Path(t) / "nope.dat"), "--no-animation", "--results", str(Path(t) / "r.json")])
    assert rc == 1


if __name__ == "__main__":
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, func in tests:
        try:
            func()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001 - report every failure, not just the first
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
