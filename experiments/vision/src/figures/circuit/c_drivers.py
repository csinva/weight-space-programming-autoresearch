"""Panel c data: images that drive the hV4, FFA and PPA readouts of the searched model (erjpu).

Top-predicted images: the four shared LAION images with the highest predicted area response. Predictions come from
readout.py and are averaged over the five subjects; the shared images are held out of every readout fit.
Measured responses are the area-mean z-scored responses, averaged over subjects.

Optimized image: gradient ascent on the subject-mean readout from noise, in a Fourier parametrization with
1/f scaling and random shifts of up to 6 px. The image mean is fixed at 0.5 and its pixel standard deviation
at 0.2; without this constraint the optimum is a blank image. Two global-layout blocks (82 features) are held
constant because they have no usable gradient. Predicted responses are reported as z-scores against the
predictions for the shared images.

Writes artifacts/figures/circuit/data/c.npz. Usage: python src/figures/circuit/c_drivers.py
"""
import numpy as np
import torch

from common import AREAS, DATA, load_model

N, STEPS, N_TOP = 128, 600, 4


def optimise(forward, w, device):
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    fy, fx = torch.fft.fftfreq(N)[:, None], torch.fft.rfftfreq(N)[None, :]
    scale = (1.0 / torch.maximum(torch.sqrt(fx ** 2 + fy ** 2), torch.tensor(1.0 / N))).to(device)
    color = torch.tensor([[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]], device=device)
    p = (torch.randn(1, 3, N, N // 2 + 1, 2, device=device) * 0.01).requires_grad_(True)
    opt = torch.optim.Adam([p], lr=0.05)

    def image():
        t = torch.fft.irfft2(torch.view_as_complex(p) * scale, s=(N, N))
        t = torch.einsum("bchw,dc->bdhw", t, color)
        t = t - t.mean()
        return (0.5 + 0.2 * t / t.std()).clamp(0, 1)

    for _ in range(STEPS):
        dx, dy = rng.randint(-6, 7, 2)
        x = torch.roll(image(), (int(dx), int(dy)), (2, 3))
        loss = -(forward(x.permute(0, 2, 3, 1).reshape(1, -1)) @ w).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        x = image()
        return x[0].permute(1, 2, 0).cpu().numpy(), float(forward(x.permute(0, 2, 3, 1).reshape(1, -1)) @ w)


def main():
    R = np.load(DATA / "readout.npz")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forward, _ = load_model(grad=True)
    W, B = R["w"].mean(0), R["b"].mean(0)
    pred, meas = R["pred"].mean(0), R["meas"].mean(0)                     # (areas, shared images)
    assert list(R["areas"]) == AREAS
    out = {"areas": AREAS, "top": [], "top_meas_z": [], "optimized": [], "optimized_z": [], "r": [],
           "top20_meas_z": []}
    for a, area in enumerate(AREAS):
        order = np.argsort(-pred[a])
        z_meas = (meas[a] - meas[a].mean()) / meas[a].std()
        image, value = optimise(forward, torch.from_numpy(W[a]).to(device), device)
        z_opt = (value + B[a] - pred[a].mean()) / pred[a].std()
        out["top"].append(R["images"][order[:N_TOP]])
        out["top_meas_z"].append(z_meas[order[:N_TOP]])
        out["top20_meas_z"].append(z_meas[order[:20]].mean())
        out["optimized"].append(image)
        out["optimized_z"].append(z_opt)
        out["r"].append(np.corrcoef(pred[a], meas[a])[0, 1])
        print(f"{area}: r = {out['r'][-1]:.3f}, top measured z = {np.round(out['top_meas_z'][-1], 2)}, "
              f"top-20 mean {out['top20_meas_z'][-1]:.2f}, optimized z = {z_opt:.1f}", flush=True)
    np.savez_compressed(DATA / "c.npz", **{k: np.array(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
