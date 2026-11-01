import numpy as np
from scipy.linalg import expm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sx = np.array([[0, 1], [1, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)
plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
zero = np.array([1, 0], dtype=complex)

def entropy_bits(rho):
    w = np.linalg.eigvalsh(rho)
    w = w[w > 1e-12]
    return float(-(w * np.log2(w)).sum())

# (a) unitary evolution of one qubit, H = (omega/2) sz
omega = 1.0
t = np.linspace(0, 4 * np.pi, 400)
P_plus, S_a = [], []
for tt in t:
    U = expm(-1j * (omega / 2) * sz * tt)
    psi = U @ plus
    P_plus.append(abs(np.vdot(plus, psi)) ** 2)
    S_a.append(entropy_bits(np.outer(psi, psi.conj())))
P_plus = np.array(P_plus)
print("(a) max |P - cos^2(wt/2)| =", np.max(abs(P_plus - np.cos(omega * t / 2) ** 2)))
print("(a) max entropy =", max(S_a))

# (b) channel of version 1: K1 = sx, K2 = sz
rho0 = np.outer(plus, plus.conj())
K = [sx, sz]
print("(b) sum K^dag K without weights =\n", sum(k.conj().T @ k for k in K).real)
raw = sum(k @ rho0 @ k.conj().T for k in K)
print("(b) raw sum, trace =", np.trace(raw).real, "\n", raw.real)
Kw = [k / np.sqrt(2) for k in K]
print("(b) weighted sum K^dag K =\n", sum(k.conj().T @ k for k in Kw).real)
rho1 = sum(k @ rho0 @ k.conj().T for k in Kw)
print("(b) rho after channel =\n", rho1.real, " entropy bits =", entropy_bits(rho1))

# (c) system + N environment qubits, H = sum_k g_k sz (x) sx^(k), global state pure
def kron_list(ops):
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return out

def run(N, gs, times):
    dim = 2 ** (N + 1)
    H = np.zeros((dim, dim), dtype=complex)
    for k in range(N):
        ops = [sz] + [I2] * N
        ops[k + 1] = sx
        H += gs[k] * kron_list(ops)
    psi0 = kron_list([plus] + [zero] * N)
    w, V = np.linalg.eigh(H)
    SA, SG, coh = [], [], []
    for tt in times:
        psi = V @ (np.exp(-1j * w * tt) * (V.conj().T @ psi0))
        M = psi.reshape(2, 2 ** N)
        rhoA = M @ M.conj().T
        SA.append(entropy_bits(rhoA))
        SG.append(entropy_bits(np.outer(psi, psi.conj())) if N <= 8 else 0.0)
        coh.append(abs(rhoA[0, 1]))
    return np.array(SA), np.array(SG), np.array(coh)

times = np.linspace(0, 6, 300)
SA1, SG1, c1 = run(1, [1.0], times)
print("(c) N=1 check coherence vs 0.5|cos 2gt|:", np.max(abs(c1 - 0.5 * abs(np.cos(2 * times)))))
rng = np.random.default_rng(7)
g8 = rng.uniform(0.5, 1.5, 8)
SA8, _, c8 = run(8, g8, times)
analytic8 = 0.5 * np.abs(np.prod([np.cos(2 * g * times) for g in g8], axis=0))
print("(c) N=8 check coherence vs analytic:", np.max(abs(c8 - analytic8)))
SG6 = run(8, g8, times)[1]
print("(c) global entropy, N=8, max =", SG6.max())
print("(c) N=8 mean S_A over t in [2,6] =", SA8[times > 2].mean())
print("g8 =", np.round(g8, 3))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(t, P_plus, color="#1f4e79", lw=2, label=r"$P_+(t)=|\langle + | \psi(t)\rangle|^2$")
ax.plot(t, S_a, color="#b03a2e", lw=1.5, ls="--", label=r"$S(\rho(t))$ (bits)")
ax.set_xlabel(r"$\omega t$")
ax.set_ylim(-0.05, 1.08)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig1.png", dpi=200)

fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(times, SA1, color="#7f8c8d", lw=1.5, label=r"$S(\rho_A)$, N = 1")
ax.plot(times, SA8, color="#1f4e79", lw=2, label=r"$S(\rho_A)$, N = 8")
ax.plot(times, SG6, color="#b03a2e", lw=1.5, ls="--", label=r"$S(\rho_{\mathrm{total}})$")
ax.set_xlabel(r"$g t$")
ax.set_ylabel("bits")
ax.set_ylim(-0.05, 1.1)
ax.legend(loc="lower right", frameon=False)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig2.png", dpi=200)
print("saved")
