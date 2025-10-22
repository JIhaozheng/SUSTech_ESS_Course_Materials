import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("images", exist_ok=True)
np.random.seed(42)

v = 1500

x0, y0 = 0, 0


def tt(x, y, x0=x0, y0=y0):
    return np.sqrt((x - x0) ** 2 + (y - y0) ** 2) / v


idx = np.linspace(0, 5, 6)
xn = 9000 * np.cos(np.pi / 3 * idx)
yn = 9000 * np.sin(np.pi / 3 * idx)
tn = tt(xn, yn)
t_noise = np.random.normal(0, 0.1, size=idx.shape)
xn = xn
yn = yn
tn = tn + t_noise


class NLSF:
    def __init__(self, xn, yn, tn, v):
        self.xn = xn
        self.yn = yn
        self.tn = tn
        self.v = v

    def _f(self, b):
        b0, b1 = b
        return np.sqrt((self.xn - b0) ** 2 + (self.yn - b1) ** 2) / self.v

    def _residual(self, b):
        obs = self.tn
        pred = self._f(b)
        return pred - obs

    def _jacobian(self, b):
        b0, b1 = b
        dx = self.xn - b0
        dy = self.yn - b1
        r = np.sqrt(dx**2 + dy**2)
        J = np.zeros((len(self.xn), 2))
        J[:, 0] = -dx / (self.v * r)
        J[:, 1] = -dy / (self.v * r)
        return J

    def fit(self, b_init, method="GN", epochs=100, tol=1e-6, miuk=1e-4):
        b = np.array(b_init, dtype=float)
        p_s = []
        p_s.append(b)
        self.method = method
        losses = []
        r = self._residual(b)
        loss = np.sum(r**2)
        losses.append(loss)
        for ii in range(epochs):
            r = self._residual(b)
            J = self._jacobian(b)
            if method == "GN":
                eps = 1e-8
                delta_b = -np.linalg.solve(J.T @ J + eps * np.eye(J.shape[1]), J.T @ r)
            elif method == "LM":
                delta_b = -np.linalg.solve(J.T @ J + miuk * np.eye(J.shape[1]), J.T @ r)
            else:
                raise ValueError("method should be 'GN' or 'LM'")
            b_new = b + delta_b
            r_new = self._residual(b_new)
            loss_new = np.sum(r_new**2)
            if method == "GN":
                b = b_new
                r = r_new
            if method == "LM":
                if loss_new < loss:
                    b = b_new
                    miuk = max(miuk / 10, 1e-7)
                else:
                    miuk = min(miuk * 10, 1e4)

            if np.sqrt(np.sum(r**2)) < tol:
                break
            loss = np.sum(r**2)
            losses.append(loss)
            p_s.append(b)
        return p_s, losses


Model = NLSF(xn, yn, tn, v)
b_init = [10000, 5000]
method1 = "GN"
method2 = "LM"
epochs = 10
tol = 1e-8
miuk = 1e-3
p_s1, losses1 = Model.fit(b_init.copy(), "GN", epochs, tol, miuk)
p_s2, losses2 = Model.fit(b_init.copy(), "LM", epochs, tol, miuk)


##################### Draw the result diagrams######################
# Convert to numpy arrays and get final results
p_s1 = np.array(p_s1)
p_s2 = np.array(p_s2)
b1 = p_s1[-1]  # Final GN result
b2 = p_s2[-1]  # Final LM result

# Calculate errors
gn_error = np.sqrt((b1[0] - x0) ** 2 + (b1[1] - y0) ** 2)
lm_error = np.sqrt((b2[0] - x0) ** 2 + (b2[1] - y0) ** 2)

# Figure 1: Loss curves comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(
    np.arange(len(losses1)), losses1, color="red", label="Gauss-Newton", linewidth=2
)
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Loss (Sum of Squared Residuals)")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_title("Gauss-Newton Loss Curve")
ax1.set_yscale("log")

ax2.plot(
    np.arange(len(losses2)),
    losses2,
    color="blue",
    label="Levenberg-Marquardt",
    linewidth=2,
)
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Loss (Sum of Squared Residuals)")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_title("Levenberg-Marquardt Loss Curve")
ax2.set_yscale("log")

plt.tight_layout()
plt.savefig("images/loss_comparison.png", dpi=300)
plt.show()

# Figure 2: Final positions comparison
plt.figure(figsize=(12, 10))

# Plot receivers
plt.scatter(
    xn,
    yn,
    c="blue",
    s=120,
    marker="s",
    label="Receivers",
    alpha=0.8,
    edgecolors="darkblue",
    linewidth=1,
)

# Annotate receiver numbers and coordinates
for i, (x, y) in enumerate(zip(xn, yn)):
    plt.annotate(
        f"R{i+1}\n({x:.0f}, {y:.0f})",
        (x, y),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=9,
        ha="left",
        va="bottom",
    )

# Plot true source with coordinate annotation
plt.scatter(
    x0,
    y0,
    c="red",
    s=300,
    marker="*",
    label=f"True Source ({x0}, {y0})",
    alpha=0.9,
    edgecolors="darkred",
    linewidth=2,
    zorder=5,
)

# Plot initial guess with coordinate annotation
plt.scatter(
    b_init[0],
    b_init[1],
    c="gray",
    s=150,
    marker="x",
    label=f"Initial Guess ({b_init[0]}, {b_init[1]})",
    alpha=0.8,
    linewidth=3,
    zorder=4,
)


# Plot GN final result with coordinate annotation
plt.scatter(
    b1[0],
    b1[1],
    c="green",
    s=200,
    marker="^",
    label=f"GN Result ({b1[0]:.6f}, {b1[1]:.6f}) Error: {gn_error:.6f}m",
    alpha=0.9,
    edgecolors="darkgreen",
    linewidth=2,
    zorder=6,
)


# Plot LM final result with coordinate annotation
plt.scatter(
    b2[0],
    b2[1],
    c="orange",
    s=200,
    marker="o",
    label=f"LM Result ({b2[0]:.6f}, {b2[1]:.6f}) Error: {lm_error:.6f}m",
    alpha=0.9,
    edgecolors="darkorange",
    linewidth=2,
    zorder=6,
)


plt.scatter(p_s1[:, 0], p_s1[:, 1], label="Evolution of GN")
plt.scatter(p_s2[:, 0], p_s2[:, 1], label="Evolution of LM")
plt.xlabel("X Coordinate (m)", fontsize=12)
plt.ylabel("Y Coordinate (m)", fontsize=12)
plt.title(
    "Source Localization Results: Gauss-Newton vs Levenberg-Marquardt",
    fontsize=14,
    fontweight="bold",
)
plt.legend(fontsize=10, loc="best", framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.axis("equal")

# Set axis range
all_x = np.concatenate([[x0, b_init[0], b1[0], b2[0]], xn])
all_y = np.concatenate([[y0, b_init[1], b1[1], b2[1]], yn])
margin = 1500
plt.xlim(min(all_x) - margin, max(all_x) + margin)
plt.ylim(min(all_y) - margin, max(all_y) + margin)

plt.tight_layout()
plt.savefig("images/final_results.png", dpi=300, bbox_inches="tight")
plt.show()

# Print results
print(f"Initial Guess: ({b_init[0]}, {b_init[1]})")
print(f"True Source: ({x0}, {y0})")
print(
    f"GN Result: ({b1[0]:.2f}, {b1[1]:.2f}), Error: {gn_error:.2f}m, Iterations: {len(p_s1)-1}, Final Loss: {losses1[-1]:.6f}"
)
print(
    f"LM Result: ({b2[0]:.2f}, {b2[1]:.2f}), Error: {lm_error:.2f}m, Iterations: {len(p_s2)-1}, Final Loss: {losses2[-1]:.6f}"
)
