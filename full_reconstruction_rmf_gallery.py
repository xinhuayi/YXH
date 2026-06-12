import os
import csv
import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    from scipy.interpolate import splprep, splev
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    import tensorflow as tf
    HAS_TF = True
except Exception:
    HAS_TF = False

# ============================================================
# Config
# ============================================================

BASE_OUTPUT_DIR = "reviewer_full_shape_gallery"

SHAPE_TAGS = [
    "open_rectangle_like",
    "helix_like",
    "s_shape_like",
    "l_shape_like",
]

DATA_DIR = "generated_wavelength_data"

THEORY_CALIB_FILE = "calibration_params_theory.csv"
STANDARD_CALIB_FILE = "calibration_params_standard.csv"

MODEL_PATH = "residual_model/final_model.keras"
SCALER_X_NPY = "residual_model/scaler_x.json"
SCALER_Y_NPY = "residual_model/scaler_y.json"

NUM_SAMPLES_PER_TAG = 50
SAMPLE_SELECTION_MODE = "even"
RANDOM_SEED = 42

NUM_SENSORS = 3
NUM_NODES = 12
ROD_LENGTH_MM = 1000.0
DS = ROD_LENGTH_MM / (NUM_NODES - 1)

THETA_THEORETICAL = np.array([0.0, 120.0, -120.0], dtype=np.float64) * np.pi / 180.0

SMOOTH_POINTS = 200
N_COLS = 5
FIG_DPI = 220

SHOW_NODE_MARKERS = True
NODE_MARKER_SIZE = 8
LINE_WIDTH = 1.3

STYLE_STANDARD = {
    "color": "#1f77b4",
    "label": "Standard / True",
    "linestyle": "-",
    "linewidth": LINE_WIDTH,
    "alpha": 0.95,
    "marker": "o",
}
STYLE_THEORY = {
    "color": "#d62728",
    "label": "Theory",
    "linestyle": "--",
    "linewidth": LINE_WIDTH,
    "alpha": 0.90,
    "marker": "^",
}
STYLE_MODEL = {
    "color": "#2ca02c",
    "label": "Model",
    "linestyle": "-.",
    "linewidth": LINE_WIDTH,
    "alpha": 0.95,
    "marker": "s",
}

# ============================================================
# Basic utils
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n

def load_calibration_params_from_csv(filename):
    with open(filename, "r", newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        rows = list(reader)

    num_points = max(int(row[0]) for row in rows)
    num_sensors = max(int(row[1]) for row in rows)

    params = np.zeros((num_points, num_sensors, 3), dtype=np.float64)
    for row in rows:
        pos_id = int(row[0]) - 1
        sensor_id = int(row[1]) - 1
        params[pos_id, sensor_id] = [float(row[2]), float(row[3]), float(row[4])]
    return params

def load_sensor_changes_from_csv(filename):
    df = pd.read_csv(filename, encoding="utf-8-sig")

    required_cols = ["sample_id", "detection_pos"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column `{c}` in {filename}")

    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    if len(sensor_cols) != NUM_SENSORS:
        raise ValueError(f"Expected {NUM_SENSORS} sensor cols, got {sensor_cols}")

    sample_ids_unique = df["sample_id"].drop_duplicates().tolist()
    num_samples = len(sample_ids_unique)

    wavelength = np.zeros((num_samples, NUM_NODES, NUM_SENSORS), dtype=np.float64)

    sample_id_to_idx = {sid: i for i, sid in enumerate(sample_ids_unique)}

    for _, row in df.iterrows():
        sidx = sample_id_to_idx[row["sample_id"]]
        pidx = int(row["detection_pos"]) - 1
        wavelength[sidx, pidx, :] = row[sensor_cols].to_numpy(dtype=np.float64)

    return wavelength, sample_ids_unique

def save_json(obj, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# ============================================================
# Analytical inverse
# ============================================================

def inverse_one_node(calib_node, measured_lambda_node):
    """
    calib_node shape: (3, 3), each sensor: [kT, kEps, alpha_deg]
    measured_lambda_node shape: (3,)
    solve:
        dl_i = kT * dT + kEps * (k1*cos(theta_i + alpha_i) - k2*sin(theta_i + alpha_i))
    unknown = [k1, k2, dT]
    """
    A = np.zeros((NUM_SENSORS, 3), dtype=np.float64)
    b = measured_lambda_node.astype(np.float64).copy()

    for i in range(NUM_SENSORS):
        kT, kEps, alpha_deg = calib_node[i]
        theta = THETA_THEORETICAL[i] + np.deg2rad(alpha_deg)

        A[i, 0] = kEps * np.cos(theta)
        A[i, 1] = -kEps * np.sin(theta)
        A[i, 2] = kT

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return x  # [k1, k2, deltaT]

def inverse_all_samples(calibration_params, measured_lambda):
    num_samples = measured_lambda.shape[0]
    states = np.zeros((num_samples, NUM_NODES, 3), dtype=np.float64)

    for s in range(num_samples):
        for p in range(NUM_NODES):
            states[s, p] = inverse_one_node(calibration_params[p], measured_lambda[s, p])

    return states

# ============================================================
# RMF reconstruction
# ============================================================

def rotation_matrix_from_axis_angle(axis, angle):
    axis = normalize(axis)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c

    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=np.float64)

def rotation_matrix_from_vector(rotvec):
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = rotvec / angle
    return rotation_matrix_from_axis_angle(axis, angle)

def rmf_from_k1k2_np(states_k1k2dt, ds=DS):
    """
    states_k1k2dt: (N, 3) -> [k1, k2, deltaT]
    return: (N, 3) centerline
    """
    states = np.asarray(states_k1k2dt, dtype=np.float64)
    n = states.shape[0]

    pts = np.zeros((n, 3), dtype=np.float64)

    t = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    nvec = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    bvec = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    for i in range(n - 1):
        k1 = states[i, 0]
        k2 = states[i, 1]

        curvature_vec = k1 * nvec + k2 * bvec
        rotvec = curvature_vec * ds
        R = rotation_matrix_from_vector(rotvec)

        t = normalize(R @ t)
        nvec = normalize(R @ nvec)
        bvec = normalize(np.cross(t, nvec))

        pts[i + 1] = pts[i] + ds * t

    return pts

def rmf_from_batch(states_batch, ds=DS):
    num_samples = states_batch.shape[0]
    out = np.zeros((num_samples, NUM_NODES, 3), dtype=np.float64)
    for i in range(num_samples):
        out[i] = rmf_from_k1k2_np(states_batch[i], ds=ds)
    return out

# ============================================================
# State conversion / metrics
# ============================================================

def k1k2_to_kappa_phi_deltaT(states):
    out = np.zeros_like(states)
    k1 = states[..., 0]
    k2 = states[..., 1]
    dT = states[..., 2]

    out[..., 0] = np.sqrt(k1 * k1 + k2 * k2)
    out[..., 1] = np.arctan2(k2, k1)
    out[..., 2] = dT
    return out

def compute_tip_errors(pred_shape, true_shape):
    return np.linalg.norm(pred_shape[:, -1, :] - true_shape[:, -1, :], axis=1)

def compute_mean_point_errors(pred_shape, true_shape):
    return np.linalg.norm(pred_shape - true_shape, axis=2).mean(axis=1)

def select_indices(standard_shape, theory_shape, model_shape, k=50, mode="even", seed=42):
    n = standard_shape.shape[0]
    k = min(k, n)

    model_tip = compute_tip_errors(model_shape, standard_shape)
    order = np.argsort(model_tip)

    if mode == "best":
        return order[:k].tolist()

    if mode == "worst":
        return order[-k:].tolist()

    if mode == "median":
        center = n // 2
        half = k // 2
        start = max(0, center - half)
        end = min(n, start + k)
        idx = order[start:end]
        if len(idx) < k:
            idx = order[max(0, end - k):end]
        return idx.tolist()

    if mode == "random":
        rng = np.random.default_rng(seed)
        return rng.choice(n, size=k, replace=False).tolist()

    if mode == "even":
        if k == 1:
            return [n // 2]
        return np.linspace(0, n - 1, k, dtype=int).tolist()

    raise ValueError(f"Unknown selection mode: {mode}")

# ============================================================
# Deep learning model utils
# ============================================================

def build_node_features(measured_lambda, theory_calib):
    """
    输入特征可按你训练时的方法调整。
    这里给一个通用稳定版:
    [dl1, dl2, dl3, kT1, kEps1, alpha1, kT2, kEps2, alpha2, kT3, kEps3, alpha3]
    """
    num_samples = measured_lambda.shape[0]
    x = np.zeros((num_samples, NUM_NODES, NUM_SENSORS + NUM_SENSORS * 3), dtype=np.float64)

    for s in range(num_samples):
        for p in range(NUM_NODES):
            feats = []
            feats.extend(measured_lambda[s, p].tolist())
            for sensor_idx in range(NUM_SENSORS):
                feats.extend(theory_calib[p, sensor_idx].tolist())
            x[s, p] = np.array(feats, dtype=np.float64)

    return x

def load_scaler_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def transform_with_scaler(x, scaler):
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    std = np.asarray(scaler["std"], dtype=np.float64)
    return (x - mean) / np.maximum(std, 1e-12)

def inverse_transform_with_scaler(x, scaler):
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    std = np.asarray(scaler["std"], dtype=np.float64)
    return x * np.maximum(std, 1e-12) + mean

def predict_model_states(model_path, scaler_x_path, scaler_y_path, x):
    if not HAS_TF:
        raise ImportError("TensorFlow is not installed, cannot run model inference.")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model: {model_path}")
    if not os.path.exists(scaler_x_path):
        raise FileNotFoundError(f"Missing scaler: {scaler_x_path}")
    if not os.path.exists(scaler_y_path):
        raise FileNotFoundError(f"Missing scaler: {scaler_y_path}")

    model = tf.keras.models.load_model(model_path, compile=False)

    scaler_x = load_scaler_json(scaler_x_path)
    scaler_y = load_scaler_json(scaler_y_path)

    x2 = x.reshape(-1, x.shape[-1])
    x2s = transform_with_scaler(x2, scaler_x)
    xs = x2s.reshape(x.shape)

    y_pred_s = model.predict(xs, verbose=0)
    y2s = y_pred_s.reshape(-1, y_pred_s.shape[-1])
    y2 = inverse_transform_with_scaler(y2s, scaler_y)
    y = y2.reshape(y_pred_s.shape)

    return y.astype(np.float64)

# ============================================================
# Plotting utils
# ============================================================

def smooth_curve_xyz(points_xyz, smooth_points=200):
    pts = np.asarray(points_xyz, dtype=np.float64)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected shape (N, 3), got {pts.shape}")

    if pts.shape[0] < 2:
        return pts.copy()

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]

    if total < 1e-12:
        return np.repeat(pts[:1], smooth_points, axis=0)

    u_new = np.linspace(0.0, total, smooth_points)

    if HAS_SCIPY and pts.shape[0] >= 4:
        try:
            k = min(3, pts.shape[0] - 1)
            tck, _ = splprep(
                [pts[:, 0], pts[:, 1], pts[:, 2]],
                u=s,
                s=0.0,
                k=k,
            )
            x_new, y_new, z_new = splev(u_new, tck)
            return np.stack([x_new, y_new, z_new], axis=1)
        except Exception:
            pass

    x_new = np.interp(u_new, s, pts[:, 0])
    y_new = np.interp(u_new, s, pts[:, 1])
    z_new = np.interp(u_new, s, pts[:, 2])
    return np.stack([x_new, y_new, z_new], axis=1)

def set_equal_3d_axes(ax, curves, margin_ratio=0.08):
    pts = np.concatenate(curves, axis=0)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = max(maxs - mins)
    radius = 0.5 * span * (1.0 + margin_ratio)
    if radius < 1e-6:
        radius = 1.0

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

def plot_one_sample(ax, standard_curve, theory_curve, model_curve, title_text):
    sm_standard = smooth_curve_xyz(standard_curve, smooth_points=SMOOTH_POINTS)
    sm_theory = smooth_curve_xyz(theory_curve, smooth_points=SMOOTH_POINTS)
    sm_model = smooth_curve_xyz(model_curve, smooth_points=SMOOTH_POINTS)

    ax.plot(
        sm_standard[:, 0], sm_standard[:, 1], sm_standard[:, 2],
        color=STYLE_STANDARD["color"],
        linestyle=STYLE_STANDARD["linestyle"],
        linewidth=STYLE_STANDARD["linewidth"],
        alpha=STYLE_STANDARD["alpha"],
    )
    ax.plot(
        sm_theory[:, 0], sm_theory[:, 1], sm_theory[:, 2],
        color=STYLE_THEORY["color"],
        linestyle=STYLE_THEORY["linestyle"],
        linewidth=STYLE_THEORY["linewidth"],
        alpha=STYLE_THEORY["alpha"],
    )
    ax.plot(
        sm_model[:, 0], sm_model[:, 1], sm_model[:, 2],
        color=STYLE_MODEL["color"],
        linestyle=STYLE_MODEL["linestyle"],
        linewidth=STYLE_MODEL["linewidth"],
        alpha=STYLE_MODEL["alpha"],
    )

    if SHOW_NODE_MARKERS:
        ax.scatter(
            standard_curve[:, 0], standard_curve[:, 1], standard_curve[:, 2],
            color=STYLE_STANDARD["color"],
            s=NODE_MARKER_SIZE,
            alpha=0.85,
            marker=STYLE_STANDARD["marker"],
        )
        ax.scatter(
            theory_curve[:, 0], theory_curve[:, 1], theory_curve[:, 2],
            color=STYLE_THEORY["color"],
            s=NODE_MARKER_SIZE,
            alpha=0.85,
            marker=STYLE_THEORY["marker"],
        )
        ax.scatter(
            model_curve[:, 0], model_curve[:, 1], model_curve[:, 2],
            color=STYLE_MODEL["color"],
            s=NODE_MARKER_SIZE,
            alpha=0.85,
            marker=STYLE_MODEL["marker"],
        )

    set_equal_3d_axes(ax, [sm_standard, sm_theory, sm_model])

    ax.set_title(title_text, fontsize=8, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")

def make_summary_text(tag, standard_shape, theory_shape, model_shape):
    theory_point = compute_mean_point_errors(theory_shape, standard_shape)
    model_point = compute_mean_point_errors(model_shape, standard_shape)
    theory_tip = compute_tip_errors(theory_shape, standard_shape)
    model_tip = compute_tip_errors(model_shape, standard_shape)

    mean_theory_point = float(np.mean(theory_point))
    mean_model_point = float(np.mean(model_point))
    mean_theory_tip = float(np.mean(theory_tip))
    mean_model_tip = float(np.mean(model_tip))

    ratio_point = mean_theory_point / max(mean_model_point, 1e-12)
    ratio_tip = mean_theory_tip / max(mean_model_tip, 1e-12)

    text = (
        f"{tag}\n"
        f"Mean point error: theory {mean_theory_point:.2f} mm | model {mean_model_point:.2f} mm | x{ratio_point:.1f}\n"
        f"Mean tip error: theory {mean_theory_tip:.2f} mm | model {mean_model_tip:.2f} mm | x{ratio_tip:.1f}"
    )
    return text

# ============================================================
# Per-tag processing
# ============================================================

def process_one_tag(tag):
    wavelength_file = os.path.join(DATA_DIR, f"wavelength_changes_{tag}.csv")
    if not os.path.exists(wavelength_file):
        raise FileNotFoundError(f"Missing wavelength file: {wavelength_file}")

    print(f"\nLoading wavelength: {wavelength_file}")
    measured_lambda, sample_ids = load_sensor_changes_from_csv(wavelength_file)

    print("Loading calibration files...")
    theory_calib = load_calibration_params_from_csv(THEORY_CALIB_FILE)
    standard_calib = load_calibration_params_from_csv(STANDARD_CALIB_FILE)

    if theory_calib.shape != (NUM_NODES, NUM_SENSORS, 3):
        raise ValueError(f"Theory calibration shape mismatch: {theory_calib.shape}")
    if standard_calib.shape != (NUM_NODES, NUM_SENSORS, 3):
        raise ValueError(f"Standard calibration shape mismatch: {standard_calib.shape}")

    print("Computing analytical inverse states...")
    theory_states = inverse_all_samples(theory_calib, measured_lambda)
    standard_states = inverse_all_samples(standard_calib, measured_lambda)

    print("Building theory and standard RMF shapes...")
    theory_shape = rmf_from_batch(theory_states)
    standard_shape = rmf_from_batch(standard_states)

    print("Building features for deep model...")
    x = build_node_features(measured_lambda, theory_calib)

    print("Predicting model states...")
    model_states = predict_model_states(
        model_path=MODEL_PATH,
        scaler_x_path=SCALER_X_NPY,
        scaler_y_path=SCALER_Y_NPY,
        x=x,
    )

    if model_states.shape != (measured_lambda.shape[0], NUM_NODES, 3):
        raise ValueError(f"Unexpected model output shape: {model_states.shape}")

    print("Building model RMF shapes...")
    model_shape = rmf_from_batch(model_states)

    return {
        "tag": tag,
        "sample_ids": sample_ids,
        "wavelength": measured_lambda,
        "theory_states": theory_states,
        "standard_states": standard_states,
        "model_states": model_states,
        "theory_shape": theory_shape,
        "standard_shape": standard_shape,
        "model_shape": model_shape,
    }

def make_tag_gallery(tag_data, out_dir, num_samples=50, selection_mode="even", seed=42):
    tag = tag_data["tag"]
    standard_shape = tag_data["standard_shape"]
    theory_shape = tag_data["theory_shape"]
    model_shape = tag_data["model_shape"]
    sample_ids = tag_data["sample_ids"]

    indices = select_indices(
        standard_shape=standard_shape,
        theory_shape=theory_shape,
        model_shape=model_shape,
        k=num_samples,
        mode=selection_mode,
        seed=seed,
    )

    n = len(indices)
    n_cols = N_COLS
    n_rows = math.ceil(n / n_cols)

    fig = plt.figure(figsize=(n_cols * 4.2, n_rows * 3.7), constrained_layout=False)

    summary_text = make_summary_text(tag, standard_shape, theory_shape, model_shape)
    fig.suptitle(summary_text, fontsize=14, y=0.995)

    handles = []
    labels = []

    rows = []

    for plot_i, sample_idx in enumerate(indices, start=1):
        ax = fig.add_subplot(n_rows, n_cols, plot_i, projection="3d")

        standard_curve = standard_shape[sample_idx]
        theory_curve = theory_shape[sample_idx]
        model_curve = model_shape[sample_idx]

        sample_name = str(sample_idx)
        if sample_ids is not None and sample_idx < len(sample_ids):
            sample_name = str(sample_ids[sample_idx])

        theory_tip = float(np.linalg.norm(theory_curve[-1] - standard_curve[-1]))
        model_tip = float(np.linalg.norm(model_curve[-1] - standard_curve[-1]))
        theory_point = float(np.mean(np.linalg.norm(theory_curve - standard_curve, axis=1)))
        model_point = float(np.mean(np.linalg.norm(model_curve - standard_curve, axis=1)))

        title_text = (
            f"id={sample_name}\n"
            f"T:{theory_tip:.2f}  M:{model_tip:.2f} mm"
        )

        plot_one_sample(
            ax=ax,
            standard_curve=standard_curve,
            theory_curve=theory_curve,
            model_curve=model_curve,
            title_text=title_text,
        )

        if plot_i == 1:
            line1, = ax.plot([], [], [], color=STYLE_STANDARD["color"],
                             linestyle=STYLE_STANDARD["linestyle"],
                             linewidth=STYLE_STANDARD["linewidth"],
                             label=STYLE_STANDARD["label"])
            line2, = ax.plot([], [], [], color=STYLE_THEORY["color"],
                             linestyle=STYLE_THEORY["linestyle"],
                             linewidth=STYLE_THEORY["linewidth"],
                             label=STYLE_THEORY["label"])
            line3, = ax.plot([], [], [], color=STYLE_MODEL["color"],
                             linestyle=STYLE_MODEL["linestyle"],
                             linewidth=STYLE_MODEL["linewidth"],
                             label=STYLE_MODEL["label"])
            handles = [line1, line2, line3]
            labels = [STYLE_STANDARD["label"], STYLE_THEORY["label"], STYLE_MODEL["label"]]

        rows.append({
            "tag": tag,
            "sample_idx": int(sample_idx),
            "sample_name": sample_name,
            "theory_mean_point_error_mm": theory_point,
            "model_mean_point_error_mm": model_point,
            "theory_tip_error_mm": theory_tip,
            "model_tip_error_mm": model_tip,
            "tip_improvement_ratio": theory_tip / max(model_tip, 1e-12),
        })

    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.965))
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])

    png_path = os.path.join(out_dir, f"{tag}_gallery_{n}_samples.png")
    pdf_path = os.path.join(out_dir, f"{tag}_gallery_{n}_samples.pdf")
    csv_path = os.path.join(out_dir, f"{tag}_gallery_{n}_samples_summary.csv")

    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {csv_path}")

def save_tag_arrays(tag_data, out_dir):
    np.save(os.path.join(out_dir, "standard_shape.npy"), tag_data["standard_shape"])
    np.save(os.path.join(out_dir, "theory_shape.npy"), tag_data["theory_shape"])
    np.save(os.path.join(out_dir, "model_shape.npy"), tag_data["model_shape"])

    np.save(os.path.join(out_dir, "standard_states.npy"), tag_data["standard_states"])
    np.save(os.path.join(out_dir, "theory_states.npy"), tag_data["theory_states"])
    np.save(os.path.join(out_dir, "model_states.npy"), tag_data["model_states"])

    pd.DataFrame({"sample_id": tag_data["sample_ids"]}).to_csv(
        os.path.join(out_dir, "sample_ids.csv"), index=False, encoding="utf-8-sig"
    )

    theory_point = compute_mean_point_errors(tag_data["theory_shape"], tag_data["standard_shape"])
    model_point = compute_mean_point_errors(tag_data["model_shape"], tag_data["standard_shape"])
    theory_tip = compute_tip_errors(tag_data["theory_shape"], tag_data["standard_shape"])
    model_tip = compute_tip_errors(tag_data["model_shape"], tag_data["standard_shape"])

    metrics = {
        "tag": tag_data["tag"],
        "num_samples": int(tag_data["standard_shape"].shape[0]),
        "theory_mean_point_error_mm": float(np.mean(theory_point)),
        "model_mean_point_error_mm": float(np.mean(model_point)),
        "theory_rmse_point_error_mm": float(np.sqrt(np.mean(np.square(theory_point)))),
        "model_rmse_point_error_mm": float(np.sqrt(np.mean(np.square(model_point)))),
        "theory_mean_tip_error_mm": float(np.mean(theory_tip)),
        "model_mean_tip_error_mm": float(np.mean(model_tip)),
        "theory_p95_tip_error_mm": float(np.percentile(theory_tip, 95)),
        "model_p95_tip_error_mm": float(np.percentile(model_tip, 95)),
    }

    save_json(metrics, os.path.join(out_dir, "metrics.json"))

# ============================================================
# Main
# ============================================================

def main():
    print("\n========== Full Reconstruction + RMF + Gallery ==========")
    print(f"DATA_DIR = {DATA_DIR}")
    print(f"THEORY_CALIB_FILE = {THEORY_CALIB_FILE}")
    print(f"STANDARD_CALIB_FILE = {STANDARD_CALIB_FILE}")
    print(f"MODEL_PATH = {MODEL_PATH}")
    print(f"OUTPUT_DIR = {BASE_OUTPUT_DIR}")

    ensure_dir(BASE_OUTPUT_DIR)

    all_rows = []

    for tag in SHAPE_TAGS:
        print("\n" + "=" * 90)
        print(f"Processing tag: {tag}")
        print("=" * 90)

        try:
            tag_data = process_one_tag(tag)
        except Exception as e:
            print(f"[Skip] {tag}: {e}")
            continue

        tag_out_dir = os.path.join(BASE_OUTPUT_DIR, tag)
        ensure_dir(tag_out_dir)

        save_tag_arrays(tag_data, tag_out_dir)
        make_tag_gallery(
            tag_data=tag_data,
            out_dir=tag_out_dir,
            num_samples=NUM_SAMPLES_PER_TAG,
            selection_mode=SAMPLE_SELECTION_MODE,
            seed=RANDOM_SEED,
        )

        theory_point = compute_mean_point_errors(tag_data["theory_shape"], tag_data["standard_shape"])
        model_point = compute_mean_point_errors(tag_data["model_shape"], tag_data["standard_shape"])
        theory_tip = compute_tip_errors(tag_data["theory_shape"], tag_data["standard_shape"])
        model_tip = compute_tip_errors(tag_data["model_shape"], tag_data["standard_shape"])

        all_rows.append({
            "tag": tag,
            "num_samples": int(tag_data["standard_shape"].shape[0]),
            "theory_mean_point_error_mm": float(np.mean(theory_point)),
            "model_mean_point_error_mm": float(np.mean(model_point)),
            "theory_mean_tip_error_mm": float(np.mean(theory_tip)),
            "model_mean_tip_error_mm": float(np.mean(model_tip)),
            "improvement_mean_point": float(np.mean(theory_point) / max(np.mean(model_point), 1e-12)),
            "improvement_mean_tip": float(np.mean(theory_tip) / max(np.mean(model_tip), 1e-12)),
        })

    if all_rows:
        summary_csv = os.path.join(BASE_OUTPUT_DIR, "validation_summary.csv")
        summary_df = pd.DataFrame(all_rows)
        summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        print("\n" + "=" * 90)
        print("Overall validation summary")
        print("=" * 90)
        print(summary_df.to_string(index=False))
        print(f"\nSaved overall summary: {summary_csv}")
    else:
        print("\nNo tags were processed successfully; nothing to summarize.")

    print("\n========== Done ==========")


if __name__ == "__main__":
    main()
