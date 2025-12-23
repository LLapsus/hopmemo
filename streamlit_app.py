import numpy as np
import streamlit as st

from alnum_dataset import generate_alnum_dataset
from hopfield import HopfieldNetwork

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns


H = W = 28
N_NEURONS = H * W


def _init_state():
    ss = st.session_state
    ss.setdefault("stored_patterns", [])  # list[np.ndarray] shape (784,)
    ss.setdefault("stored_labels", [])  # list[str]
    ss.setdefault("last_W", None)  # np.ndarray | None
    ss.setdefault("pool_indices", None)  # np.ndarray | None
    ss.setdefault("pool_pos", 0)
    ss.setdefault("rng_seed", 0)
    ss.setdefault("hop_config", None)
    ss.setdefault("hop", None)


def _build_hop(config: dict) -> HopfieldNetwork:
    return HopfieldNetwork(
        n_neurons=N_NEURONS,
        learning_method=config["learning_method"],
        damped_lam=config["damped_lam"],
        damped_centered=config["damped_centered"],
        damped_zero_diagonal=config["damped_zero_diagonal"],
    )


def _rebuild_hop_if_needed(config: dict):
    ss = st.session_state
    if ss.hop is None or ss.hop_config != config:
        ss.hop = _build_hop(config)
        ss.hop_config = dict(config)
        ss.last_W = ss.hop.W.copy()
        if ss.stored_patterns:
            ss.hop.memorize(np.stack(ss.stored_patterns, axis=0), labels=np.array(ss.stored_labels, dtype=object))
            ss.last_W = ss.hop.W.copy()


def _pattern_to_image(p_vec: np.ndarray) -> np.ndarray:
    img = p_vec.reshape(H, W).astype(np.float32)
    img = (img + 1.0) / 2.0
    return img


def _make_pool_indices(labels: np.ndarray, pool_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pool_size = int(np.clip(pool_size, 1, labels.shape[0]))
    return rng.choice(labels.shape[0], size=pool_size, replace=False)


def _plot_pattern(pattern: np.ndarray, *, title: str, fs=2):
    """Plot binary pattern."""
    
    # Reshape pattern to 2D image
    img = pattern.reshape(H, W).astype(np.float32)
    
    # Set colormap
    cmap = sns.color_palette(["#e9e9cd", "#f7f7f7", "#08009c"], as_cmap=True)
    m = max(abs(np.nanmin(img)), abs(np.nanmax(img)))  # keep symmetric scale
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-m, vmax=m)
    
    # Plot pattern
    fig, ax = plt.subplots(figsize=(fs, fs), dpi=140)
    sns.heatmap(img, cmap=cmap, cbar=False, 
                norm=norm, vmin=-1, vmax=1, ax=ax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    st.pyplot(fig, clear_figure=True)


def _plot_heatmap(img2d: np.ndarray, *, title: str, cmap: str = "viridis", vmin=None, vmax=None):
    """Helpers to plot a heatmap."""

    fig, ax = plt.subplots(figsize=(3, 3), dpi=140)
    sns.heatmap(img2d, ax=ax, cmap=cmap, cbar=True, vmin=vmin, vmax=vmax)
    ax.set_title(title, size=12)
    ax.set_xticks([])
    ax.set_yticks([])
    st.pyplot(fig, clear_figure=True)

#------------------------------------------------------------------------------
# Streamlit app
#------------------------------------------------------------------------------

st.set_page_config(page_title="Hopmemo - Streamlit", layout="wide")
_init_state()

st.title("Hopmemo - Hopfieldova síť")

#--- Sidebar config ---

with st.sidebar:
    st.header("Výber vzorov")
    pool_size = st.slider("Veľkosť výberu", min_value=1, max_value=36, value=10, step=1)
    rng_seed = st.number_input("Seed", min_value=0, max_value=10_000, value=int(st.session_state.rng_seed), step=1)

    st.header("Učenie (váhy)")
    learning_method = st.selectbox(
        "Metóda",
        ["hebbian", "storkey", "pinv_centered", "pinv_damped"],
        index=["hebbian", "storkey", "pinv_centered", "pinv_damped"].index(
            st.session_state.hop_config["learning_method"] if st.session_state.hop_config else "hebbian"
        ),
    )
    damped_lam = st.slider("lambda (pinv_damped)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    damped_centered = st.toggle("centered (pinv_damped)", value=True)
    damped_zero_diagonal = st.toggle("zero diagonal", value=True)

    st.header("Retrieval")
    theta = st.number_input("theta", value=0.0, step=0.1)
    max_iter = st.number_input("max_iterations", min_value=1, max_value=200, value=50, step=1)
    update_rule = st.selectbox("update_rule", ["async", "sync"], index=0)
    use_local_biases = st.toggle("use_local_biases", value=False)
    noise_p = st.slider("Šum p (na vstupe)", min_value=0.0, max_value=0.5, value=0.10, step=0.01)

config = {
    "learning_method": learning_method,
    "damped_lam": float(damped_lam),
    "damped_centered": bool(damped_centered),
    "damped_zero_diagonal": bool(damped_zero_diagonal),
}
_rebuild_hop_if_needed(config)

X_all, y_all = generate_alnum_dataset()
X_all = X_all.reshape(X_all.shape[0], -1)

if st.session_state.pool_indices is None or st.button("Resamplovať výber", type="secondary"):
    st.session_state.rng_seed = int(rng_seed)
    st.session_state.pool_indices = _make_pool_indices(y_all, pool_size, seed=int(rng_seed))
    st.session_state.pool_pos = 0

pool_indices = st.session_state.pool_indices
X_pool = X_all[pool_indices]
y_pool = y_all[pool_indices]

colA, colB = st.columns([1.0, 1.0], gap="large")

# --- Current pattern and controls ---

with colA:
    st.subheader("Aktuálny vzor")
    pos = int(np.clip(st.session_state.pool_pos, 0, len(y_pool) - 1))
    st.session_state.pool_pos = pos

    left, mid, right = st.columns([1, 1, 1])
    with left:
        if st.button("⬅ Predošlý", use_container_width=True):
            st.session_state.pool_pos = (pos - 1) % len(y_pool)
            st.rerun()
    with mid:
        if st.button("Náhodný", use_container_width=True):
            st.session_state.pool_pos = int(np.random.default_rng(int(rng_seed)).integers(0, len(y_pool)))
            st.rerun()
    with right:
        if st.button("Ďalší ➡", use_container_width=True):
            st.session_state.pool_pos = (pos + 1) % len(y_pool)
            st.rerun()

    label = str(y_pool[st.session_state.pool_pos])
    p = X_pool[st.session_state.pool_pos].astype(int)
    _plot_pattern(p, title=f"Vzor: {label}", fs=2)

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("Zapamatuj", type="primary", use_container_width=True):
            prev_W = st.session_state.hop.W.copy()
            st.session_state.stored_patterns.append(p.copy())
            st.session_state.stored_labels.append(label)
            st.session_state.hop.memorize(p, labels=[label])
            st.session_state.last_W = prev_W
            st.rerun()
    with c2:
        if st.button("Reset siete", type="secondary", use_container_width=True):
            st.session_state.stored_patterns = []
            st.session_state.stored_labels = []
            st.session_state.hop.reset_network()
            st.session_state.last_W = st.session_state.hop.W.copy()
            st.rerun()
    with c3:
        if st.button("Vymazať posledný", use_container_width=True, disabled=(len(st.session_state.stored_patterns) == 0)):
            st.session_state.stored_patterns.pop()
            st.session_state.stored_labels.pop()
            st.session_state.hop = _build_hop(config)
            st.session_state.hop_config = dict(config)
            if st.session_state.stored_patterns:
                st.session_state.hop.memorize(
                    np.stack(st.session_state.stored_patterns, axis=0),
                    labels=np.array(st.session_state.stored_labels, dtype=object),
                )
            st.session_state.last_W = st.session_state.hop.W.copy()
            st.rerun()

    st.caption(f"Uložené vzory: {st.session_state.hop.num_memories()}")

with colB:
    st.subheader("Váhy")
    W_now = st.session_state.hop.W
    st.markdown("**Matica váh W** (784x784)")
    m = float(np.max(np.abs(W_now))) or 1.0
    _plot_heatmap(W_now, title="W (aktuálne)", cmap="RdBu_r", vmin=-m, vmax=m)

st.divider()
st.subheader("Rekonštrukcia")

if st.session_state.hop.num_memories() == 0:
    st.info("Najprv ulož nejaké vzory (Memorize), potom má retrieval zmysel.")
else:
    rng = np.random.default_rng(0)
    noisy = p.copy()
    n = noisy.size
    k = int(round(float(noise_p) * n))
    if k > 0:
        idx = rng.choice(n, size=k, replace=False)
        noisy[idx] *= -1

    out = st.session_state.hop.retrieve(
        noisy,
        theta=float(theta),
        max_iterations=int(max_iter),
        update_rule=str(update_rule),
        use_local_biases=bool(use_local_biases),
    )

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        _plot_heatmap(_pattern_to_image(p), title=f"Originál: {label}", cmap="gray", vmin=0.0, vmax=1.0)
    with cc2:
        _plot_heatmap(_pattern_to_image(noisy), title=f"Vstup (šum p={noise_p:.2f})", cmap="gray", vmin=0.0, vmax=1.0)
    with cc3:
        _plot_heatmap(_pattern_to_image(out), title="Výstup (po retrieval)", cmap="gray", vmin=0.0, vmax=1.0)

    try:
        best_mem, best_label, best_score = st.session_state.hop.nearest_memory(out, metric="hamming")
        st.caption(f"Najbližšia uložená pamäť (hamming): {best_label} (distance={int(best_score)})")
    except Exception:
        pass
