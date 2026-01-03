import math
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
        damped_centered=True,
        damped_zero_diagonal=True,
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


def _memorize_pattern(p_vec: np.ndarray, label: str):
    ss = st.session_state
    prev_W = ss.hop.W.copy()
    ss.stored_patterns.append(p_vec.copy())
    ss.stored_labels.append(label)
    ss.hop.memorize(p_vec, labels=[label])
    ss.last_W = prev_W


def _rebuild_after_edit():
    """Rebuild Hopfield state from stored patterns after removal."""
    ss = st.session_state
    ss.hop = _build_hop(ss.hop_config)
    ss.last_W = ss.hop.W.copy()
    if ss.stored_patterns:
        ss.hop.memorize(
            np.stack(ss.stored_patterns, axis=0),
            labels=np.array(ss.stored_labels, dtype=object),
        )
        ss.last_W = ss.hop.W.copy()


def _remove_pattern(label: str):
    ss = st.session_state
    if label not in ss.stored_labels:
        return
    idx = ss.stored_labels.index(label)
    ss.stored_labels.pop(idx)
    ss.stored_patterns.pop(idx)
    _rebuild_after_edit()


def _plot_pattern(pattern: np.ndarray, *, title: str = "", fs=2):
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
    if title != "":
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    
    st.pyplot(fig, clear_figure=True)


def _plot_heatmap(img2d: np.ndarray, *, title: str = "", cmap: str = "viridis", vmin=None, vmax=None):
    """Helpers to plot a heatmap."""

    fig, ax = plt.subplots(figsize=(3, 3), dpi=140)
    sns.heatmap(img2d, ax=ax, cmap=cmap, cbar=True, 
                vmin=vmin, vmax=vmax, square=True, 
                cbar_kws={"shrink": 0.8})
    # Set colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=6)
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    # Set title
    if title != "":
        ax.set_title(title)
    
    st.pyplot(fig, clear_figure=True, width="content")

#------------------------------------------------------------------------------
# Streamlit app
#------------------------------------------------------------------------------

st.set_page_config(page_title="Hopmemo - Streamlit", layout="wide")
_init_state()

st.title("Hopmemo - Hopfieldova síť")

# Style adjustments
st.markdown(
    """
    <style>
    /* Make primary buttons red (used for 'zapomenout') */
    button[data-testid="baseButton-primary"] {
        background-color: #c62828;
        color: #ffffff;
    }
    /* Soft highlight for the global clear button (identified via help/title attr) */
    button[title="clear-all"] {
        background-color: #fff9c4 !important;
        color: #3c3c0a !important;
        border: 1px solid #fdd835 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

#--- Sidebar config ---

with st.sidebar:
    st.header("Učící algoritmus")
    learning_method = st.selectbox(
        "Metoda",
        ["hebbian", "storkey", "pinv centered", "pinv damped"],
        index=["hebbian", "storkey", "pinv_centered", "pinv_damped"].index(
            st.session_state.hop_config["learning_method"] if st.session_state.hop_config else "hebbian"
        ),
    )
    damped_lam = st.slider("lambda (pinv_damped)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)

    st.header("Rekonstrukce")
    theta = st.number_input("theta", value=0.0, step=0.1)
    max_iter = st.number_input("max_iterations", min_value=1, max_value=200, value=50, step=1)
    update_rule = st.selectbox("update_rule", ["async", "sync"], index=0)
    use_local_biases = st.toggle("use_local_biases", value=False)
    noise_p = st.slider("Šum p (na vstupe)", min_value=0.0, max_value=0.5, value=0.10, step=0.01)

config = {
    "learning_method": learning_method,
    "damped_lam": float(damped_lam)
}
_rebuild_hop_if_needed(config)

X_all, y_all = generate_alnum_dataset()
X_all = X_all.reshape(X_all.shape[0], -1)

X_pool = X_all
y_pool = y_all

# --- Dataset grid ---

st.subheader("Dataset")
st.markdown(
    """Klikni na tlačidlo pod obrázkem; Hopfieldova síť si tento vzor zapamatuje tak, že si ho zapíše do svých vah."""
)

n_cols = 8
n_rows = int(math.ceil(len(y_pool) / n_cols))
for row in range(n_rows):
    cols = st.columns(n_cols, gap="small")
    for c_idx in range(n_cols):
        idx = row * n_cols + c_idx
        if idx >= len(y_pool):
            continue

        label = str(y_pool[idx])
        img = _pattern_to_image(X_pool[idx])
        is_memorized = label in st.session_state.stored_labels

        with cols[c_idx]:
            # st.image(img, use_container_width=True, caption=label)
            _plot_pattern(X_pool[idx], title=label, fs=1.5)
            if is_memorized:
                if st.button("zapomenout", key=f"remove_{idx}", use_container_width=True, type="primary"):
                    _remove_pattern(label)
                    st.rerun()
            else:
                if st.button("zapamatovat", key=f"remember_{idx}", use_container_width=True):
                    _memorize_pattern(X_pool[idx].astype(int), label)
                    st.session_state.pool_pos = idx
                    st.rerun()

# Clear all memories
if st.button("zapomenout všechny", type="secondary", use_container_width=True, help="clear-all"):
    st.session_state.stored_patterns = []
    st.session_state.stored_labels = []
    st.session_state.hop.reset_network()
    st.session_state.last_W = st.session_state.hop.W.copy()
    st.rerun()

st.divider()

# --- Weights of the Hopfield network ---

st.subheader("Váhy")
st.markdown(
    """Tady můžeš sledovat váhy Hopfieldovy sítě. V matici vah jsou zapsané všechny vzory, které si síť zapamatovala."""
)

pos = int(np.clip(st.session_state.pool_pos, 0, len(y_pool) - 1))
st.session_state.pool_pos = pos
label = str(y_pool[pos])
p = X_pool[pos].astype(int)

# Current weight matrix and its max abs value
W_now = st.session_state.hop.W

# Display information
st.markdown(f"""
            - :blue[Velikost matice vah:] {W_now.shape[0]:d} x {W_now.shape[1]:d}<br>
            - :blue[Počet uložených vzorů:] {st.session_state.hop.num_memories()}
            """, unsafe_allow_html=True)

# Display weight matrix
m = float(np.max(np.abs(W_now))) or 1.0
_plot_heatmap(W_now, cmap="RdBu_r", vmin=-m, vmax=m)

# --- Retrieval ---

st.divider()
st.subheader("Rekonštrukce vzoru")

if st.session_state.hop.num_memories() == 0:
    st.info("Najprv pridaj vzory cez mriežku vyššie, potom má retrieval zmysel.")
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
