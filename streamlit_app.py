import hashlib
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


def _hash_array(arr: np.ndarray) -> str:
    """Stable hash for caching diagnostics."""
    a = np.asarray(arr)
    return hashlib.md5(a.view(np.uint8)).hexdigest()


def _as_pm1(arr: np.ndarray) -> np.ndarray:
    """Coerce patterns to +/-1."""
    a = np.asarray(arr)
    if a.size == 0:
        return a.astype(float)
    amin, amax = a.min(), a.max()
    if amin >= 0 and amax <= 1:
        return (2 * a - 1).astype(float)
    return a.astype(float)


def _compute_diagnostics(patterns: list[np.ndarray], W: np.ndarray) -> dict:
    """
    Compute load, pairwise correlations, and stability margins.
    Returns dict with alpha, corr_mean, corr_max, unstable_frac, percentiles, margins_flat.
    """
    if not patterns:
        return {
            "P": 0,
            "N": W.shape[0],
            "alpha": 0.0,
            "corr_mean": np.nan,
            "corr_max": np.nan,
            "unstable_frac": np.nan,
            "percentiles": (np.nan, np.nan, np.nan),
            "margins_flat": np.array([]),
        }

    X = np.stack(patterns, axis=0)
    P, N = X.shape[0], X.shape[1]
    X_pm = _as_pm1(X)

    # Calculate load
    alpha = P / float(N)

    # Calculate pairwise correlations between patterns
    if P < 2:
        corr_mean = np.nan
        corr_max = np.nan
    else:
        C = (X_pm @ X_pm.T) / float(N)
        iu = np.triu_indices(P, k=1)
        vals = np.abs(C[iu])
        corr_mean = float(vals.mean()) if vals.size else np.nan
        corr_max = float(vals.max()) if vals.size else np.nan

    # Calculate fraction of unstable bits across all patterns and neurons
    H = X_pm @ W
    M = X_pm * H
    margins_flat = M.ravel()
    unstable_frac = float((margins_flat < 0).mean()) if margins_flat.size else np.nan
    percentiles = tuple(np.percentile(margins_flat, [5, 50, 95])) if margins_flat.size else (np.nan, np.nan, np.nan)

    return {
        "P": P,
        "N": N,
        "alpha": float(alpha),
        "corr_mean": corr_mean,
        "corr_max": corr_max,
        "unstable_frac": unstable_frac,
        "percentiles": percentiles,
        "margins_flat": margins_flat,
    }


def _compute_diagnostics_cached(patterns: list[np.ndarray], W: np.ndarray) -> dict:
    """Cache diagnostics based on pattern/W hashes to avoid recompute."""
    ss = st.session_state
    ss.setdefault("diag_cache", {})
    pat_arr = np.stack(patterns, axis=0) if patterns else np.empty((0, N_NEURONS), dtype=float)
    key = (_hash_array(pat_arr), _hash_array(W))
    if key in ss.diag_cache:
        return ss.diag_cache[key]
    diag = _compute_diagnostics(patterns, W)
    ss.diag_cache[key] = diag
    return diag


def _apply_transformations(pattern: np.ndarray, *, noise: float = 0.0, invert: bool = False, hide_half: bool = False, seed: int = 0) -> np.ndarray:
    """Apply simple degradations to a pattern before retrieval."""
    rng = np.random.default_rng(seed)
    x = pattern.astype(float).copy()

    if invert:
        x *= -1

    if hide_half:
        n_hide = x.size // 2
        if n_hide > 0:
            idx_hide = rng.choice(x.size, size=n_hide, replace=False)
            x[idx_hide] = -1  # hide information: set pixels to -1

    if noise > 0.0:
        k = int(round(noise * x.size))
        if k > 0:
            idx_noise = rng.choice(x.size, size=k, replace=False)
            x[idx_noise] *= -1

    return x


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
    sns.heatmap(img, cmap=cmap, cbar=False, square=True,
                norm=norm, vmin=-1, vmax=1, ax=ax)
    if title != "":
        ax.set_title(title, size=8)
    ax.set_xticks([])
    ax.set_yticks([])
    
    st.pyplot(fig, clear_figure=True, width="content")


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


def render_pattern_fig(pattern: np.ndarray, *, title: str = "", fs=2):
    img = pattern.reshape(H, W).astype(np.float32)

    m = max(abs(img.min()), abs(img.max()))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-m, vmax=m)

    fig, ax = plt.subplots(figsize=(fs, fs), dpi=140)
    ax.imshow(
        img,
        cmap=plt.cm.coolwarm,   # alebo vlastná cmap
        norm=norm,
        interpolation="nearest",
    )
    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title, fontsize=8)

    fig.tight_layout(pad=0.2)
    return fig


@st.cache_data
def cached_pattern_fig(pattern_bytes: bytes, title: str, fs: int):
    pattern = np.frombuffer(pattern_bytes, dtype=np.int8)
    return render_pattern_fig(pattern, title=title, fs=fs)


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
        padding: 0.25rem 0.25rem !important;
        color: #ffffff;
    }
    /* Make default buttons (e.g., 'zapamatovat') more compact */
    button[data-testid="baseButton-secondary"] {
        padding: 0.25rem 0.25rem !important;
        min-height: 26px !important;
        font-size: 0.9rem !important;
    }
    /* Extra targeting for memorize buttons */
    button[title="remember-btn"] {
        padding: 0.2rem 0.35rem !important;
        min-height: 22px !important;
        font-size: 0.82rem !important;
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
    ### Training configuration
    st.header("Učící algoritmus")
    # Learning method for Hopfield network
    learning_method = st.selectbox(
        "Metoda",
        ["hebbian", "storkey", "pinv_centered", "pinv_damped"],
        index=["hebbian", "storkey", "pinv_centered", "pinv_damped"].index(
            st.session_state.hop_config["learning_method"] if st.session_state.hop_config else "hebbian"
        ), help="""Způsob, jakým Hopfieldova síť ukládá vzory do svých vah.\n
        - hebbian: Hebbovská pravidla učení.
        - storkey: Storkeyho pravidla učení.
        - pinv_centered: Pseudoinverzní metoda s centrováním vzorů.
        - pinv_damped: Pseudoinverzní metoda s tlumením.""",
    )
    # Damping lambda for pinv_damped method
    damped_lam = st.slider("lambda (pinv_damped)", min_value=0.0, max_value=1.0, value=0.1, step=0.01,
        help="Tlumící faktor používaný v pinv_damped metodě učení."
    )

    ### Retrieval configuration
    st.header("Rekonstrukce")
    # Global bias for retrieval
    theta = st.number_input("Globální bias", value=0.0, step=0.1,
        help="Hodnota přidaná k aktivaci každého neuronu během rekonstrukce.")
    # Maximum number of iterations for retrieval
    max_iter = st.number_input("Maximální počet iterací", min_value=1, max_value=200, value=50, step=1,
        help="Maximální počet kroků během rekonstrukce vzoru."
    )
    # Update rule for retrieval (async/sync)
    update_rule = st.selectbox("Způsob aktualizace", ["async", "sync"], index=0,  
        help="""Způsob, jakým jsou neurony aktualizovány během rekonstrukce.\n
        - async: Neurony jsou aktualizovány jeden po druhém v náhodném pořadí.
        - sync: Všechny neurony jsou aktualizovány současně.
        """
    )
    # Use local biases for retrieval
    use_local_biases = st.toggle("Použi lokální biasy", value=False, 
        help="Použití individuálních biasů pro každý neuron během rekonstrukce."
    )
    
    ### Transformations of samples for retrieval
    st.header("Transformace vzorů")
    # Random seed for sample transformations
    seed_val = st.number_input("Seed", value=42, step=1, key="retrieval_seed", 
        help="Inicilizace generátoru náhodných čísel pro transformace vzorů."
    )
    
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
                if st.button("zapamatovat", key=f"remember_{idx}", use_container_width=True, type="secondary"):
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

# --- Weights of the Hopfield network + diagnostics ---

st.subheader("Učení sítě")
st.markdown(
    """Tady můžeš sledovat váhy Hopfieldovy sítě i rychlou diagnostiku zaplnění a stability."""
)

pos = int(np.clip(st.session_state.pool_pos, 0, len(y_pool) - 1))
st.session_state.pool_pos = pos
label = str(y_pool[pos])
p = X_pool[pos].astype(int)

W_now = st.session_state.hop.W

colW, colDiag = st.columns([1.15, 1], gap="large")

with colW:
    st.markdown("**Matice Vah**")
    st.markdown(f"""
                - :blue[Velikost matice vah:] {W_now.shape[0]:d} x {W_now.shape[1]:d}
                """, unsafe_allow_html=True)

    m = float(np.max(np.abs(W_now))) or 1.0
    _plot_heatmap(W_now, cmap="RdBu_r", vmin=-m, vmax=m)

with colDiag:
    st.markdown("**Diagnostika**")
    diag = _compute_diagnostics_cached(st.session_state.stored_patterns, W_now) if st.session_state.stored_patterns else None
    
    # Formatting helper of the diagnostic values
    def _fmt(x):
        if not st.session_state.stored_patterns:
            return "---"
        return "n/a" if x is None or np.isnan(x) else f"{x:.3f}"
    
    # Estimate color indicators
    # Default: gray circle
    # Green: good, Yellow: warning, Red: bad
    alpha_indicator         = "⚪"
    corr_mean_indicator     = "⚪"
    corr_max_indicator      = "⚪"
    unstable_frac_indicator = "⚪"
    if diag:
        # Color indicator for load
        if _fmt(diag['alpha']) != "n/a":
            if diag['alpha'] < 0.05:
                alpha_indicator = "🟢"
            elif diag['alpha'] < 0.12:
                alpha_indicator = "🟡"
            else:
                alpha_indicator = "🔴"
        # Color indicator for corr_mean
        if _fmt(diag['corr_mean']) != "n/a":    
            if diag['corr_mean'] < 0.1:
                corr_mean_indicator = "🟢"
            elif diag['corr_mean'] < 0.3:
                corr_mean_indicator = " 🟡"
            else:
                corr_mean_indicator = "🔴"    
        # Color indicator for corr_max
        if _fmt(diag['corr_max']) != "n/a":
            if diag['corr_max'] < 0.3:
                corr_max_indicator = "🟢"
            elif diag['corr_max'] < 0.6:
                corr_max_indicator = "🟡"
            else:
                corr_max_indicator = "🔴"
        # Color indicator for unstable_frac
        if _fmt(diag['unstable_frac']) != "n/a":
            if diag['unstable_frac'] < 0.001:
                unstable_frac_indicator = "🟢"
            elif diag['unstable_frac'] < 0.05:
                unstable_frac_indicator = "🟡"
            else:
                unstable_frac_indicator = "🔴"

    P  = len(st.session_state.stored_patterns)
    N  = W_now.shape[0]
    Nw = int(N * (N - 1) / 2)  # number of weights
    st.markdown(
        f"- :blue[počet neuronů]<br>  N = {N:d}\n"
        f"- :blue[počet vah v síti]<br>  N(N-1)/2 = {Nw:d}\n"
        f"- :blue[počet zapamatovaných vzorů]<br>  P = {P:d}\n"
        f"- :blue[zatížení sítě] {alpha_indicator}<br> α = P/N = {_fmt(diag['alpha'] if diag else None)}\n"
        f"- :blue[průměrná párová korelace mezi vzory] {corr_mean_indicator}<br> E(C) = {_fmt(diag['corr_mean'] if diag else None)}\n"
        f"- :blue[maximální párová korelace mezi vzory] {corr_max_indicator}<br> max(C) = {_fmt(diag['corr_max'] if diag else None)}\n"
        f"- :blue[poměr nestabilních bitů] {unstable_frac_indicator}<br/> {_fmt(diag['unstable_frac'] if diag else None)}",
        unsafe_allow_html=True
    )

# --- Retrieval ---

st.divider()
st.subheader("Rekonstrukce vzoru")

if st.session_state.hop.num_memories() == 0:
    st.info("Nejprve nauč Hopfieldovou síť několit vzorů z datasetu.")
else:
    st.markdown("Vyber zapamatovaný vzor, aplikuj transformace a spusti rekonstrukci.")

    labels = st.session_state.stored_labels
    options_idx = list(range(len(labels)))
    default_idx = min(st.session_state.get("retrieval_idx", 0), len(labels) - 1)

    sel_idx = st.selectbox(
        "Vyber zapamatovaný vzor",
        options_idx,
        index=default_idx,
        format_func=lambda i: f"{i}: {labels[i]}",
        key="retrieval_selectbox",
    )
    noise_val = st.slider("Šum p", min_value=0.0, max_value=0.5, value=0.1, step=0.01, key="retrieval_noise")
    invert_val = st.checkbox("Invertovat vstup", value=False, key="retrieval_invert")
    # hide_half_val = st.checkbox("Zakryt polovinu (náhodne)", value=False, help="Náhodne vynuluje polovicu pixelov.")
    hide_half_val = 0  # Ignore hide half option for now
    # seed_val = st.number_input("Inicilizace generátoru náhodních čísel", value=42, step=1, key="retrieval_seed")

    base = np.array(st.session_state.stored_patterns[sel_idx])
    noisy = _apply_transformations(base, noise=noise_val, invert=invert_val, hide_half=hide_half_val, seed=int(seed_val))
    st.session_state.retrieval_idx = int(sel_idx)
    current_params = {
        "sel_idx": int(sel_idx),
        "noise": float(noise_val),
        "invert": bool(invert_val),
        "hide_half": hide_half_val,
        "seed": int(seed_val),
        "theta": float(theta),
        "max_iter": int(max_iter),
        "update_rule": str(update_rule),
        "use_local_biases": bool(use_local_biases),
    }
    prev_params = st.session_state.get("retrieval_params")
    if prev_params is None or prev_params != current_params:
        st.session_state.retrieval_last = None
    st.session_state.retrieval_params = current_params

    run_retrieval = st.button("Spustiť rekonstrukci")

    if run_retrieval:
        out = st.session_state.hop.retrieve(
            noisy,
            theta=float(theta),
            max_iterations=int(max_iter),
            update_rule=str(update_rule),
            use_local_biases=bool(use_local_biases),
        )
        st.session_state.retrieval_last = {
            "label": labels[sel_idx],
            "original": base,
            "input": noisy,
            "output": out,
            "noise": noise_val,
            "invert": invert_val,
            "hide_half": hide_half_val,
        }

    last = st.session_state.get("retrieval_last")

    c1, c2, c3 = st.columns(3)
    with c1:
        _plot_pattern(base, title="Originál", fs=1.5)
    with c2:
        desc_parts = [f"šum p={noise_val:.2f}"]
        if invert_val:
            desc_parts.append("invert")
        if hide_half_val:
            desc_parts.append("half hidden")
        desc = ", ".join(desc_parts)
        _plot_pattern(noisy, title="Vstup", fs=1.5)
    with c3:
        if last:
            _plot_pattern(last["output"], title="Výstup", fs=1.5)
            try:
                best_mem, best_label, best_score = st.session_state.hop.nearest_memory(last["output"], metric="hamming")
                st.caption(f"Nejbližší uložená vzpomínka: {best_label}")
            except Exception:
                pass
