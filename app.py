import numpy as np
import streamlit as st

from hopfield_app.hopfield import HopfieldNetwork  # uprav podľa tvojej štruktúry

N = 8  # 8x8

st.set_page_config(page_title="Hopfield demo", layout="wide")
st.title("Hopfield network – associative memory demo")

# --- helpers ---
def grid_to_state(grid: np.ndarray) -> np.ndarray:
    # grid: 0/1 -> state: -1/+1
    return np.where(grid > 0, 1, -1).astype(int).reshape(-1)

def state_to_grid(state: np.ndarray) -> np.ndarray:
    # state: -1/+1 -> grid: 0/1
    s = state.reshape(N, N)
    return (s > 0).astype(int)

def corrupt_state(state: np.ndarray, flip_prob: float, rng: np.random.Generator) -> np.ndarray:
    flips = rng.random(state.shape) < flip_prob
    out = state.copy()
    out[flips] *= -1
    return out

def draw_grid(grid: np.ndarray, key_prefix: str):
    # klikateľná mriežka pomocou checkboxov
    for r in range(N):
        cols = st.columns(N)
        for c in range(N):
            key = f"{key_prefix}_{r}_{c}"
            val = bool(grid[r, c])
            new_val = cols[c].checkbox(" ", value=val, key=key)
            grid[r, c] = 1 if new_val else 0
    return grid

# --- session state init ---
if "rng_seed" not in st.session_state:
    st.session_state.rng_seed = 0
if "grid" not in st.session_state:
    st.session_state.grid = np.zeros((N, N), dtype=int)
if "stored" not in st.session_state:
    st.session_state.stored = []
if "noisy_state" not in st.session_state:
    st.session_state.noisy_state = None
if "history" not in st.session_state:
    st.session_state.history = None

left, right = st.columns([1, 1])

with left:
    st.subheader("1) Create / edit a pattern (8×8)")
    st.session_state.grid = draw_grid(st.session_state.grid, "edit")

    colA, colB, colC = st.columns(3)
    if colA.button("Clear"):
        st.session_state.grid[:] = 0
    if colB.button("Invert"):
        st.session_state.grid = 1 - st.session_state.grid
    st.session_state.rng_seed = colC.number_input("RNG seed", min_value=0, max_value=10_000, value=st.session_state.rng_seed)

    st.divider()
    st.subheader("2) Store patterns")
    if st.button("Store current pattern"):
        st.session_state.stored.append(grid_to_state(st.session_state.grid))
        st.success(f"Stored patterns: {len(st.session_state.stored)}")

    if st.button("Remove last pattern") and st.session_state.stored:
        st.session_state.stored.pop()
        st.info(f"Stored patterns: {len(st.session_state.stored)}")

    st.write("Stored:", len(st.session_state.stored))

with right:
    st.subheader("3) Corrupt + Recall")
    flip_prob = st.slider("Noise level (flip probability)", 0.0, 0.5, 0.15, 0.01)

    if len(st.session_state.stored) == 0:
        st.warning("Store at least one pattern first.")
    else:
        rng = np.random.default_rng(int(st.session_state.rng_seed))

        # train network from stored patterns
        net = HopfieldNetwork(N * N)
        net.train(np.array(st.session_state.stored))

        current = grid_to_state(st.session_state.grid)

        col1, col2, col3 = st.columns(3)

        if col1.button("Corrupt"):
            st.session_state.noisy_state = corrupt_state(current, flip_prob, rng)
            st.session_state.history = None

        if col2.button("Recall"):
            if st.session_state.noisy_state is None:
                st.session_state.noisy_state = corrupt_state(current, flip_prob, rng)
            # try to use history if available in your implementation
            try:
                recalled, history = net.retrieve_with_history(st.session_state.noisy_state, max_steps=50)
                st.session_state.history = history
            except Exception:
                recalled = net.retrieve(st.session_state.noisy_state, max_steps=50)
                st.session_state.history = None
            st.session_state.grid = state_to_grid(recalled)

        if col3.button("Step (if history)"):
            # optional: if you implement stepping later
            st.info("Implement stepping if you store full history.")

        st.divider()
        c1, c2 = st.columns(2)

        with c1:
            st.caption("Noisy input")
            if st.session_state.noisy_state is None:
                st.write("(none yet)")
            else:
                st.image(state_to_grid(st.session_state.noisy_state).astype(np.uint8) * 255, clamp=True, width=256)

        with c2:
            st.caption("Current state")
            st.image(st.session_state.grid.astype(np.uint8) * 255, clamp=True, width=256)

        if st.session_state.history is not None:
            st.divider()
            st.caption("Energy vs step")
            energies = [net.energy(h) for h in st.session_state.history]
            st.line_chart(energies)
