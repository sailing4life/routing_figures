import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import csv

st.set_page_config(layout="wide")
st.title("Expedition Wind Analysis")

# === Sidebar controls ===
st.sidebar.header("Settings")
ws_max = st.sidebar.number_input("Max TWS bin (kt)", min_value=8, max_value=80, value=36, step=4)
ws_step = st.sidebar.number_input("TWS bin step (kt)", min_value=2, max_value=10, value=4, step=1)
dir_step = st.sidebar.number_input("Direction bin step (°)", min_value=5, max_value=45, value=10, step=5)
xtick_step = st.sidebar.selectbox("Angular tick step (°)", options=[30, 45, 60, 90], index=1)
show_bar_labels = st.sidebar.checkbox("Show segment % labels", value=True)
segment_label_floor = st.sidebar.slider("Label segments above (%)", 0, 10, 2)
show_total_labels = st.sidebar.checkbox("Show ring total % labels", value=True)
ring_label_floor = st.sidebar.slider("Label totals above (%)", 0, 20, 6)

st.sidebar.subheader("Time series")
gap_minutes = st.sidebar.slider("Break line on gaps larger than (minutes)", 15, 360, 120, step=15)
label_every = st.sidebar.slider("Annotate every Nth point", 1, 50, 8, step=1)

# === CSV uploader ===
uploaded_file = st.file_uploader("Upload Expedition routing CSV", type="csv")

# --- helpers ---
@st.cache_data(show_spinner=False)
def read_csv_to_df(file_bytes: bytes) -> tuple[pd.DataFrame, list[list[str]]]:
    rows: list[list[str]] = []
    text = file_bytes.decode("utf-8", errors="ignore")
    reader = csv.reader(io.StringIO(text), delimiter=",", quotechar='"')
    for row in reader:
        rows.append(row)
    if not rows:
        return pd.DataFrame(), rows
    header = rows[0]
    data_rows = [row for row in rows[1:] if len(row) == len(header)]
    df = pd.DataFrame(data_rows, columns=header)
    return df, rows


def first_col_containing(df: pd.DataFrame, substrings: list[str]):
    substrings = [s.lower() for s in substrings]
    for col in df.columns:
        low = col.lower()
        if any(s in low for s in substrings):
            return col
    return None


def detect_model(rows: list[list[str]]) -> str:
    # Look for a row that has a cell with 'model' and a value next to it
    for row in rows:
        cells = [c.strip() for c in row if c and c.strip()]
        if not cells:
            continue
        for i, c in enumerate(cells[:-1]):
            if "model" in c.lower():
                return cells[i+1]
    return ""


def parse_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df.get(col, np.nan), errors="coerce")


def format_timerange(ts: pd.Series) -> tuple[str, str]:
    if ts.isna().all():
        return "?", "?"
    start = pd.to_datetime(ts.min(), errors="coerce")
    end = pd.to_datetime(ts.max(), errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return "?", "?"
    return start.strftime("%d-%b-%Y %H:%M"), end.strftime("%d-%b-%Y %H:%M")


# === Polar stack helper reused by both TWD and TWA plots ===

def wind_polar(
    df: pd.DataFrame,
    angle_col: str,
    speed_col: str,
    angle_bins_deg: int,
    ws_max: int,
    ws_step: int,
    zero_at: str,
    theta_direction_ccw: bool,
    title_prefix: str,
    time_range: tuple[str, str],
    model_name: str,
    show_segment_labels: bool,
    seg_floor: float,
    show_total_labels: bool,
    total_floor: float,
    xtick_step_deg: int,
):
    # bins and labels
    dir_bins = np.arange(0, 360 + angle_bins_deg, angle_bins_deg)
    if dir_bins[-1] != 360:
        dir_bins[-1] = 360
    angles = np.deg2rad((dir_bins[:-1] + dir_bins[1:]) / 2)

    tws_bins = np.arange(0, ws_max + ws_step, ws_step)
    tws_labels = [f"{tws_bins[i]}–{tws_bins[i+1]} kt" for i in range(len(tws_bins) - 1)]

    dfx = df.copy()
    if angle_col == "Twa":
        # Normalize TWA to [-180, 180)
        dfx[angle_col] = ((dfx[angle_col] + 180) % 360) - 180
        dir_vals = (dfx[angle_col] + 180) % 360  # for binning
    else:
        dir_vals = dfx[angle_col] % 360

    dfx["dir_bin"] = pd.cut(dir_vals, bins=dir_bins, labels=False, include_lowest=True)
    dfx["tws_bin"] = pd.cut(dfx[speed_col], bins=tws_bins, labels=tws_labels, include_lowest=True)

    counts = dfx.groupby(["dir_bin", "tws_bin"]).size().unstack(fill_value=0)
    percentages = counts.reindex(index=np.arange(len(dir_bins) - 1), fill_value=0)
    total = percentages.values.sum() or 1
    percentages = percentages / total * 100

    total_percent = percentages.sum(axis=1).values
    max_percent = float(np.nanmax(total_percent)) if len(total_percent) else 0.0
    ylim = int(np.ceil((max_percent + 2) / 2.0) * 2) if max_percent > 0 else 2
    rgrid_ticks = list(range(2, ylim + 1, 2))

    colors = [
        "#bde0fe", "#a2d2ff", "#90e0ef", "#48cae4",
        "#00b4d8", "#0096c7", "#0077b6", "#023e8a",
        "#03045e",
    ]

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 7))
    if zero_at.upper().startswith("N"):
        ax.set_theta_zero_location("N")
    elif zero_at.upper().startswith("S"):
        ax.set_theta_zero_location("S")
    elif zero_at.upper().startswith("E"):
        ax.set_theta_zero_location("E")
    else:
        ax.set_theta_zero_location("W")

    ax.set_theta_direction(-1 if theta_direction_ccw else 1)

    width = np.deg2rad(angle_bins_deg)
    bottom = np.zeros(len(angles))
    for i, label in enumerate(tws_labels):
        heights = percentages[label].values if label in percentages.columns else np.zeros(len(angles))
        ax.bar(
            angles,
            heights,
            width=width,
            bottom=bottom,
            color=colors[i % len(colors)],
            edgecolor='black',
            linewidth=0.5,
            label=label,
        )
        bottom += heights

    if show_segment_labels:
        bottom = np.zeros(len(angles))
        for i, label in enumerate(tws_labels):
            heights = percentages[label].values if label in percentages.columns else np.zeros(len(angles))
            for angle, h, b in zip(angles, heights, bottom):
                if h >= seg_floor:
                    ax.text(angle, b + h / 2, f"{int(round(h))}%", ha='center', va='center', fontsize=8)
            bottom += heights

    if show_total_labels:
        for angle, tot in zip(angles, total_percent):
            if tot >= total_floor:
                ax.text(angle, tot + 1, f"{int(round(tot))}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Angular ticks – fewer and larger for readability
    step = int(xtick_step_deg)
    tick_angles = np.arange(0, 360 + step, step)
    if angle_col == "Twa":
        labels = [f"{int(a - 180)}°" for a in tick_angles]
    else:
        labels = [f"{int(a)}°" for a in tick_angles]
    ax.set_xticks(np.deg2rad(tick_angles))
    ax.set_xticklabels(labels)
    ax.tick_params(axis='x', labelsize=10, pad=6)

    ax.set_rgrids(rgrid_ticks, angle=90)
    ax.set_ylim(0, ylim)

    start_time, end_time = time_range
    title_str = f"{title_prefix} (% Time Sailed)\n{start_time} to {end_time}"
    if model_name:
        title_str += f"\n Model: {model_name}"
    ax.set_title(title_str, va='bottom')
    ax.legend(title="TWS", loc="upper right", bbox_to_anchor=(1.25, 1.02))
    fig.tight_layout()
    return fig


if uploaded_file:
    # === Load ===
    df_raw, rows = read_csv_to_df(uploaded_file.getvalue())
    if df_raw.empty:
        st.warning("CSV appears empty after parsing.")
        st.stop()

    # --- Column detection ---
    df = df_raw.copy()

    # Number fields
    if "Twa" not in df.columns:
        twa_col = first_col_containing(df, ["twa"]) or "Twa"
        if twa_col not in df.columns:
            st.error("No TWA column found.")
            st.stop()
        df.rename(columns={twa_col: "Twa"}, inplace=True)

    if "Tws" not in df.columns:
        tws_col = first_col_containing(df, ["tws", "wind speed"]) or "Tws"
        if tws_col not in df.columns:
            st.error("No TWS column found.")
            st.stop()
        df.rename(columns={tws_col: "Tws"}, inplace=True)

    twd_col = first_col_containing(df, ["twd", "true wind dir"])  # Expedition variants
    if twd_col:
        df.rename(columns={twd_col: "Twd°M"}, inplace=True)
    else:
        df["Twd°M"] = np.nan
        st.warning("No TWD column found in the CSV.")

    # Numeric coercion
    df["Twa"] = parse_numeric(df, "Twa")
    df["Tws"] = parse_numeric(df, "Tws")
    df["Twd°M"] = parse_numeric(df, "Twd°M")

    # Time column
    time_col = first_col_containing(df, ["time", "utc", "date"])  # handles e.g. "TimeUTC"
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        start_time, end_time = format_timerange(df[time_col])
    else:
        start_time = end_time = "?"

    # Model
    model_name = detect_model(rows)

    # Drop rows with essential NaNs
    df = df.dropna(subset=["Twa", "Tws"])  # allow Twd°M NaN for TWA plot

    # === Layout ===
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("TWD vs TWS")
        if df["Twd°M"].notna().any():
            fig_twd = wind_polar(
                df.dropna(subset=["Twd°M"]),
                angle_col="Twd°M",
                speed_col="Tws",
                angle_bins_deg=int(dir_step),
                ws_max=int(ws_max),
                ws_step=int(ws_step),
                zero_at="N",
                theta_direction_ccw=True,
                title_prefix="TWD vs TWS",
                time_range=(start_time, end_time),
                model_name=model_name,
                show_segment_labels=show_bar_labels,
                seg_floor=float(segment_label_floor),
                show_total_labels=show_total_labels,
                total_floor=float(ring_label_floor),
                xtick_step_deg=int(xtick_step),
            )
            st.pyplot(fig_twd)
            buf = io.BytesIO()
            fig_twd.savefig(buf, format='png', dpi=200, bbox_inches='tight')
            st.download_button("Download TWD plot (PNG)", data=buf.getvalue(), file_name="twd_vs_tws.png", mime="image/png")
        else:
            st.info("Cannot plot TWD vs TWS without a TWD column.")

    with col2:
        st.subheader("TWA vs TWS")
        fig_twa = wind_polar(
            df,
            angle_col="Twa",
            speed_col="Tws",
            angle_bins_deg=int(dir_step),
            ws_max=int(ws_max),
            ws_step=int(ws_step),
            zero_at="S",
            theta_direction_ccw=True,
            title_prefix="TWA vs TWS",
            time_range=(start_time, end_time),
            model_name=model_name,
            show_segment_labels=show_bar_labels,
            seg_floor=float(segment_label_floor),
            show_total_labels=show_total_labels,
            total_floor=float(ring_label_floor),
            xtick_step_deg=int(xtick_step),
        )
        st.pyplot(fig_twa)
        buf2 = io.BytesIO()
        fig_twa.savefig(buf2, format='png', dpi=200, bbox_inches='tight')
        st.download_button("Download TWA plot (PNG)", data=buf2.getvalue(), file_name="twa_vs_tws.png", mime="image/png")

    # === TWS/TWD Time Series ===
    if time_col:
        st.subheader("TWS/TWD Time Series")
        fig, ax1 = plt.subplots(figsize=(16, 9))

        # sort by time and break on large gaps to avoid diagonal spans
        import matplotlib.dates as mdates

        dft = df.sort_values(time_col).copy()
        # Break lines on gaps
        if time_col and pd.api.types.is_datetime64_any_dtype(dft[time_col]):
            dft["_gap_s"] = dft[time_col].diff().dt.total_seconds()
            gap_sec = int(gap_minutes * 60)
            dft.loc[dft["_gap_s"] > gap_sec, ["Tws", "Twd°M"]] = np.nan

        ax1.plot(dft[time_col], dft["Tws"], label='TWS')
        ax1.set_ylabel("TWS (kt)")

        # Label only every Nth point for readability
        for i, (x, y) in enumerate(zip(dft[time_col], dft["Tws"])):
            if pd.notna(y) and (i % int(label_every) == 0):
                ax1.text(x, y, f"{int(round(y))}", fontsize=7, va='bottom')

        ax2 = ax1.twinx()
        if dft["Twd°M"].notna().any():
            ax2.plot(dft[time_col], dft["Twd°M"], label='TWD')
            ax2.set_ylabel("TWD (°)")
            for i, (x, y) in enumerate(zip(dft[time_col], dft["Twd°M"])):
                if pd.notna(y) and (i % int(label_every) == 0):
                    ax2.text(x, y, f"{int(round(y))}", fontsize=7, va='top')
        else:
            ax2.set_ylabel("TWD (°)")

        # Nicely formatted time axis
        locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        formatter = mdates.ConciseDateFormatter(locator)
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(formatter)
        fig.autofmt_xdate()

        # marks (first column containing 'mark'), only on value changes and spaced apart
        mark_col = first_col_containing(dft, ["mark"])
        if mark_col:
            last_t = None
            min_spacing = pd.Timedelta(minutes=max(15, gap_minutes // 4))
            y_top = ax1.get_ylim()[1]
            change = (dft[mark_col] != dft[mark_col].shift(1)) & dft[mark_col].notna()
            for t, m in zip(dft.loc[change, time_col], dft.loc[change, mark_col]):
                if last_t is None or (t - last_t) > min_spacing:
                    ax1.axvline(t, linestyle='--', alpha=0.4)
                    ax1.text(t, y_top, str(m), rotation=90, va='top', ha='right', fontsize=8)
                    last_t = t

        title_str = f"TWS/TWD Time Series\n{start_time} to {end_time}"
        if model_name:
            title_str += f"\n Model: {model_name}"
        plt.title(title_str)
        fig.tight_layout()
        st.pyplot(fig)
        buf3 = io.BytesIO()
        fig.savefig(buf3, format='png', dpi=200, bbox_inches='tight')
        st.download_button("Download time series (PNG)", data=buf3.getvalue(), file_name="tws_twd_timeseries.png", mime="image/png")

    # === Optional: table of TWA vs TWS bins ===
    st.subheader("TWA × TWS distribution (% of time)")
    # Build the same bins used in plots for consistency
    twa_bins_edges = np.linspace(-180, 180, int(360/dir_step) + 1)
    tws_bins_edges = np.arange(0, ws_max + ws_step, ws_step)
    dftab = df.copy()
    dftab['TWA_bin'] = pd.cut(dftab['Twa'], bins=twa_bins_edges, include_lowest=True)
    dftab['TWS_bin'] = pd.cut(dftab['Tws'], bins=tws_bins_edges, include_lowest=True)
    ct = pd.crosstab(dftab['TWA_bin'], dftab['TWS_bin'])
    pct = ct / ct.values.sum() * 100
    st.dataframe(pct.style.format("{:.1f}"))

else:
    st.info("Upload a CSV file to begin.")
