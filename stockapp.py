# robust_unsupervised_vn_app_patched.py
import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================================
# Try vnstock Quote import (3.3+)
# =========================================
try:
    from vnstock import Quote
    vnstock_ok = True
    vn_import_error = None
except Exception as e:
    Quote = None
    vnstock_ok = False
    vn_import_error = str(e)

# try yfinance fallback
try:
    import yfinance as yf
    yf_ok = True
except Exception:
    yf = None
    yf_ok = False

st.set_page_config(layout="wide", page_title="Stock Behavior Explorer — vnstock 3.3 (patched)")
st.title("📊 Stock Behavior Explorer (vnstock 3.3 API — patched)")
st.write("Uses vnstock 3.3 `Quote().history()` with optional yfinance fallback.")

# -------------------------
# Utilities / indicators
# -------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def parse_volume(v):
    """
    Parse a volume value possibly displayed as '5.16M', '1,234', '1.2K', or numeric.
    Returns int or np.nan on failure.
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return int(v)
    try:
        s = str(v).strip()
        s = s.replace(',', '')
        if s == '':
            return np.nan
        last = s[-1].upper()
        if last == 'M':
            return int(float(s[:-1]) * 1_000_000)
        if last == 'K':
            return int(float(s[:-1]) * 1_000)
        # plain number
        return int(float(s))
    except Exception:
        return np.nan

def standardize_columns(df):
    """
    Normalize columns to single-level names, lowercase keys mapped to standard names.
    Returns DataFrame with columns Open, High, Low, Close, Volume (if present).
    """
    df = df.copy()
    # handle MultiIndex columns (e.g., from yfinance in some cases)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # lower-case lookup
    cols_lower = {c.lower(): c for c in df.columns}

    rename = {}
    map_keys = {
        "open": "Open",
        "o": "Open",
        "high": "High",
        "h": "High",
        "low": "Low",
        "l": "Low",
        "close": "Close",
        "c": "Close",
        "volume": "Volume",
        "v": "Volume",
    }

    for k, std in map_keys.items():
        if k in cols_lower:
            rename[cols_lower[k]] = std

    if rename:
        df = df.rename(columns=rename)

    return df

def unify_df(df):
    """
    Convert df to unified OHLCV + indicators. Return (df, error_message or None)
    """
    if df is None:
        return None, "DataFrame is None"
    if len(df) == 0:
        return None, "DataFrame empty"

    # copy to avoid side effects
    df = df.copy()

    # If 'time' column exists (vnstock), use it as index
    time_col = None
    for possible in ("time", "Time", "date", "Date"):
        if possible in df.columns:
            time_col = possible
            break

    if time_col:
        try:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df = df.set_index(time_col)
        except Exception:
            # fallback: try to coerce index
            pass
    else:
        # attempt to ensure index is datetime
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            # leave as-is; later checks will fail
            pass

    # standardize columns
    df = standardize_columns(df)

    # if Volume exists but is not numeric (strings like '5.16M'), parse
    if "Volume" in df.columns:
        if not np.issubdtype(df["Volume"].dtype, np.number):
            df["Volume"] = df["Volume"].apply(parse_volume)

    # ensure required columns
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(set(df.columns)):
        return None, f"Missing required columns. Found: {list(df.columns)}"

    # compute indicators
    try:
        df["Returns"] = df["Close"].pct_change()
        df["Volatility"] = df["Returns"].rolling(7).std()
        df["SMA10"] = df["Close"].rolling(10).mean()
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["RSI"] = compute_rsi(df["Close"])
    except Exception as e:
        return None, f"Indicator computation failed: {e}"

    df = df.dropna()
    if df.empty:
        return None, "All rows dropped after dropna()"

    return df, None

# -------------------------
# Loaders
# -------------------------
def load_vnstock_quote(ticker, start, end, logger=st.write):
    """
    Use vnstock Quote() API. Try several sources if available.
    Returns (df, message) where df is standardized df or None.
    """
    if not vnstock_ok:
        return None, f"vnstock import failed: {vn_import_error}"

    sources = ["VCI", "TCBS", "MSN", "VND"]
    last_err = None
    for src in sources:
        try:
            logger(f"➡️ vnstock Quote(): {ticker} (source={src})")
            q = Quote(symbol=ticker, source=src)
            # history may be named history(...) in the Quote API
            try:
                df = q.history(start=start.strftime("%Y-%m-%d"),
                               end=end.strftime("%Y-%m-%d"),
                               interval="1D")
            except TypeError:
                # fallback if different signature
                df = q.history(start=start.strftime("%Y-%m-%d"),
                               end=end.strftime("%Y-%m-%d"))
            # unify
            std, err = unify_df(df)
            if err:
                last_err = f"source={src} unify error: {err}"
                logger(f"   ⚠ {last_err}")
                continue
            return std, f"Loaded via vnstock Quote() source={src}"
        except Exception as e:
            last_err = f"source={src} exception: {e}"
            logger(f"   ✖ {last_err}")
            continue

    return None, f"vnstock attempts failed: {last_err}"

def load_yfinance(ticker, start, end, logger=st.write):
    if not yf_ok:
        return None, "yfinance not available"
    try:
        logger(f"➡️ yfinance: downloading {ticker}")
        df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
        if df is None or df.empty:
            return None, "yfinance returned empty"
        std, err = unify_df(df)
        if err:
            return None, f"yfinance unify error: {err}"
        return std, "Loaded via yfinance"
    except Exception as e:
        return None, f"yfinance exception: {e}"

# -------------------------
# Feature matrix creator
# -------------------------
def make_feature_matrix(data_dict):
    rows = []
    for t, df in data_dict.items():
        if df is None:
            continue
        try:
            rows.append({
                "Ticker": t,
                "AvgReturn": df["Returns"].mean(),
                "Volatility": df["Volatility"].mean(),
                "RSI": df["RSI"].mean(),
                "SMA10_Slope": (df["SMA10"].iloc[-1] - df["SMA10"].iloc[0]) if len(df["SMA10"])>1 else 0,
                "SMA20_Slope": (df["SMA20"].iloc[-1] - df["SMA20"].iloc[0]) if len(df["SMA20"])>1 else 0,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)

# -------------------------
# UI
# -------------------------
st.info("VN tickers → vnstock Quote(); foreign tickers → yfinance fallback (optional).")
raw = st.text_input("Tickers (comma-separated)", "VNM, HPG, FPT, MWG")
tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]

months = st.slider("Months of history", 3, 36, 12)
end = datetime.datetime.now()
start = end - datetime.timedelta(days=months * 30)

use_yf = st.checkbox("Enable yfinance fallback", True)

if st.button("Load Data"):
    stock_data = {}
    load_messages = {}
    st.subheader("📥 Load logs")

    for t in tickers:
        st.write(f"\n---\n🔍 Fetching {t}")
        df = None
        msg = None

        # try vnstock Quote
        if vnstock_ok:
            df, msg = load_vnstock_quote(t, start, end, logger=st.write)
            if df is None:
                st.write(f"❌ vnstock failed: {msg}")

        # fallback to yfinance if allowed or vnstock not available
        if (df is None) and use_yf:
            df, msg = load_yfinance(t, start, end, logger=st.write)
            if df is None:
                st.error(f"❌ Final fail for {t}: {msg}")
                load_messages[t] = f"Failed: {msg}"
                continue

        if df is None:
            st.error(f"❌ No data for {t} (vnstock not used and yfinance disabled)")
            load_messages[t] = "Failed: no loader"
            continue

        stock_data[t] = df
        load_messages[t] = msg
        st.success(f"✅ Loaded {t}: {msg} ({len(df)} rows)")

    st.write("📌 Summary:", load_messages)

    if len(stock_data) < 3:
        st.error("⚠️ Need at least 3 valid stocks for clustering. Try increasing timeframe, adding tickers, or enabling yfinance.")
        st.stop()

    # make features
    st.subheader("🧮 Feature Matrix")
    features = make_feature_matrix(stock_data)
    if features.empty:
        st.error("Feature matrix is empty — nothing to cluster.")
        st.stop()
    st.dataframe(features)

    # prepare X
    X = features.drop(columns=["Ticker"], errors="ignore")
    X_scaled = StandardScaler().fit_transform(X)

    # clustering slider bounds safe
    max_clusters = max(2, min(10, len(features)))
    st.subheader("🔢 Clustering")
    k = st.slider("Number of clusters (KMeans)", min_value=2, max_value=max_clusters, value=min(3, max_clusters))
    try:
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
        features["Cluster"] = kmeans.fit_predict(X_scaled)
    except Exception as e:
        st.error(f"KMeans fit failed: {e}")
        st.stop()

    # PCA
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)
    features["PC1"] = pcs[:, 0]
    features["PC2"] = pcs[:, 1]

    fig = px.scatter(
        features, x="PC1", y="PC2",
        color=features["Cluster"].astype(str),
        text="Ticker", size="Volatility",
        title="PCA Cluster Visualization"
    )
    st.plotly_chart(fig, use_container_width=True)

    # anomaly detection
    st.subheader("⚠️ Anomaly Detection (IsolationForest)")
    iso = IsolationForest(contamination=0.1, random_state=42)
    try:
        features["Anomaly"] = iso.fit_predict(X_scaled)
        anomalies = features[features["Anomaly"] == -1]
    except Exception as e:
        st.error(f"IsolationForest failed: {e}")
        anomalies = pd.DataFrame()

    st.write("🚨 Detected anomalies (if any):")
    st.dataframe(anomalies)

    # price viewer
    st.subheader("📈 Price Viewer")
    ticker_choice = st.selectbox("Select ticker to inspect", options=features["Ticker"].tolist())
    df_view = stock_data[ticker_choice].copy()

    # add SMA for viewing if missing
    if "SMA10" not in df_view.columns:
        df_view["SMA10"] = df_view["Close"].rolling(10).mean()
    if "SMA20" not in df_view.columns:
        df_view["SMA20"] = df_view["Close"].rolling(20).mean()

    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.7, 0.3], vertical_spacing=0.04,
                         specs=[[{"type": "candlestick"}], [{"type": "bar"}]])
    fig2.add_trace(go.Candlestick(x=df_view.index, open=df_view["Open"], high=df_view["High"],
                                  low=df_view["Low"], close=df_view["Close"], name="Price"), row=1, col=1)
    fig2.add_trace(go.Scatter(x=df_view.index, y=df_view["SMA10"], name="SMA10"), row=1, col=1)
    fig2.add_trace(go.Scatter(x=df_view.index, y=df_view["SMA20"], name="SMA20"), row=1, col=1)
    fig2.add_trace(go.Bar(x=df_view.index, y=df_view["Volume"], name="Volume"), row=2, col=1)

    fig2.update_layout(height=650, title=f"{ticker_choice} price")
    st.plotly_chart(fig2, use_container_width=True)

    st.success("✔ Completed")
