# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import math
import datetime as dt
from xml.etree import ElementTree as ET

# === ADSENSE: LOAD SCRIPT ===
st.markdown("""
<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-5164382331668899"
     crossorigin="anonymous"></script>
""", unsafe_allow_html=True)

# -----------------------
# Black–Scholes utilities (with scaled Greeks)
# -----------------------
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)

def _d1_d2(S: float, K: float, r: float, sigma: float, T: float):
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2

def bs_price_and_greeks(S: float, K: float, r: float, sigma: float, T: float, kind: str):
    kind = kind.lower()
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        raise ValueError("S, K, sigma, and T must be positive.")
    d1, d2 = _d1_d2(S, K, r, sigma, T)

    if kind == "call":
        price = S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
        delta = _norm_cdf(d1)
        theta = (-S * _norm_pdf(d1) * sigma / (2 * math.sqrt(T))
                 - r * K * math.exp(-r * T) * _norm_cdf(d2))
        rho = K * T * math.exp(-r * T) * _norm_cdf(d2)
    elif kind == "put":
        price = K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
        delta = _norm_cdf(d1) - 1
        theta = (-S * _norm_pdf(d1) * sigma / (2 * math.sqrt(T))
                 + r * K * math.exp(-r * T) * _norm_cdf(-d2))
        rho = -K * T * math.exp(-r * T) * _norm_cdf(-d2)
    else:
        raise ValueError("kind must be 'call' or 'put'")

    gamma = _norm_pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * _norm_pdf(d1) * math.sqrt(T)

    greeks = {
        "delta": delta,
        "gamma": gamma,
        "vega": vega / 100,           # per 1% vol
        "theta_per_day": theta / 365, # per day
        "rho": rho / 100,             # per 100 bps
    }
    return price, greeks

# ---------------------------------
# Market data: Yahoo Finance + Manual Override
# ---------------------------------
@st.cache_data(ttl=3600)
def fetch_spot_and_hist(ticker: str, years_back: int = 5):
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker cannot be empty.")

    hist = yf.Ticker(ticker).history(period=f"{years_back}y", auto_adjust=True)
    if hist.empty:
        end = pd.Timestamp.today().normalize()
        start = end - pd.Timedelta(days=int(years_back * 365.25))
        hist = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if hist.empty:
        hist = yf.download(ticker, period="6mo", auto_adjust=True, progress=False)

    if hist.empty or "Close" not in hist.columns:
        raise RuntimeError(
            f"No price data for '{ticker}'. "
            "Check spelling, add suffix (e.g., '.HK'), or try 'AAPL'. "
            "Wait 1 min if rate-limited."
        )

    spot = float(hist["Close"].iloc[-1])
    return spot, hist

def realized_vol(hist: pd.DataFrame, window_days: int) -> float:
    rets = np.log(hist["Close"]).diff().dropna()
    if len(rets) < max(20, window_days // 3):
        raise RuntimeError("Not enough data for volatility.")
    sigma_daily = rets.tail(window_days).std(ddof=1)
    return sigma_daily * np.sqrt(252.0)

# ---------------------------------------
# Risk-free rate: Dual-mode XML Parser
# ---------------------------------------
TREASURY_URL = "https://home.treasury.gov/sites/default/files/interest-rates/yield.xml"
TENOR_TO_YEARS = {
    "BC_1MONTH": 1/12, "BC_2MONTH": 2/12, "BC_3MONTH": 3/12, "BC_6MONTH": 6/12,
    "BC_1YEAR": 1.0, "BC_2YEAR": 2.0, "BC_3YEAR": 3.0, "BC_5YEAR": 5.0,
    "BC_7YEAR": 7.0, "BC_10YEAR": 10.0, "BC_20YEAR": 20.0, "BC_30YEAR": 30.0,
}

def _parse_as_qr_bc_cm(root):
    new_dates = root.findall(".//G_NEW_DATE")
    if not new_dates:
        return None
    best_node, best_dt = None, None
    for nd in new_dates:
        date_txt = nd.findtext("BID_CURVE_DATE") or nd.findtext(".//BID_CURVE_DATE")
        if not date_txt:
            continue
        d = pd.to_datetime(date_txt, dayfirst=True, errors="coerce")
        if pd.isna(d):
            continue
        if best_dt is None or d > best_dt:
            best_dt, best_node = d, nd
    if best_node is None:
        return None
    bc_cat = best_node.find(".//G_BC_CAT")
    if bc_cat is None:
        return None
    ylds = {}
    for tag in TENOR_TO_YEARS:
        val = bc_cat.findtext(tag)
        ylds[tag] = float(val)/100.0 if val and val.strip() else None
    return best_dt.date(), ylds

def _parse_as_atom(root):
    entries = (root.findall(".//{http://www.w3.org/2005/Atom}entry")
               or root.findall(".//entry"))
    if not entries:
        return None
    ns_m = "{http://schemas.microsoft.com/ado/2007/08/dataservices/metadata}"
    ns_d = "{http://schemas.microsoft.com/ado/2007/08/dataservices}"
    latest_props, latest_date = None, None
    for e in entries:
        props = e.find(f".//{ns_m}properties") or e.find(".//properties")
        if props is None:
            continue
        date_text = (props.findtext(f"{ns_d}NEW_DATE") or props.findtext("NEW_DATE"))
        if not date_text:
            continue
        d = pd.to_datetime(date_text)
        if latest_date is None or d > latest_date:
            latest_date, latest_props = d, props
    if latest_props is None:
        return None
    ylds = {}
    for tag in TENOR_TO_YEARS:
        val = (latest_props.findtext(f"{ns_d}{tag}") or latest_props.findtext(tag))
        clean_val = val.strip() if val else ""
        ylds[tag] = float(clean_val)/100.0 if clean_val else None
    return latest_date.date(), ylds

@st.cache_data(ttl=86400)
def fetch_latest_treasury_par_yields():
    try:
        r = requests.get(TREASURY_URL, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        root = ET.fromstring(r.content)
        parsed = _parse_as_qr_bc_cm(root) or _parse_as_atom(root)
        if not parsed:
            raise RuntimeError("Treasury XML: unrecognized structure")
        return parsed
    except Exception as e:
        st.warning("UST fetch failed. Using default 4.5%.")
        return dt.date.today(), {tag: 0.045 for tag in TENOR_TO_YEARS}

def pick_rate_nearest(T_years: float, yields: dict):
    available = [(tag, TENOR_TO_YEARS[tag]) for tag in TENOR_TO_YEARS if yields.get(tag) is not None]
    if not available:
        return "BC_1YEAR", 0.045
    tenor_tag, _ = min(available, key=lambda kv: abs(kv[1] - T_years))
    return tenor_tag, float(yields[tenor_tag])

def human_tenor(tag: str) -> str:
    return tag.replace("BC_", "").replace("YEAR", "Y").replace("MONTH", "M")

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Option Pricer", layout="centered")
st.title("Black–Scholes Option Pricer")

# === TOP BANNER AD ===
st.markdown("""
<ins class="adsbygoogle"
     style="display:block"
     data-ad-client="ca-pub-5164382331668899"
     data-ad-slot="1234567890"
     data-ad-format="auto"
     data-full-width-responsive="true"></ins>
<script>
     (adsbygoogle = window.adsbygoogle || []).push({});
</script>
""", unsafe_allow_html=True)

st.markdown("**Real-time spot from [Yahoo Finance](https://finance.yahoo.com), realized vol, UST risk-free rate**")

# Disclaimer
st.markdown("""
<div style="background-color:#fff3cd; padding:10px; border-radius:5px; border-left:4px solid #ffc107;">
<strong>Disclaimer:</strong> This tool uses <strong>Yahoo Finance</strong> for price data and <strong>U.S. Treasury</strong> XML feed for rates. 
Data may be delayed, incomplete, or inaccurate. <strong>Not for trading or investment decisions.</strong> Use at your own risk.
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    ticker = st.text_input("Ticker", value="MSFT", help="e.g. MSFT, AAPL, 0883.HK")
    
    # Manual spot price override
    use_manual_spot = st.checkbox("Override Spot Price (Manual)", value=False)
    if use_manual_spot:
        S_manual = st.number_input("Manual Spot Price", min_value=0.01, value=450.0, step=0.5)
    else:
        S_manual = None

    strike = st.number_input("Strike Price", min_value=0.01, value=450.0, step=0.5)
    kind = st.radio("Option Type", ["Call", "Put"], horizontal=True)

with col2:
    maturity_mode = st.radio("Maturity Input", ["By Date", "By Days"], horizontal=True)
    if maturity_mode == "By Date":
        expiry = st.date_input("Expiry Date", value=dt.date.today() + dt.timedelta(days=180))
        days_to_expiry = (expiry - dt.date.today()).days
    else:
        days_to_expiry = st.number_input("Days to Expiry", min_value=1, value=180)
        expiry = dt.date.today() + dt.timedelta(days=days_to_expiry)

    vol_window = st.selectbox("Volatility Window", 
                              ["7d", "30d", "3m (~63d)", "6m (~126d)", "1y (252d)", "3y (~756d)"], 
                              index=1)
    years_back = st.slider("History Years", 1, 10, 5)

with col3:
    rate_mode = st.radio("Risk-Free Rate", ["Auto (UST)", "Manual"], horizontal=True)
    r_manual = 0.045
    if rate_mode == "Manual":
        r_manual = st.number_input("Manual r (annual)", min_value=0.0, value=0.045, step=0.001, format="%.4f")
    
    st.markdown("### What-If Revaluation")
    enable_whatif = st.checkbox("Enable What-If", value=False)
    if enable_whatif:
        eval_date = st.date_input("Evaluation Date", value=dt.date.today() + dt.timedelta(days=5))
        S_override = st.number_input("What-If Spot Price", min_value=0.01, value=450.0, step=0.5)

if st.button("Calculate", type="primary"):
    with st.spinner("Fetching data..."):
        try:
            today = dt.date.today()
            T_base = days_to_expiry / 365.0
            if T_base <= 0:
                st.error("Expiry must be in the future.")
                st.stop()

            # --- Spot Price Logic ---
            if use_manual_spot:
                S_base = S_manual
                spot_source = "Manual input"
                _, hist = fetch_spot_and_hist(ticker, years_back)
            else:
                S_base, hist = fetch_spot_and_hist(ticker, years_back)
                spot_source = f"Yahoo Finance (as of {pd.Timestamp.today().strftime('%Y-%m-%d')})"

            vol_map = {"7d": 7, "30d": 30, "3m (~63d)": 63, "6m (~126d)": 126, "1y (252d)": 252, "3y (~756d)": 756}
            sigma = realized_vol(hist, vol_map[vol_window])

            # Risk-free
            r = 0.045
            rf_msg = "Risk-free: Using default 4.5%"
            if rate_mode == "Auto (UST)":
                try:
                    rec_date, ylds = fetch_latest_treasury_par_yields()
                    tenor_tag, r = pick_rate_nearest(T_base, ylds)
                    rf_msg = f"Risk-free: **{human_tenor(tenor_tag)}** UST ({rec_date}) → {r:.4%}"
                except Exception:
                    st.warning("UST fetch failed. Using default 4.5%.")
            else:
                r = r_manual
                rf_msg = f"Risk-free: Manual → {r:.4%}"

            price_base, greeks_base = bs_price_and_greeks(S_base, strike, r, sigma, T_base, kind.lower())

            st.success("**Base Evaluation**")
            st.write(f"**{ticker.upper()}** | Spot: `{S_base:.4f}` | **Source: {spot_source}**")
            st.write(f"Vol ({vol_window}): **{sigma:.4%}** | {rf_msg}")
            st.metric(f"**{kind} Price**", f"{price_base:.4f}")

            cols = st.columns(5)
            with cols[0]: st.metric("Delta", f"{greeks_base['delta']:.4f}")
            with cols[1]: st.metric("Gamma", f"{greeks_base['gamma']:.4f}")
            with cols[2]: st.metric("Vega (per 1%)", f"{greeks_base['vega']:.4f}")
            with cols[3]: st.metric("Theta (per day)", f"{greeks_base['theta_per_day']:.4f}")
            with cols[4]: st.metric("Rho (per 100 bps)", f"{greeks_base['rho']:.4f}")

            if enable_whatif:
                if eval_date >= expiry:
                    st.error("Eval date must be before expiry.")
                elif eval_date < today:
                    st.error("Eval date cannot be in the past.")
                else:
                    T_what = (expiry - eval_date).days / 365.0
                    price_what, greeks_what = bs_price_and_greeks(S_override, strike, r, sigma, T_what, kind.lower())
                    st.markdown("---")
                    st.success("**What-If Revaluation**")
                    st.write(f"Eval Date: `{eval_date}` | Spot: `{S_override:.4f}` | T: `{T_what:.4f}` years")
                    st.metric(f"**{kind} Price (What-If)**", f"{price_what:.4f}")
                    cols = st.columns(5)
                    with cols[0]: st.metric("Delta", f"{greeks_what['delta']:.4f}")
                    with cols[1]: st.metric("Gamma", f"{greeks_what['gamma']:.4f}")
                    with cols[2]: st.metric("Vega (per 1%)", f"{greeks_what['vega']:.4f}")
                    with cols[3]: st.metric("Theta (per day)", f"{greeks_what['theta_per_day']:.4f}")
                    with cols[4]: st.metric("Rho (per 100 bps)", f"{greeks_what['rho']:.4f}")

        except Exception as e:
            st.error(f"**Error:** {e}")

# === OPTIONAL: SIDEBAR AD ===
with st.sidebar:
    st.markdown("### Sponsored")
    st.markdown("""
    <ins class="adsbygoogle"
         style="display:block"
         data-ad-client="ca-pub-5164382331668899"
         data-ad-slot="1234567891"
         data-ad-format="auto"
         data-full-width-responsive="true"></ins>
    <script>
         (adsbygoogle = window.adsbygoogle || []).push({});
    </script>
    """, unsafe_allow_html=True)