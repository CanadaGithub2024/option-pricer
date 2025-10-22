# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import math
import datetime as dt
from xml.etree import ElementTree as ET

# -----------------------
# Black–Scholes utilities
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
        "vega": vega,
        "theta_per_year": theta,
        "rho": rho,
    }
    return price, greeks

# ---------------------------------
# Market data
# ---------------------------------
@st.cache_data(ttl=3600)
def fetch_spot_and_hist(ticker: str, years_back: int = 5):
    t = yf.Ticker(ticker)
    hist = t.history(period=f"{years_back}y", auto_adjust=True)
    if hist.empty:
        end = pd.Timestamp.today().normalize()
        start = end - pd.Timedelta(days=int(years_back * 365.25))
        hist = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if hist.empty:
        hist = yf.download(ticker, period="6mo", auto_adjust=True, progress=False)

    spot = hist["Close"].iloc[-1] if not hist.empty and "Close" in hist.columns else None
    if spot is None:
        fi = getattr(t, "fast_info", {})
        spot = fi.get("lastPrice") or fi.get("last_price")
    if spot is None or hist.empty:
        raise RuntimeError(f"No price data for '{ticker}'. Try suffix like '.HK' for non-US stocks.")
    return float(spot), hist

def realized_vol(hist: pd.DataFrame, window_days: int) -> float:
    rets = np.log(hist["Close"]).diff().dropna()
    if len(rets) < max(20, window_days // 3):
        raise RuntimeError("Not enough data for volatility.")
    sigma_daily = rets.tail(window_days).std(ddof=1)
    return sigma_daily * np.sqrt(252.0)

# ---------------------------------------
# Risk-free rate
# ---------------------------------------
TREASURY_URL = "https://home.treasury.gov/sites/default/files/interest-rates/yield.xml"
TENOR_TO_YEARS = {
    "BC_1MONTH": 1/12, "BC_2MONTH": 2/12, "BC_3MONTH": 3/12, "BC_6MONTH": 6/12,
    "BC_1YEAR": 1.0, "BC_2YEAR": 2.0, "BC_3YEAR": 3.0, "BC_5YEAR": 5.0,
    "BC_7YEAR": 7.0, "BC_10YEAR": 10.0, "BC_20YEAR": 20.0, "BC_30YEAR": 30.0,
}

@st.cache_data(ttl=86400)
def fetch_latest_treasury_par_yields():
    r = requests.get(TREASURY_URL, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    root = ET.fromstring(r.content)
    entries = root.findall(".//{http://www.w3.org/2005/Atom}entry") or root.findall(".//entry")
    latest_props = None
    latest_date = None
    ns_d = "{http://schemas.microsoft.com/ado/2007/08/dataservices}"
    for e in entries:
        props = e.find(".//properties") or e.find(f".//{ns_d}properties")
        if not props: continue
        date_text = props.findtext(f"{ns_d}NEW_DATE") or props.findtext("NEW_DATE")
        if not date_text: continue
        d = pd.to_datetime(date_text).date()
        if latest_date is None or d > latest_date:
            latest_date, latest_props = d, props
    if not latest_props:
        raise RuntimeError("Failed to parse Treasury XML.")
    ylds = {}
    for tag in TENOR_TO_YEARS:
        val = latest_props.findtext(f"{ns_d}{tag}") or latest_props.findtext(tag)
        ylds[tag] = float(val)/100.0 if val and val.strip() else None
    return latest_date, ylds

def pick_rate_nearest(T_years: float, yields: dict):
    available = [(tag, TENOR_TO_YEARS[tag]) for tag in TENOR_TO_YEARS if yields.get(tag)]
    if not available:
        return "BC_1YEAR", 0.0
    tenor_tag, _ = min(available, key=lambda kv: abs(kv[1] - T_years))
    return tenor_tag, float(yields[tenor_tag])

def human_tenor(tag: str) -> str:
    return tag.replace("BC_", "").replace("YEAR", "Y").replace("MONTH", "M")

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Option Pricer", layout="centered")
st.title("Black–Scholes Option Pricer")
st.markdown("**Real-time spot, realized vol, UST risk-free rate**")

col1, col2, col3 = st.columns(3)

with col1:
    ticker = st.text_input("Ticker", value="MSFT", help="e.g. AAPL, 0883.HK")
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

            S_base, hist = fetch_spot_and_hist(ticker, years_back)
            vol_map = {"7d": 7, "30d": 30, "3m (~63d)": 63, "6m (~126d)": 126, "1y (252d)": 252, "3y (~756d)": 756}
            sigma = realized_vol(hist, vol_map[vol_window])

            if rate_mode == "Auto (UST)":
                rec_date, ylds = fetch_latest_treasury_par_yields()
                tenor_tag, r = pick_rate_nearest(T_base, ylds)
                rf_msg = f"Risk-free: **{human_tenor(tenor_tag)}** UST ({rec_date}) → {r:.4%}"
            else:
                r = r_manual
                rf_msg = f"Risk-free: Manual → {r:.4%}"

            price_base, greeks_base = bs_price_and_greeks(S_base, strike, r, sigma, T_base, kind.lower())

            st.success("**Base Evaluation**")
            st.write(f"**{ticker.upper()}** | Spot: `{S_base:.4f}` | T: `{T_base:.4f}` years")
            st.write(f"Vol ({vol_window}): **{sigma:.4%}** | {rf_msg}")
            st.metric(f"**{kind} Price**", f"{price_base:.4f}")

            cols = st.columns(5)
            with cols[0]: st.metric("Delta", f"{greeks_base['delta']:.4f}")
            with cols[1]: st.metric("Gamma", f"{greeks_base['gamma']:.4f}")
            with cols[2]: st.metric("Vega", f"{greeks_base['vega']:.4f}")
            with cols[3]: st.metric("Theta/yr", f"{greeks_base['theta_per_year']:.4f}")
            with cols[4]: st.metric("Rho", f"{greeks_base['rho']:.4f}")

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
                    with cols[2]: st.metric("Vega", f"{greeks_what['vega']:.4f}")
                    with cols[3]: st.metric("Theta/yr", f"{greeks_what['theta_per_year']:.4f}")
                    with cols[4]: st.metric("Rho", f"{greeks_what['rho']:.4f}")

        except Exception as e:
            st.error(f"**Error:** {e}")