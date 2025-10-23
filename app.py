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

            # === BASE RESULTS ===
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

            # === AD 2: BELOW RESULTS (ONLY AFTER RESULTS) ===
            st.markdown("---")
            st.markdown("""
            <div style="text-align:center; margin:20px 0;">
              <ins class="adsbygoogle"
                   style="display:block"
                   data-ad-client="ca-pub-5164382331668899"
                   data-ad-slot="9876543210"
                   data-ad-format="auto"
                   data-full-width-responsive="true"></ins>
              <script>
                   (adsbygoogle = window.adsbygoogle || []).push({});
              </script>
            </div>
            """, unsafe_allow_html=True)

            # === WHAT-IF: NOW SAFE INSIDE try BLOCK ===
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