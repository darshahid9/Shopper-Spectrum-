import streamlit as st
import pandas as pd
import numpy as np
import pickle
import difflib
import plotly.graph_objects as go
import plotly.express as px

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
.main { background-color: #F9F9F7; }
.block-container { padding-top: 1.4rem; }

/* ── Sidebar ── */
div[data-testid='stSidebar'] { background: #1E1E2E !important; }
div[data-testid='stSidebar'] p,
div[data-testid='stSidebar'] span,
div[data-testid='stSidebar'] label,
div[data-testid='stSidebar'] .stCaption { color: #ccc !important; }
div[data-testid='stSidebar'] h2 { color: white !important; }

/* Sidebar nav buttons */
div[data-testid='stSidebar'] .stButton > button {
    width: 100% !important;
    text-align: left !important;
    background: transparent !important;
    border: none !important;
    border-left: 3px solid transparent !important;
    border-radius: 0 8px 8px 0 !important;
    color: #ccc !important;
    font-size: 14px !important;
    padding: 10px 16px !important;
    margin-bottom: 2px !important;
    transition: all 0.15s ease !important;
}
div[data-testid='stSidebar'] .stButton > button:hover {
    background: rgba(255,255,255,0.08) !important;
    color: white !important;
}
div[data-testid='stSidebar'] .stButton > button:focus {
    box-shadow: none !important;
}

/* ── Metric cards ── */
.metric-card {
    background: white; border-radius: 12px;
    padding: 1.1rem 1.3rem; border-left: 4px solid;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07); margin-bottom: 0.5rem;
}
.metric-card .lbl { font-size: 11px; color: #999; font-weight: 600;
    text-transform: uppercase; letter-spacing: .06em; }
.metric-card .val { font-size: 26px; font-weight: 700; margin: 3px 0 2px; }
.metric-card .sub { font-size: 12px; color: #aaa; }

/* ── Segment result card ── */
.seg-card {
    background: white; border-radius: 14px;
    padding: 1.6rem; box-shadow: 0 2px 8px rgba(0,0,0,0.09);
    text-align: center;
}
.seg-emoji { font-size: 56px; line-height: 1.1; }
.seg-name  { font-size: 22px; font-weight: 700; margin: 8px 0 4px; }
.seg-tag   { font-size: 14px; color: #777; margin-bottom: 12px; }
.seg-action {
    background: #F4F4F1; border-radius: 8px;
    padding: 12px 14px; font-size: 13px; color: #444; text-align: left;
}

/* ── Toggle pill buttons (input mode switch) ── */
.toggle-wrap { display: flex; gap: 8px; margin-bottom: 16px; }
.toggle-btn {
    flex: 1; padding: 9px 0; border-radius: 8px; border: 2px solid #ddd;
    background: white; cursor: pointer; font-size: 13px; font-weight: 500;
    color: #555; text-align: center; transition: all 0.15s;
}
.toggle-btn.active {
    background: #4C72B0; border-color: #4C72B0; color: white; font-weight: 600;
}

/* ── Segment pill buttons ── */
.pill-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 16px; }
.pill {
    padding: 8px 18px; border-radius: 999px; border: 2px solid;
    font-size: 13px; font-weight: 500; cursor: pointer;
    transition: all 0.15s; background: white;
}
.pill.active { color: white !important; }

/* ── Recommendation cards ── */
.rec-card {
    background: white; border-radius: 10px;
    padding: 0.85rem 1.1rem; box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    margin-bottom: 8px; display: flex; align-items: center;
    justify-content: space-between;
}
.rec-name  { font-size: 14px; font-weight: 600; color: #222; }
.rec-sub   { font-size: 12px; color: #999; margin-top: 3px; }
.rec-score { font-size: 20px; font-weight: 700; }
.bar-wrap  { background: #eee; border-radius: 6px; height: 5px; margin-top: 7px; }
.bar-fill  { height: 5px; border-radius: 6px; }

/* ── Nav active indicator injected via JS trick (st.button workaround) ── */
.nav-active > button {
    background: rgba(255,255,255,0.13) !important;
    border-left: 3px solid #7C9FD4 !important;
    color: white !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Config ───────────────────────────────────────────────────────────────────
SEGMENT_CONFIG = {
    "Champions": {
        "emoji": "🏆", "color": "#4C72B0",
        "tagline": "Recent, frequent, high-value buyers",
        "action": "Reward with loyalty perks, early access, and VIP offers. "
                  "Protect at all costs — these customers drive a disproportionate share of revenue.",
    },
    "Regulars": {
        "emoji": "🛍️", "color": "#55A868",
        "tagline": "Consistent buyers with growth potential",
        "action": "Upsell with bundles, cross-sell adjacent categories, and increase "
                  "order frequency through targeted promotions.",
    },
    "Hibernating": {
        "emoji": "😴", "color": "#C44E52",
        "tagline": "Lapsed customers — haven't bought in a long time",
        "action": "Launch win-back campaigns with time-limited discounts. "
                  "A 'We miss you' email sequence with a personalised offer works well.",
    },
    "High-Return Risk": {
        "emoji": "⚠️", "color": "#DD8452",
        "tagline": "Active but return a significant portion of purchases",
        "action": "Investigate return reasons, improve product descriptions and photos, "
                  "review sizing guides. Consider return-rate thresholds for wholesale accounts.",
    },
}

# ─── Load models ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    with open("models/kmeans_model.pkl",         "rb") as f: kmeans       = pickle.load(f)
    with open("models/scaler.pkl",               "rb") as f: scaler       = pickle.load(f)
    with open("models/label_map.pkl",            "rb") as f: label_map    = pickle.load(f)
    with open("models/similarity_matrix.pkl",    "rb") as f: sim_matrix   = pickle.load(f)
    with open("models/product_list.pkl",         "rb") as f: product_list = pickle.load(f)
    with open("models/prod_coverage.pkl",        "rb") as f: prod_cov     = pickle.load(f)
    with open("models/segment_descriptions.pkl", "rb") as f: seg_desc     = pickle.load(f)
    rfm = pd.read_csv("data/rfm_clustered.csv")
    return kmeans, scaler, label_map, sim_matrix, product_list, prod_cov, seg_desc, rfm

kmeans, scaler, label_map, sim_matrix, product_list, prod_cov, seg_desc, rfm = load_models()

# ─── Helpers ──────────────────────────────────────────────────────────────────
def predict_segment(recency, frequency, monetary, return_rate):
    row    = pd.DataFrame([[recency, np.log1p(frequency), np.log1p(monetary), return_rate]],
                          columns=["Recency","Frequency","Monetary","ReturnRate"])
    cluster = kmeans.predict(scaler.transform(row))[0]
    return label_map[cluster]

def get_recommendations(product_name, n=5):
    query = product_name.strip().title()
    matched, was_fuzzy = query, False
    if query not in sim_matrix.index:
        hits = difflib.get_close_matches(query, product_list, n=1, cutoff=0.3)
        if not hits:
            hints = difflib.get_close_matches(query, product_list, n=3, cutoff=0.1)
            return None, False, pd.Series(dtype=float), hints
        matched, was_fuzzy = hits[0], True
    recs = sim_matrix[matched].drop(matched).sort_values(ascending=False).head(n)
    return matched, was_fuzzy, recs, []

def score_color(s):
    return "#4C72B0" if s >= 0.5 else ("#55A868" if s >= 0.3 else "#DD8452")

def score_label(s):
    return "Strong match" if s >= 0.5 else ("Good match" if s >= 0.3 else "Weak match")

# ─── Session state defaults ───────────────────────────────────────────────────
if "page"         not in st.session_state: st.session_state.page         = "Overview"
if "input_mode"   not in st.session_state: st.session_state.input_mode   = "type"
if "active_seg"   not in st.session_state: st.session_state.active_seg   = "Champions"

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 Shopper Spectrum")
    st.markdown("<span style='color:#aaa;font-size:13px'>E-Commerce Analytics Suite</span>",
                unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#333;margin:12px 0'>", unsafe_allow_html=True)

    nav_items = [("📊", "Overview"), ("🎯", "Customer Segmentation"), ("💡", "Product Recommendations")]
    for icon, label in nav_items:
        is_active = st.session_state.page == label
        # Wrap in a div to apply the active CSS class
        if is_active:
            st.markdown("<div class='nav-active'>", unsafe_allow_html=True)
        if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.page = label
            st.rerun()
        if is_active:
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#333;margin:12px 0'>", unsafe_allow_html=True)
    st.markdown("<span style='color:#888;font-size:12px;font-weight:600;text-transform:uppercase;"
                "letter-spacing:.06em'>Dataset</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='color:#ccc;font-size:13px'>🧑 <b>{len(rfm):,}</b> customers</span>",
                unsafe_allow_html=True)
    st.markdown(f"<span style='color:#ccc;font-size:13px'>📦 <b>{len(product_list):,}</b> products</span>",
                unsafe_allow_html=True)
    st.markdown("<span style='color:#ccc;font-size:13px'>🌍 <b>38</b> countries</span>",
                unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#333;margin:12px 0'>", unsafe_allow_html=True)
    st.markdown("<span style='color:#555;font-size:11px'>Shopper Spectrum v1.0<br>"
                "Python · scikit-learn · Streamlit</span>", unsafe_allow_html=True)

page = st.session_state.page

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("📊 Shopper Spectrum — Overview")
    st.markdown("UK-based online retailer · Dec 2022 – Dec 2023 · B2B wholesale")
    st.divider()

    total_rev   = rfm["Monetary"].sum()
    avg_spend   = rfm["Monetary"].mean()
    champions   = (rfm["Segment"] == "Champions").sum()

    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val, sub, clr in [
        (c1, "Net Revenue",        f"£{total_rev:,.0f}", "After returns",                    "#4C72B0"),
        (c2, "Total Customers",    f"{len(rfm):,}",       "Customer-linked orders",           "#55A868"),
        (c3, "Avg Customer Spend", f"£{avg_spend:,.0f}",  "Net of returns",                   "#DD8452"),
        (c4, "Champions",          f"{champions:,}",      f"{champions/len(rfm)*100:.1f}% of base", "#9B59B6"),
    ]:
        col.markdown(f"""
        <div class="metric-card" style="border-color:{clr}">
            <div class="lbl">{lbl}</div>
            <div class="val" style="color:{clr}">{val}</div>
            <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    col_l, col_r = st.columns([1, 1.4])

    with col_l:
        st.subheader("Customer Segments")
        sc = rfm["Segment"].value_counts().reset_index()
        sc.columns = ["Segment","Count"]
        sc["Color"]   = sc["Segment"].map(lambda s: SEGMENT_CONFIG.get(s,{}).get("color","#999"))
        sc["Revenue"] = sc["Segment"].map(rfm.groupby("Segment")["Monetary"].sum())
        sc["RevPct"]  = (sc["Revenue"]/total_rev*100).round(1)
        fig1 = go.Figure(go.Bar(
            x=sc["Segment"], y=sc["Count"], marker_color=sc["Color"],
            text=sc["Count"], textposition="outside",
            customdata=sc[["RevPct"]],
            hovertemplate="<b>%{x}</b><br>Customers: %{y:,}<br>Revenue: %{customdata[0]:.1f}%<extra></extra>",
        ))
        fig1.update_layout(height=340, margin=dict(t=10,b=10,l=10,r=10),
                           plot_bgcolor="#F9F9F7", paper_bgcolor="rgba(0,0,0,0)",
                           yaxis_title="Customers", showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

    with col_r:
        st.subheader("RFM Space — Recency vs Monetary")
        rp = rfm[rfm["Monetary"]>0].copy()
        rp["MonLog"] = np.log1p(rp["Monetary"])
        fig2 = px.scatter(rp, x="Recency", y="MonLog", color="Segment",
                          color_discrete_map={s: SEGMENT_CONFIG[s]["color"] for s in SEGMENT_CONFIG},
                          opacity=0.55,
                          hover_data={"Recency":True,"Monetary":":,.0f","Frequency":True,"MonLog":False},
                          labels={"MonLog":"Monetary (log £)","Recency":"Recency (days)"})
        fig2.update_traces(marker=dict(size=5))
        fig2.update_layout(height=340, margin=dict(t=10,b=10,l=10,r=10),
                           plot_bgcolor="#F9F9F7", paper_bgcolor="rgba(0,0,0,0)",
                           legend=dict(title="Segment", orientation="h", y=-0.22))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Segment Revenue Contribution")
    st_tbl = rfm.groupby("Segment").agg(
        Customers=("CustomerID","count"), Revenue=("Monetary","sum"),
        AvgSpend=("Monetary","mean"), AvgRecency=("Recency","mean"),
        AvgFrequency=("Frequency","mean"),
    ).reset_index()
    st_tbl["Revenue %"]   = (st_tbl["Revenue"]   / total_rev  * 100).round(1)
    st_tbl["Customers %"] = (st_tbl["Customers"] / len(rfm)   * 100).round(1)
    st_tbl["Revenue"]     = st_tbl["Revenue"].map("£{:,.0f}".format)
    st_tbl["AvgSpend"]    = st_tbl["AvgSpend"].map("£{:,.0f}".format)
    st_tbl["AvgRecency"]  = st_tbl["AvgRecency"].round(0).astype(int).astype(str) + " days"
    st_tbl["AvgFrequency"]= st_tbl["AvgFrequency"].round(1)
    st_tbl = st_tbl.rename(columns={"AvgSpend":"Avg Spend","AvgRecency":"Avg Recency","AvgFrequency":"Avg Orders"})
    st.dataframe(st_tbl, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CUSTOMER SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Customer Segmentation":
    st.title("🎯 Customer Segmentation")
    st.markdown("Adjust the RFM values below — the segment prediction updates live.")
    st.divider()

    col_form, col_result = st.columns([1, 1.1], gap="large")

    with col_form:
        st.subheader("Customer RFM Inputs")
        recency     = st.slider("📅  Recency — days since last purchase",
                                0, 400, 60, 1, help="Lower = more recent = better")
        frequency   = st.slider("🔁  Frequency — number of unique orders",
                                1, 100, 3, 1, help="More orders = more loyal")
        monetary    = st.number_input("💷  Monetary — net spend (£)",
                                0.0, 300000.0, 800.0, 50.0,
                                help="Total spend minus returns")
        return_rate = st.slider("↩️  Return Rate — fraction of spend returned",
                                0.0, 1.0, 0.0, 0.01, format="%.2f",
                                help="0.0 = no returns · 1.0 = returned everything")
        st.divider()
        st.caption("**How it works:** Inputs are log-transformed (Frequency & Monetary), "
                   "scaled with the training StandardScaler, then passed to the fitted "
                   "KMeans model (k=4). The cluster number maps to a business segment label.")

    with col_result:
        st.subheader("Live Prediction")
        segment = predict_segment(recency, frequency, monetary, return_rate)
        cfg     = SEGMENT_CONFIG[segment]

        st.markdown(f"""
        <div class="seg-card" style="border-top:4px solid {cfg['color']}">
            <div class="seg-emoji">{cfg['emoji']}</div>
            <div class="seg-name" style="color:{cfg['color']}">{segment}</div>
            <div class="seg-tag">{cfg['tagline']}</div>
            <div class="seg-action">💡 <b>Recommended Action</b><br><br>{cfg['action']}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**This customer vs segment average**")
        seg_avg = rfm[rfm["Segment"]==segment][["Recency","Frequency","Monetary","ReturnRate"]].mean()
        cmp = pd.DataFrame({
            "Metric":        ["Recency (days)","Frequency (orders)","Monetary (£)","Return Rate"],
            "This Customer": [recency, frequency, f"£{monetary:,.0f}", f"{return_rate:.2f}"],
            "Segment Avg":   [f"{seg_avg['Recency']:.0f}", f"{seg_avg['Frequency']:.1f}",
                              f"£{seg_avg['Monetary']:,.0f}", f"{seg_avg['ReturnRate']:.2f}"],
        })
        st.dataframe(cmp, use_container_width=True, hide_index=True)
        seg_size = (rfm["Segment"]==segment).sum()
        st.caption(f"This segment contains **{seg_size:,}** customers ({seg_size/len(rfm)*100:.1f}% of base).")

    st.divider()

    # ── Segment explorer with pill buttons ───────────────────────────────────
    st.subheader("Explore All Segments")
    st.markdown("Select a segment to browse its customers:")

    # Render pill buttons via columns (one per segment)
    pill_cols = st.columns(len(SEGMENT_CONFIG))
    for i, (seg, cfg) in enumerate(SEGMENT_CONFIG.items()):
        with pill_cols[i]:
            is_active = st.session_state.active_seg == seg
            btn_label = f"{cfg['emoji']}  {seg}"
            if st.button(btn_label, key=f"pill_{seg}", use_container_width=True,
                         type="primary" if is_active else "secondary"):
                st.session_state.active_seg = seg
                st.rerun()

    # Show selected segment customers
    active = st.session_state.active_seg
    seg_df = rfm[rfm["Segment"]==active][
        ["CustomerID","Recency","Frequency","GrossSpend","ReturnAmount","Monetary","ReturnRate"]
    ].copy()
    seg_df["ReturnRate"]   = seg_df["ReturnRate"].map("{:.2%}".format)
    seg_df["GrossSpend"]   = seg_df["GrossSpend"].map("£{:,.0f}".format)
    seg_df["ReturnAmount"] = seg_df["ReturnAmount"].abs().map("£{:,.0f}".format)
    seg_df["Monetary"]     = seg_df["Monetary"].map("£{:,.0f}".format)

    cfg_active = SEGMENT_CONFIG[active]
    st.markdown(f"""
    <div style="background:white;border-radius:10px;padding:12px 16px;
                border-left:4px solid {cfg_active['color']};
                box-shadow:0 1px 4px rgba(0,0,0,0.07);margin:8px 0 12px">
        <span style="font-size:20px">{cfg_active['emoji']}</span>
        <b style="font-size:15px;margin-left:8px;color:{cfg_active['color']}">{active}</b>
        <span style="font-size:13px;color:#888;margin-left:10px">{cfg_active['tagline']}</span>
    </div>""", unsafe_allow_html=True)

    st.dataframe(seg_df.head(50), use_container_width=True, hide_index=True)
    st.caption(f"Showing top 50 of {len(seg_df):,} customers in this segment.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PRODUCT RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Product Recommendations":
    st.title("💡 Product Recommendations")
    st.markdown("Find products frequently bought together — item-based collaborative filtering.")
    st.divider()

    col_input, col_recs = st.columns([1, 1.2], gap="large")

    with col_input:
        st.subheader("Search")

        # ── Toggle buttons instead of radio ──────────────────────────────────
        st.markdown("**Input method**")
        t1, t2 = st.columns(2)
        with t1:
            if st.button("🔍  Type a name",
                         type="primary" if st.session_state.input_mode=="type" else "secondary",
                         use_container_width=True):
                st.session_state.input_mode = "type"
                st.rerun()
        with t2:
            if st.button("📋  Browse all",
                         type="primary" if st.session_state.input_mode=="browse" else "secondary",
                         use_container_width=True):
                st.session_state.input_mode = "browse"
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        if st.session_state.input_mode == "browse":
            product_input = st.selectbox("Select a product", [""] + sorted(product_list),
                                         label_visibility="collapsed")
        else:
            product_input = st.text_input("Product name",
                                          placeholder="e.g. white hanging heart, cake stand…",
                                          label_visibility="collapsed")

        n_recs = st.slider("Number of recommendations", 3, 10, 5)

        st.divider()
        st.caption("**How it works:** A binary co-purchase matrix "
                   "(4,289 customers × 1,000 products) captures which customers "
                   "bought which products. Cosine similarity finds products that "
                   "share the same buyers.")
        st.caption("Covers **~87%** of all transactions · partial names & typos supported.")

    with col_recs:
        st.subheader("Recommendations")

        if product_input and product_input != "":
            matched, was_fuzzy, recs, hints = get_recommendations(product_input, n=n_recs)

            if matched is None:
                st.error(f"No match found for **\"{product_input}\"**.")
                if hints:
                    st.info(f"Did you mean: **{', '.join(hints[:3])}**?")
            else:
                if was_fuzzy:
                    st.info(f"Closest match: **{matched}**")
                else:
                    st.success(f"Showing results for: **{matched}**")

                if matched in prod_cov.index:
                    st.caption(f"📊 Bought by **{prod_cov.loc[matched]:,}** unique customers.")

                st.markdown("<br>", unsafe_allow_html=True)

                for rank, (prod, score) in enumerate(recs.items(), 1):
                    clr = score_color(score)
                    lbl = score_label(score)
                    nc  = prod_cov.loc[prod] if prod in prod_cov.index else "—"
                    bar = int(score * 100)
                    st.markdown(f"""
                    <div class="rec-card">
                        <div style="flex:1">
                            <div class="rec-name">#{rank} &nbsp; {prod}</div>
                            <div class="rec-sub">👥 {nc:,} customers &nbsp;·&nbsp; {lbl}</div>
                            <div class="bar-wrap">
                                <div class="bar-fill" style="width:{bar}%;background:{clr}"></div>
                            </div>
                        </div>
                        <div style="margin-left:18px;text-align:right">
                            <div class="rec-score" style="color:{clr}">{score:.3f}</div>
                            <div style="font-size:11px;color:#bbb">similarity</div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("#### Similarity Scores")
                fig_r = go.Figure(go.Bar(
                    x=recs.values, y=recs.index, orientation="h",
                    marker_color=[score_color(s) for s in recs.values],
                    text=[f"{s:.3f}" for s in recs.values], textposition="outside",
                    hovertemplate="<b>%{y}</b><br>Similarity: %{x:.3f}<extra></extra>",
                ))
                fig_r.update_layout(
                    height=60 + n_recs*44,
                    margin=dict(t=10,b=10,l=10,r=70),
                    plot_bgcolor="#F9F9F7", paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(title="Cosine Similarity", range=[0,1]),
                    yaxis=dict(autorange="reversed"), showlegend=False,
                )
                st.plotly_chart(fig_r, use_container_width=True)

        else:
            st.info("Enter or select a product on the left to see recommendations.")
            st.markdown("#### 🔥 Most Popular Products")
            top10 = prod_cov.head(10).reset_index()
            top10.columns = ["Product","Unique Customers"]
            st.dataframe(top10, use_container_width=True, hide_index=True)