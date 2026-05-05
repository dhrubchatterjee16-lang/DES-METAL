import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json, os

st.set_page_config(
    page_title="DES Lithium Recovery Predictor",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container{padding-top:1.5rem}
    .badge-ok  {background:#d4edda;color:#155724;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600}
    .badge-mid {background:#fff3cd;color:#856404;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600}
    .badge-low {background:#f8d7da;color:#721c24;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600}
    .li-card   {background:linear-gradient(135deg,#e3f2fd,#f1f8e9);border:1px solid #90caf9;
                border-radius:12px;padding:16px 20px;margin-bottom:10px}
    .tip-box   {background:#f1f3f5;border-left:3px solid #4c6ef5;border-radius:6px;
                padding:12px 16px;font-size:13px;line-height:1.8;margin-top:8px}
    .section-header{font-size:12px;font-weight:700;color:#6c757d;letter-spacing:.07em;
                    text-transform:uppercase;margin-bottom:6px}
    div[data-testid="stMetric"]{background:#f8f9fa;border-radius:10px;padding:10px 14px}
    .stTabs [data-baseweb="tab"]{font-size:13px;padding:8px 16px}
    .model-badge{background:#e8f5e9;color:#2e7d32;padding:5px 12px;border-radius:8px;
                 font-size:12px;font-weight:600;display:inline-block;margin-bottom:10px}
    .warn-badge {background:#fff8e1;color:#f57f17;padding:5px 12px;border-radius:8px;
                 font-size:12px;font-weight:600;display:inline-block;margin-bottom:10px;margin-left:8px}
</style>
""", unsafe_allow_html=True)

# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Loading Li-specific XGBoost models...")
def load_models():
    from model_data import load_all_models, ENCODERS, MODEL_RESULTS
    models, le_hba, le_hbd = load_all_models()
    return models, le_hba, le_hbd, ENCODERS, MODEL_RESULTS

models, le_hba, le_hbd, ENCODERS, MODEL_RESULTS = load_models()

LI_HBAS = set(ENCODERS['li_hbas'])
LI_HBDS = set(ENCODERS['li_hbds'])
ALL_HBAS = sorted(le_hba.classes_.tolist())
ALL_HBDS = sorted(le_hbd.classes_.tolist())

# HBAs/HBDs known best for Li from LIB cathodes (literature-curated)
BEST_LI_HBAS = [h for h in ['choline chloride','lactic acid','citric acid',
    'betaine','nicotinamide','glycine','dl-tartaric acid','malic acid',
    'guanidine hydrochloride','acetylcholine chloride'] if h in le_hba.classes_]
BEST_LI_HBDS = [h for h in ['ethylene glycol','oxalic acid','levulinic acid',
    'glycerol','lactic acid','malonic acid','glutaric acid','dl-malic acid',
    'citric acid','dl-tartaric acid','urea','phenol','acetic acid'] if h in le_hbd.classes_]

# Li-specific source materials
LI_SOURCES = {
    "LIB cathode (NMC/NCA)":    {"bench_li": 95, "bench_co": 92, "bench_ni": 90, "bench_mn": 88},
    "LIB cathode (LFP)":        {"bench_li": 97, "bench_co":  0, "bench_ni":  0, "bench_mn":  0},
    "LIB cathode (LCO)":        {"bench_li": 98, "bench_co": 95, "bench_ni":  0, "bench_mn":  0},
    "LIB cathode (LMO)":        {"bench_li": 93, "bench_co":  0, "bench_ni":  0, "bench_mn": 91},
    "Mixed battery black mass":  {"bench_li": 88, "bench_co": 82, "bench_ni": 80, "bench_mn": 78},
    "Li-ion pouch cell":         {"bench_li": 91, "bench_co": 85, "bench_ni": 83, "bench_mn": 81},
    "EV battery module":         {"bench_li": 86, "bench_co": 80, "bench_ni": 78, "bench_mn": 75},
}

OX_BOOST  = {"None":0, "H₂O₂":9, "Citric acid":6, "Ascorbic acid":8, "FeSO₄":5}
ASSIST_B  = {"Conventional":0, "Microwave":8, "Ultrasound":6, "Microwave + Ultrasound":12}
SOLID_LIQ = {"1:10":1.0, "1:20":1.2, "1:50":1.5, "1:100":1.8}
PRETREAT  = {"None":0, "Calcination (500°C)":5, "Mechanical crushing":3,
             "Calcination + crushing":8, "Sieving (<75μm)":4}
LEACH_TIME= {"30 min":0.85, "60 min":1.0, "90 min":1.1, "120 min":1.15, "180 min":1.18}

REGEN_METHODS = {
    "Vacuum evaporation":          {"li_recovery":0.60, "cycles":6,  "energy":"medium"},
    "HBD replenishment":           {"li_recovery":0.70, "cycles":5,  "energy":"low"},
    "Evaporation + replenishment": {"li_recovery":0.85, "cycles":8,  "energy":"high"},
    "Anti-solvent precipitation":  {"li_recovery":0.88, "cycles":7,  "energy":"medium"},
    "Membrane separation":         {"li_recovery":0.80, "cycles":10, "energy":"high"},
    "No regeneration":             {"li_recovery":0.00, "cycles":1,  "energy":"none"},
}

LI_DES_SYSTEMS = [
    dict(name="ChCl : Oxalic acid (1:1)",    li_eff=97, cost=0.72, green=0.82, visc=45),
    dict(name="ChCl : Lactic acid (1:2)",    li_eff=95, cost=0.80, green=0.85, visc=38),
    dict(name="ChCl : Levulinic acid (1:2)", li_eff=98, cost=0.65, green=0.78, visc=112),
    dict(name="ChCl : Malonic acid (1:1)",   li_eff=96, cost=0.70, green=0.80, visc=89),
    dict(name="ChCl : Citric acid (1:1)",    li_eff=93, cost=0.68, green=0.83, visc=850),
    dict(name="Lactic acid : Betaine (3:1)", li_eff=94, cost=0.75, green=0.88, visc=386),
    dict(name="ChCl : Ethylene glycol (1:2)",li_eff=88, cost=0.90, green=0.90, visc=14),
    dict(name="ChCl : Glycerol (1:2)",       li_eff=86, cost=0.92, green=0.92, visc=113),
    dict(name="ChCl : Tartaric acid (1:1)",  li_eff=95, cost=0.71, green=0.81, visc=76),
    dict(name="ChCl : Glutaric acid (1:1)",  li_eff=94, cost=0.73, green=0.79, visc=65),
    dict(name="Betaine : Citric acid (1:1)", li_eff=92, cost=0.69, green=0.84, visc=920),
    dict(name="Nicotinamide : Oxalic (1:1)", li_eff=96, cost=0.60, green=0.76, visc=55),
]

# ── Prediction functions ───────────────────────────────────────────────────────
def predict_all(hba, hbd, ratio, temp_K, water):
    try:
        hba_enc    = le_hba.transform([hba])[0]
        hbd_enc    = le_hbd.transform([hbd])[0]
        is_li_hba  = int(hba in LI_HBAS)
        is_li_hbd  = int(hbd in LI_HBDS)
    except ValueError as e:
        return None, str(e)
    X = np.array([[hba_enc, hbd_enc, ratio, temp_K, water, is_li_hba, is_li_hbd]])
    preds = {}
    for target, (model, log_t) in models.items():
        val = float(model.predict(X)[0])
        preds[target] = max(0, round(np.expm1(val) if log_t else val, 4))
    return preds, None

def predict_li_recovery(preds, temp_K, source, oxidant, assist, sl_ratio, pretreat, leach_time):
    """Combine XGBoost-predicted properties with process parameters for Li recovery %."""
    src = LI_SOURCES[source]
    base = src["bench_li"]

    visc   = preds['viscosity_cP']
    ph     = preds['pH_acidity']
    redox  = preds['reduction_potential_V_vs_SHE']
    li_sc  = preds.get('li_extraction_score', 60)

    # Viscosity penalty: >200 cP reduces mass transfer
    visc_factor = max(0.70, 1 - max(0, visc - 50) / 3000)
    # pH bonus: pH < 3 dissolves Li oxides faster
    ph_factor   = 1.0 + max(0, (3 - ph) * 0.03) if ph < 5 else max(0.85, 1 - (ph-5)*0.04)
    # Redox: more negative helps reduce Co3+→Co2+ releasing Li
    redox_factor = 1.0 + abs(min(0, redox)) * 0.03
    # Temperature: optimal 60-90°C
    temp_C = temp_K - 273.15
    temp_factor = 0.82 + min(0.18, (temp_C - 20) / 400)
    # Process boosts
    ox_boost  = OX_BOOST[oxidant] / 100
    as_boost  = ASSIST_B[assist]  / 100
    sl_factor = SOLID_LIQ[sl_ratio]
    pt_boost  = PRETREAT[pretreat] / 100
    lt_factor = LEACH_TIME[leach_time]

    # Li extraction score from dedicated model (0-100)
    li_score_weight = li_sc / 100

    eff = (base * visc_factor * ph_factor * redox_factor * temp_factor * sl_factor * lt_factor
           * (1 + ox_boost) * (1 + as_boost) * (1 + pt_boost))
    # Blend with dedicated Li model score
    eff = 0.65 * eff + 0.35 * (li_score_weight * base)
    return min(99.5, max(20, round(eff, 1)))

def sweep_temps(hba, hbd, ratio, water):
    temps = np.arange(278.15, 378.15, 5)
    results = {'temps_C': (temps - 273.15).tolist()}
    try:
        hba_enc   = le_hba.transform([hba])[0]
        hbd_enc   = le_hbd.transform([hbd])[0]
        is_li_hba = int(hba in LI_HBAS)
        is_li_hbd = int(hbd in LI_HBDS)
    except ValueError:
        return None
    X = np.array([[hba_enc, hbd_enc, ratio, t, water, is_li_hba, is_li_hbd] for t in temps])
    for prop in ['viscosity_cP', 'pH_acidity', 'density_kg_m3', 'li_extraction_score']:
        model, log_t = models[prop]
        vals = model.predict(X)
        results[prop] = [max(0, float(np.expm1(v) if log_t else v)) for v in vals]
    return results

def mc_uncertainty(hba, hbd, ratio, temp_K, water, n=80):
    samples = {'viscosity_cP': [], 'li_extraction_score': []}
    for _ in range(n):
        p, _ = predict_all(hba, hbd,
            ratio  * (1 + np.random.normal(0, 0.03)),
            temp_K + np.random.normal(0, 2),
            float(np.clip(water + np.random.normal(0, 0.01), 0, 1)))
        if p:
            samples['viscosity_cP'].append(p['viscosity_cP'])
            samples['li_extraction_score'].append(p.get('li_extraction_score', 0))
    return samples

def get_feature_importance():
    names = ['HBA identity','HBD identity','Molar ratio','Temperature','Water content',
             'Is Li-HBA','Is Li-HBD']
    fi = {}
    for target, (model, _) in models.items():
        fi[target] = dict(zip(names, model.feature_importances_.tolist()))
    return fi

def is_pareto(systems):
    flags = []
    for i, s in enumerate(systems):
        dominated = any(
            j!=i and o["li_eff"]>=s["li_eff"] and o["cost"]<=s["cost"]
            and o["green"]>=s["green"]
            and (o["li_eff"]>s["li_eff"] or o["cost"]<s["cost"] or o["green"]>s["green"])
            for j,o in enumerate(systems))
        flags.append(not dominated)
    return flags

def build_li_tips(preds, temp_K, source, oxidant, assist, pretreat):
    tips = []
    visc  = preds['viscosity_cP']
    ph    = preds['pH_acidity']
    redox = preds['reduction_potential_V_vs_SHE']
    li_sc = preds.get('li_extraction_score', 0)
    temp_C = temp_K - 273.15

    if visc > 200:
        tips.append(f"🔴 **High viscosity ({visc:.0f} cP)** — add 10–20% water or raise temperature to improve Li⁺ ion mobility and leaching kinetics.")
    elif visc < 30:
        tips.append(f"🟢 **Excellent viscosity ({visc:.0f} cP)** — good Li⁺ mass transfer expected.")
    if ph > 4:
        tips.append(f"🔴 **pH {ph:.1f} too high** — switch to a more acidic HBD (oxalic acid, malonic acid) to dissolve Li₂CO₃ and LiCoO₂ effectively.")
    elif ph < 2:
        tips.append(f"🟢 **Highly acidic (pH {ph:.1f})** — excellent for dissolving Li oxide phases.")
    if redox > -0.8:
        tips.append(f"⚠️ **Reduction potential weak ({redox:.2f} V)** — consider adding a reductant (H₂O₂, ascorbic acid) to reduce Co³⁺ → Co²⁺ and release bound Li⁺.")
    if temp_C < 50:
        tips.append(f"⚠️ **Temperature {temp_C:.0f}°C is low** — raise to 60–90°C for optimal Li leaching rate from cathode materials.")
    if oxidant == "None" and "NMC" in source or "NCA" in source or "LCO" in source:
        tips.append("⚠️ **Add reductant** — H₂O₂ or ascorbic acid significantly boosts Li recovery from NMC/NCA/LCO by reducing transition metals.")
    if pretreat == "None":
        tips.append("💡 **Calcination pre-treatment** at 500°C removes organic binder (PVDF) and improves cathode dissolution by 5–8%.")
    if li_sc < 50:
        tips.append(f"🔴 **Li suitability score low ({li_sc:.0f}/100)** — consider ChCl:Oxalic acid or ChCl:Levulinic acid which score >75.")
    if not tips:
        tips.append("🟢 **Optimal conditions** — this DES is well-suited for Li extraction from LIB cathodes.")
    return tips[:4]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔋 Li Recovery Predictor")
    st.markdown("XGBoost models trained on **5,790 DES data points** with Li-specific feature engineering.")
    st.divider()

    filter_li = st.toggle("🔋 Show Li-optimised compounds only", value=True)
    hba_opts = BEST_LI_HBAS if filter_li else ALL_HBAS
    hbd_opts = BEST_LI_HBDS if filter_li else ALL_HBDS

    hba = st.selectbox("HBA (hydrogen bond acceptor)",
                        hba_opts,
                        index=hba_opts.index('choline chloride') if 'choline chloride' in hba_opts else 0)
    hbd = st.selectbox("HBD (hydrogen bond donor)",
                        hbd_opts,
                        index=hbd_opts.index('oxalic acid') if 'oxalic acid' in hbd_opts else 0)
    ratio = st.slider("Molar ratio HBA:HBD", 0.05, 10.0, 1.0, 0.05)
    water = st.slider("Water content (mol fraction)", 0.0, 0.50, 0.0, 0.01)

    st.markdown("### Process Conditions")
    temp_C_val = st.slider("Temperature (°C)", 20, 120, 70, 5)
    temp_K_val = temp_C_val + 273.15

    st.markdown("### Li Recovery Parameters")
    source   = st.selectbox("Battery source material", list(LI_SOURCES.keys()))
    oxidant  = st.selectbox("Reductant/oxidant additive", list(OX_BOOST.keys()))
    assist   = st.selectbox("Assist method", list(ASSIST_B.keys()))
    sl_ratio = st.selectbox("Solid:Liquid ratio", list(SOLID_LIQ.keys()), index=1)
    pretreat = st.selectbox("Pre-treatment", list(PRETREAT.keys()))
    leach_t  = st.selectbox("Leaching time", list(LEACH_TIME.keys()), index=1)

# ── Run predictions ────────────────────────────────────────────────────────────
preds, err = predict_all(hba, hbd, ratio, temp_K_val, water)
if err:
    st.error(f"Prediction error: {err}"); st.stop()

visc_pred  = preds['viscosity_cP']
dens_pred  = preds['density_kg_m3']
ph_pred    = preds['pH_acidity']
redox_pred = preds['reduction_potential_V_vs_SHE']
li_score   = preds.get('li_extraction_score', 0)
li_rec     = predict_li_recovery(preds, temp_K_val, source, oxidant, assist,
                                  sl_ratio, pretreat, leach_t)
bench_li   = LI_SOURCES[source]["bench_li"]
tips       = build_li_tips(preds, temp_K_val, source, oxidant, assist, pretreat)
fi         = get_feature_importance()
mc         = mc_uncertainty(hba, hbd, ratio, temp_K_val, water)

grade_css  = "badge-ok" if li_rec>=90 else ("badge-mid" if li_rec>=75 else "badge-low")
grade      = "✅ Excellent" if li_rec>=90 else ("⚠️ Acceptable" if li_rec>=75 else "❌ Improve conditions")
li_flag    = "🔋 Li-optimised" if (hba in LI_HBAS and hbd in LI_HBDS) else "⚠️ Not Li-specific"
li_flag_css = "model-badge" if (hba in LI_HBAS and hbd in LI_HBDS) else "warn-badge"

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🔋 DES Lithium Recovery Predictor")
st.markdown("**5 XGBoost models** · Viscosity · Density · pH · Redox potential · **Li extraction score** · Trained on 5,790 real data points")
col_b1, col_b2 = st.columns([1,5])
with col_b1:
    st.markdown(f'<span class="model-badge">🤖 XGBoost v2 — Li-specific</span>', unsafe_allow_html=True)
with col_b2:
    st.markdown(f'<span class="{li_flag_css}">{li_flag}</span>', unsafe_allow_html=True)

tabs = st.tabs(["🔋 Li Recovery","📊 DES Properties","🔍 Feature Importance",
                "🌡️ Property Sweep","♻️ Reuse Simulator","📈 DES Ranker","ℹ️ Model Info"])

# ════════ TAB 1 — LI RECOVERY (main tab) ════════
with tabs[0]:
    # Top metrics row
    delta_col = "green" if li_rec >= bench_li else "red"
    delta_sym  = "▲" if li_rec >= bench_li else "▼"
    visc_col   = "#37b24d" if visc_pred < 100 else ("#f76707" if visc_pred < 300 else "#f03e3e")
    ph_col     = "#37b24d" if ph_pred < 3 else ("#f76707" if ph_pred < 5 else "#f03e3e")
    redox_col  = "#37b24d" if redox_pred < -1.2 else ("#f76707" if redox_pred < -0.8 else "#f03e3e")
    li_sc_col  = "#37b24d" if li_score >= 70 else ("#f76707" if li_score >= 50 else "#f03e3e")

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:16px">
      <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:14px 16px">
        <div style="font-size:11px;color:#94a3b8;font-weight:600;letter-spacing:.05em;margin-bottom:6px">LI RECOVERY</div>
        <div style="font-size:28px;font-weight:700;color:#f1f5f9">{li_rec:.1f}<span style="font-size:16px">%</span></div>
        <div style="font-size:12px;color:{delta_col};margin-top:4px">{delta_sym} {abs(li_rec-bench_li):.1f}% vs benchmark ({bench_li}%)</div>
      </div>
      <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:14px 16px">
        <div style="font-size:11px;color:#94a3b8;font-weight:600;letter-spacing:.05em;margin-bottom:6px">LI SUIT. SCORE</div>
        <div style="font-size:28px;font-weight:700;color:{li_sc_col}">{li_score:.1f}<span style="font-size:14px;color:#64748b">/100</span></div>
        <div style="font-size:12px;color:#64748b;margin-top:4px">XGBoost Li model</div>
      </div>
      <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:14px 16px">
        <div style="font-size:11px;color:#94a3b8;font-weight:600;letter-spacing:.05em;margin-bottom:6px">VISCOSITY</div>
        <div style="font-size:28px;font-weight:700;color:{visc_col}">{visc_pred:.0f}<span style="font-size:14px"> cP</span></div>
        <div style="font-size:12px;color:#64748b;margin-top:4px">{"Good" if visc_pred<100 else ("Moderate" if visc_pred<300 else "High")}</div>
      </div>
      <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:14px 16px">
        <div style="font-size:11px;color:#94a3b8;font-weight:600;letter-spacing:.05em;margin-bottom:6px">pH</div>
        <div style="font-size:28px;font-weight:700;color:{ph_col}">{ph_pred:.2f}</div>
        <div style="font-size:12px;color:#64748b;margin-top:4px">{"Acidic (ideal)" if ph_pred<3 else ("Moderate" if ph_pred<6 else "Basic")}</div>
      </div>
      <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:14px 16px">
        <div style="font-size:11px;color:#94a3b8;font-weight:600;letter-spacing:.05em;margin-bottom:6px">REDOX POTENTIAL</div>
        <div style="font-size:28px;font-weight:700;color:{redox_col}">{redox_pred:.3f}<span style="font-size:14px"> V</span></div>
        <div style="font-size:12px;color:#64748b;margin-top:4px">vs SHE</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f'<span class="{grade_css}">{grade}</span>', unsafe_allow_html=True)
    st.markdown(f"**Benchmark for {source}:** {bench_li}%")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        # Li recovery gauge
        st.markdown('<div class="section-header">Li recovery gauge</div>', unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=li_rec,
            delta={'reference': bench_li, 'valueformat': '.1f', 'suffix': '%'},
            title={'text': f"Li Recovery vs Benchmark ({bench_li}%)", 'font': {'size': 13}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "#4c6ef5"},
                'steps': [
                    {'range': [0,  60], 'color': "#fee2e2"},
                    {'range': [60, 75], 'color': "#fef9c3"},
                    {'range': [75,100], 'color': "#dcfce7"},
                ],
                'threshold': {'line': {'color': "#ef4444", 'width': 3},
                              'thickness': 0.8, 'value': bench_li}
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20,r=20,t=40,b=10),
                                 paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Li suitability breakdown
        st.markdown('<div class="section-header">Li suitability components</div>', unsafe_allow_html=True)
        visc_s  = max(0, min(100, (1 - max(0, visc_pred-50)/3000)*100))
        ph_s    = max(0, min(100, (1 - max(0, ph_pred-1)/7)*100))
        redox_s = max(0, min(100, abs(min(0, redox_pred))/2*100))
        comp_df = pd.DataFrame({
            'Component': ['Viscosity score', 'pH score', 'Redox score'],
            'Value': [round(visc_s,1), round(ph_s,1), round(redox_s,1)],
            'Weight': ['35%', '40%', '25%'],
            'Status': [
                '✅' if visc_s > 70 else ('⚠️' if visc_s > 40 else '❌'),
                '✅' if ph_s   > 70 else ('⚠️' if ph_s   > 40 else '❌'),
                '✅' if redox_s> 50 else ('⚠️' if redox_s> 25 else '❌'),
            ]
        })
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

    with col2:
        # Optimisation tips
        st.markdown('<div class="section-header">Li-specific optimisation advice</div>', unsafe_allow_html=True)
        tip_html = "".join(f"<p style='margin-bottom:8px'>{t}</p>" for t in tips)
        st.markdown(f'<div class="tip-box">{tip_html}</div>', unsafe_allow_html=True)

        # Co/Ni/Mn co-recovery
        st.markdown('<div class="section-header">Co-metal recovery estimates</div>', unsafe_allow_html=True)
        src_data = LI_SOURCES[source]
        metals = {'Li': li_rec, 'Co': src_data['bench_co'], 'Ni': src_data['bench_ni'], 'Mn': src_data['bench_mn']}
        metals = {k: v for k,v in metals.items() if v > 0}
        fig_metals = go.Figure(go.Bar(
            x=list(metals.keys()), y=list(metals.values()),
            marker_color=['#4c6ef5','#37b24d','#f76707','#7950f2'][:len(metals)],
            text=[f"{v:.0f}%" for v in metals.values()], textposition='outside'
        ))
        fig_metals.update_layout(height=220, margin=dict(l=0,r=0,t=10,b=10),
                                  yaxis=dict(range=[0,110], ticksuffix="%"),
                                  xaxis_title="Metal", yaxis_title="Est. Recovery %",
                                  plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_metals, use_container_width=True)

    st.divider()

    # Sensitivity analysis for Li recovery
    st.markdown('<div class="section-header">Sensitivity — impact on Li recovery (%)</div>', unsafe_allow_html=True)

    def li_rec_for(hba2=hba, hbd2=hbd, ratio2=ratio, temp2=temp_K_val,
                   water2=water, ox2=oxidant, as2=assist, sl2=sl_ratio,
                   pt2=pretreat, lt2=leach_t):
        p2, _ = predict_all(hba2, hbd2, ratio2, temp2, water2)
        if p2 is None: return li_rec
        return predict_li_recovery(p2, temp2, source, ox2, as2, sl2, pt2, lt2)

    sens = {
        "Temp +20°C":           li_rec_for(temp2=temp_K_val+20) - li_rec,
        "Water +0.1 mol frac":  li_rec_for(water2=min(0.5, water+0.1)) - li_rec,
        "Add H₂O₂ reductant":   li_rec_for(ox2="H₂O₂") - li_rec,
        "Add microwave assist":  li_rec_for(as2="Microwave") - li_rec,
        "Calcination pretreat":  li_rec_for(pt2="Calcination (500°C)") - li_rec,
        "Extend to 120 min":     li_rec_for(lt2="120 min") - li_rec,
        "Ratio ×2":              li_rec_for(ratio2=ratio*2) - li_rec,
    }
    fig_sens = go.Figure(go.Bar(
        x=list(sens.values()), y=list(sens.keys()), orientation="h",
        marker_color=["#37b24d" if v>=0 else "#f03e3e" for v in sens.values()],
        text=[f"{v:+.1f}%" for v in sens.values()], textposition="outside"
    ))
    fig_sens.update_layout(height=260, margin=dict(l=0,r=80,t=10,b=10),
                            xaxis_title="Δ Li recovery (%)",
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_sens, use_container_width=True)

    # MC uncertainty
    st.markdown('<div class="section-header">Monte Carlo uncertainty (80 passes, ±3% input noise)</div>',
                unsafe_allow_html=True)
    col_mc1, col_mc2 = st.columns(2)
    with col_mc1:
        if mc['viscosity_cP']:
            mc_s = sorted(mc['viscosity_cP'])
            fig_mc = px.histogram(x=mc['viscosity_cP'], nbins=15, color_discrete_sequence=["#74c0fc"])
            fig_mc.add_vline(x=visc_pred, line_dash="dash", line_color="#4c6ef5",
                             annotation_text=f"{visc_pred:.0f} cP")
            fig_mc.update_layout(height=200, margin=dict(l=0,r=0,t=10,b=10),
                                  showlegend=False, xaxis_title="Viscosity (cP)",
                                  plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_mc, use_container_width=True)
            st.caption(f"90% CI: {mc_s[int(len(mc_s)*0.05)]:.0f} – {mc_s[int(len(mc_s)*0.95)]:.0f} cP")
    with col_mc2:
        if mc['li_extraction_score']:
            mc_ls = sorted(mc['li_extraction_score'])
            fig_mc2 = px.histogram(x=mc['li_extraction_score'], nbins=15, color_discrete_sequence=["#a9e34b"])
            fig_mc2.add_vline(x=li_score, line_dash="dash", line_color="#37b24d",
                              annotation_text=f"{li_score:.1f}/100")
            fig_mc2.update_layout(height=200, margin=dict(l=0,r=0,t=10,b=10),
                                   showlegend=False, xaxis_title="Li suitability score",
                                   plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_mc2, use_container_width=True)
            st.caption(f"90% CI: {mc_ls[int(len(mc_ls)*0.05)]:.1f} – {mc_ls[int(len(mc_ls)*0.95)]:.1f}")


# ════════ NEW — PROCESS OPTIMISER & DEEP ANALYSIS (still inside Tab 1) ════════

    st.markdown("---")
    st.markdown("### Deep Analysis & Optimisation")

    ana_tabs = st.tabs(["⚙️ Process Window","🗺️ Condition Heatmap",
                         "🧪 Co-Metal Selectivity","📉 Leaching Kinetics","🔬 DES Comparison"])

    # ── Analysis sub-tab 1: Process Window ─────────────────────────────────────
    with ana_tabs[0]:
        st.markdown("#### Optimal operating window for maximum Li recovery")
        temps_grid  = np.arange(30, 115, 5)
        waters_grid = np.linspace(0, 0.45, 10)
        z_rec = []
        for wg in waters_grid:
            row = []
            for tg in temps_grid:
                pg, _ = predict_all(hba, hbd, ratio, tg + 273.15, float(wg))
                if pg:
                    row.append(predict_li_recovery(pg, tg+273.15, source,
                                                    oxidant, assist, sl_ratio, pretreat, leach_t))
                else:
                    row.append(0)
            z_rec.append(row)

        fig_hw = go.Figure(go.Heatmap(
            x=temps_grid.tolist(),
            y=[f"{w:.2f}" for w in waters_grid],
            z=z_rec,
            colorscale="RdYlGn",
            zmin=55, zmax=99,
            colorbar=dict(title="Li Recovery %", ticksuffix="%"),
            hovertemplate="Temp: %{x}°C<br>Water: %{y} mol frac<br>Li Recovery: %{z:.1f}%<extra></extra>"
        ))
        fig_hw.add_scatter(x=[temp_C_val], y=[f"{water:.2f}"],
                           mode="markers", marker=dict(color="white", size=12,
                           symbol="star", line=dict(color="#4c6ef5", width=2)),
                           name="Current conditions", showlegend=True)
        fig_hw.update_layout(
            title="Li Recovery (%) — Temperature vs Water Content Heatmap",
            xaxis_title="Temperature (°C)", yaxis_title="Water content (mol frac)",
            height=360, margin=dict(l=0,r=0,t=50,b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.15)
        )
        st.plotly_chart(fig_hw, use_container_width=True)

        # Find best conditions
        best_val = 0; best_t = temp_C_val; best_w = water
        for wi, wg in enumerate(waters_grid):
            for ti, tg in enumerate(temps_grid):
                if z_rec[wi][ti] > best_val:
                    best_val = z_rec[wi][ti]; best_t = tg; best_w = wg
        st.success(f"**Optimal window:** {best_t}°C · {best_w:.2f} mol frac water → **{best_val:.1f}% Li recovery** "
                   f"(current: {li_rec:.1f}% at {temp_C_val}°C, {water:.2f} mol frac)")

    # ── Analysis sub-tab 2: Condition Heatmap (ratio vs temp) ──────────────────
    with ana_tabs[1]:
        st.markdown("#### Li recovery vs Molar Ratio × Temperature")
        ratios_g = np.linspace(0.25, 6.0, 12)
        temps_g2 = np.arange(40, 105, 5)
        z_rt = []
        for rg in ratios_g:
            row = []
            for tg in temps_g2:
                pg, _ = predict_all(hba, hbd, float(rg), tg+273.15, water)
                if pg:
                    row.append(predict_li_recovery(pg, tg+273.15, source,
                                                    oxidant, assist, sl_ratio, pretreat, leach_t))
                else:
                    row.append(0)
            z_rt.append(row)

        fig_rt = go.Figure(go.Heatmap(
            x=temps_g2.tolist(),
            y=[f"{r:.2f}" for r in ratios_g],
            z=z_rt,
            colorscale="Viridis",
            colorbar=dict(title="Li Recovery %", ticksuffix="%"),
            hovertemplate="Temp: %{x}°C<br>Ratio: %{y}<br>Li Recovery: %{z:.1f}%<extra></extra>"
        ))
        fig_rt.add_scatter(x=[temp_C_val], y=[f"{ratio:.2f}"],
                           mode="markers", marker=dict(color="red", size=12, symbol="star"),
                           name="Current", showlegend=True)
        fig_rt.update_layout(
            title="Li Recovery (%) — Molar Ratio vs Temperature",
            xaxis_title="Temperature (°C)", yaxis_title="HBA:HBD Molar Ratio",
            height=360, margin=dict(l=0,r=0,t=50,b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_rt, use_container_width=True)

        col_rt1, col_rt2 = st.columns(2)
        with col_rt1:
            best_rt = 0; best_r2 = ratio; best_t2 = temp_C_val
            for ri, rg in enumerate(ratios_g):
                for ti, tg in enumerate(temps_g2):
                    if z_rt[ri][ti] > best_rt:
                        best_rt = z_rt[ri][ti]; best_r2 = rg; best_t2 = tg
            st.info(f"**Best ratio:temp combo:** {best_r2:.2f} ratio at {best_t2}°C → **{best_rt:.1f}% Li**")
        with col_rt2:
            # Show how current ratio compares at current temp
            ratio_scan = [predict_all(hba, hbd, float(r), temp_K_val, water)[0] for r in ratios_g]
            rec_scan   = [predict_li_recovery(p, temp_K_val, source, oxidant, assist,
                                               sl_ratio, pretreat, leach_t) if p else 0
                          for p in ratio_scan]
            fig_rs = go.Figure(go.Scatter(x=ratios_g.tolist(), y=rec_scan,
                                           mode='lines+markers',
                                           line=dict(color='#4c6ef5', width=2.5)))
            fig_rs.add_vline(x=ratio, line_dash='dash', line_color='#f03e3e',
                              annotation_text=f"Current {ratio:.2f}")
            fig_rs.update_layout(title=f"Li Recovery vs Ratio @ {temp_C_val}°C",
                                  height=220, margin=dict(l=0,r=0,t=40,b=10),
                                  xaxis_title="Molar ratio", yaxis_title="Li Recovery %",
                                  plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_rs, use_container_width=True)

    # ── Analysis sub-tab 3: Co-Metal Selectivity ────────────────────────────────
    with ana_tabs[2]:
        st.markdown("#### Predicted metal recovery & Li selectivity")
        src_d = LI_SOURCES[source]

        # Li selectivity = Li recovery / average of co-metals
        co_metals = {k:v for k,v in {
            'Co': src_d['bench_co'], 'Ni': src_d['bench_ni'], 'Mn': src_d['bench_mn']
        }.items() if v > 0}
        avg_co = np.mean(list(co_metals.values())) if co_metals else 0

        # Scale co-metals by same factors affecting Li (viscosity, temp, pH)
        visc_f = max(0.72, 1 - max(0, visc_pred-50)/3000)
        temp_f = 0.82 + min(0.18, (temp_C_val-20)/400)
        metals_pred = {'Li': li_rec}
        for m, bench in co_metals.items():
            metals_pred[m] = min(99, round(bench * visc_f * temp_f * (1+OX_BOOST[oxidant]/100), 1))

        selectivity_li = li_rec / (avg_co * visc_f * temp_f + 0.01) if avg_co > 0 else 999

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            fig_sel = go.Figure()
            metal_colors = {'Li':'#4c6ef5','Co':'#37b24d','Ni':'#f76707','Mn':'#7950f2'}
            for metal, val in metals_pred.items():
                fig_sel.add_trace(go.Bar(
                    name=metal, x=[metal], y=[val],
                    marker_color=metal_colors.get(metal,'#adb5bd'),
                    text=[f"{val:.1f}%"], textposition='outside'
                ))
            fig_sel.update_layout(
                title="Predicted Metal Recovery from " + source,
                height=300, margin=dict(l=0,r=0,t=50,b=10),
                yaxis=dict(range=[0,110], ticksuffix="%"),
                showlegend=False, barmode='group',
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_sel, use_container_width=True)

        with col_s2:
            # Radar of Li vs co-metals
            if co_metals:
                cat_sel = ['Li'] + list(co_metals.keys())
                val_sel = [metals_pred['Li']] + [metals_pred.get(m, 0) for m in co_metals]
                fig_rad = go.Figure(go.Scatterpolar(
                    r=val_sel+[val_sel[0]], theta=cat_sel+[cat_sel[0]],
                    fill='toself', fillcolor='rgba(76,110,245,0.25)',
                    line=dict(color='#4c6ef5', width=2.5)))
                fig_rad.update_layout(
                    polar=dict(radialaxis=dict(range=[0,110])),
                    title="Metal Recovery Radar",
                    height=280, margin=dict(l=20,r=20,t=50,b=10),
                    paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_rad, use_container_width=True)

        # Selectivity stats
        sel_cols = st.columns(len(metals_pred))
        for col_i, (metal, val) in zip(sel_cols, metals_pred.items()):
            col_i.markdown(f"""
            <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;
                        padding:12px;text-align:center">
              <div style="font-size:11px;color:#94a3b8;font-weight:600">{metal}</div>
              <div style="font-size:24px;font-weight:700;color:#f1f5f9">{val:.1f}%</div>
              <div style="font-size:11px;color:#64748b">{"Li target" if metal=="Li" else "co-leached"}</div>
            </div>""", unsafe_allow_html=True)

        if selectivity_li < 1.1:
            st.warning(f"**Low Li selectivity ({selectivity_li:.2f}×)** — co-metals leaching at similar rate. "
                       "Consider lower temperature or adjusted pH to improve separation.")
        elif selectivity_li > 1.2:
            st.success(f"**Good Li selectivity ({selectivity_li:.2f}×)** — Li leaching preferentially over co-metals.")

    # ── Analysis sub-tab 4: Leaching Kinetics ──────────────────────────────────
    with ana_tabs[3]:
        st.markdown("#### Simulated Li leaching kinetics over time")
        st.caption("Based on first-order kinetic model calibrated to viscosity and temperature predictions.")

        times = np.linspace(0, 180, 50)
        # Rate constant depends on viscosity (diffusion-controlled) and temperature (Arrhenius)
        k_base  = 0.035
        visc_k  = k_base * max(0.3, 1 - (visc_pred - 25) / 2000)
        temp_k  = visc_k * np.exp(0.02 * (temp_C_val - 25))
        li_kin  = li_rec * (1 - np.exp(-temp_k * times))

        # Compare across temps
        kin_fig = go.Figure()
        comp_temps = [40, 60, 80, 100]
        colors_k = ['#94a3b8','#60a5fa','#4c6ef5','#1d4ed8']
        for ct, ck_col in zip(comp_temps, colors_k):
            pg_k, _ = predict_all(hba, hbd, ratio, ct+273.15, water)
            if pg_k:
                rec_k = predict_li_recovery(pg_k, ct+273.15, source,
                                             oxidant, assist, sl_ratio, pretreat, leach_t)
                visc_k2 = k_base * max(0.3, 1 - (pg_k['viscosity_cP']-25)/2000)
                temp_k2 = visc_k2 * np.exp(0.02*(ct-25))
                y_k = rec_k * (1 - np.exp(-temp_k2 * times))
                width = 2.5 if ct == temp_C_val else 1.5
                dash  = "solid" if ct == temp_C_val else "dot"
                kin_fig.add_trace(go.Scatter(
                    x=times, y=y_k, mode='lines',
                    name=f"{ct}°C ({rec_k:.0f}% final)",
                    line=dict(color=ck_col, width=width, dash=dash)))

        # Add leach time markers
        for lt_label, lt_factor in LEACH_TIME.items():
            lt_mins = int(lt_label.split()[0])
            kin_fig.add_vline(x=lt_mins, line_dash="dot", line_color="#475569",
                               line_width=1,
                               annotation_text=lt_label,
                               annotation_font_size=10)

        kin_fig.add_hline(y=li_rec, line_dash="dash", line_color="#f03e3e",
                           annotation_text=f"Current: {li_rec:.1f}%",
                           annotation_font_size=10)
        kin_fig.update_layout(
            title=f"Li Leaching Kinetics — {hba} : {hbd}",
            xaxis_title="Time (min)", yaxis_title="Li Leaching Efficiency (%)",
            height=340, margin=dict(l=0,r=0,t=50,b=10),
            legend=dict(orientation="h", y=-0.2),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(range=[0,105])
        )
        st.plotly_chart(kin_fig, use_container_width=True)

        t90 = -np.log(1 - 0.9/li_rec * (li_rec/li_rec)) / temp_k if li_rec > 0 else 999
        t90_approx = -np.log(0.1) / temp_k if temp_k > 0 else 999
        col_k1, col_k2, col_k3 = st.columns(3)
        col_k1.markdown(f"""<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;
                          padding:12px;text-align:center">
          <div style="font-size:11px;color:#94a3b8;font-weight:600">EST. TIME TO 90% OF MAX</div>
          <div style="font-size:22px;font-weight:700;color:#f1f5f9">{min(180,int(t90_approx))} min</div>
        </div>""", unsafe_allow_html=True)
        col_k2.markdown(f"""<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;
                          padding:12px;text-align:center">
          <div style="font-size:11px;color:#94a3b8;font-weight:600">RATE CONSTANT k</div>
          <div style="font-size:22px;font-weight:700;color:#60a5fa">{temp_k:.4f} min⁻¹</div>
        </div>""", unsafe_allow_html=True)
        col_k3.markdown(f"""<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;
                          padding:12px;text-align:center">
          <div style="font-size:11px;color:#94a3b8;font-weight:600">KINETIC CONTROL</div>
          <div style="font-size:16px;font-weight:700;color:#f1f5f9">
            {"Diffusion" if visc_pred>200 else "Mixed" if visc_pred>80 else "Reaction"}</div>
          <div style="font-size:11px;color:#64748b">(based on viscosity)</div>
        </div>""", unsafe_allow_html=True)

    # ── Analysis sub-tab 5: DES Comparison ─────────────────────────────────────
    with ana_tabs[4]:
        st.markdown("#### Compare current DES against top Li candidates")
        compare_systems = [
            ("choline chloride", "oxalic acid",    1.0),
            ("choline chloride", "levulinic acid", 2.0),
            ("choline chloride", "malonic acid",   1.0),
            ("choline chloride", "lactic acid",    2.0),
            ("choline chloride", "dl-malic acid",  1.0),
            ("choline chloride", "ethylene glycol",2.0),
            ("lactic acid",      "betaine",        3.0),
        ]
        compare_data = []
        for c_hba, c_hbd, c_ratio in compare_systems:
            if c_hba in le_hba.classes_ and c_hbd in le_hbd.classes_:
                cp, _ = predict_all(c_hba, c_hbd, c_ratio, temp_K_val, water)
                if cp:
                    c_rec = predict_li_recovery(cp, temp_K_val, source,
                                                 oxidant, assist, sl_ratio, pretreat, leach_t)
                    compare_data.append({
                        'System': f"{c_hba.replace('choline chloride','ChCl')} : {c_hbd} ({c_ratio}:1)",
                        'Li Recovery': round(c_rec, 1),
                        'Viscosity (cP)': round(cp['viscosity_cP'], 1),
                        'pH': round(cp['pH_acidity'], 2),
                        'Redox (V)': round(cp['reduction_potential_V_vs_SHE'], 3),
                        'Li Score': round(cp.get('li_extraction_score', 0), 1),
                        'Current': (c_hba == hba and c_hbd == hbd),
                    })

        # Add current system if not already in list
        current_in = any(d['Current'] for d in compare_data)
        if not current_in:
            compare_data.insert(0, {
                'System': f"[CURRENT] {hba} : {hbd} ({ratio}:1)",
                'Li Recovery': li_rec,
                'Viscosity (cP)': round(visc_pred, 1),
                'pH': round(ph_pred, 2),
                'Redox (V)': round(redox_pred, 3),
                'Li Score': round(li_score, 1),
                'Current': True,
            })

        cdf = pd.DataFrame(compare_data)
        cdf_sorted = cdf.sort_values('Li Recovery', ascending=False).reset_index(drop=True)

        # Bar comparison
        bar_colors = ['#f59e0b' if r['Current'] else '#4c6ef5' for _, r in cdf_sorted.iterrows()]
        fig_cmp = go.Figure(go.Bar(
            x=cdf_sorted['System'], y=cdf_sorted['Li Recovery'],
            marker_color=bar_colors,
            text=[f"{v:.1f}%" for v in cdf_sorted['Li Recovery']],
            textposition='outside'
        ))
        fig_cmp.add_hline(y=bench_li, line_dash="dash", line_color="#94a3b8",
                           annotation_text=f"Benchmark {bench_li}%")
        fig_cmp.update_layout(
            title=f"Li Recovery Comparison at {temp_C_val}°C (gold = current selection)",
            height=320, margin=dict(l=0,r=0,t=50,b=10),
            yaxis=dict(range=[0, 105], ticksuffix="%"),
            xaxis_tickangle=-30,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_cmp, use_container_width=True)

        # Table with conditional color
        show_cols = ['System','Li Recovery','Viscosity (cP)','pH','Li Score']
        st.dataframe(
            cdf_sorted[show_cols].style.background_gradient(
                subset=['Li Recovery','Li Score'], cmap='RdYlGn', vmin=50, vmax=100
            ).background_gradient(
                subset=['Viscosity (cP)'], cmap='RdYlGn_r', vmin=0, vmax=500
            ).background_gradient(
                subset=['pH'], cmap='RdYlGn_r', vmin=1, vmax=8
            ),
            use_container_width=True, hide_index=True
        )


# ════════ TAB 2 — DES PROPERTIES ════════
with tabs[1]:
    st.markdown("### All Predicted DES Properties")
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:16px">
      <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px 14px">
        <div style="font-size:10px;color:#94a3b8;font-weight:600;margin-bottom:4px">VISCOSITY · R²=0.896</div>
        <div style="font-size:22px;font-weight:700;color:#f1f5f9">{visc_pred:.1f} <span style="font-size:12px;color:#64748b">cP</span></div>
      </div>
      <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px 14px">
        <div style="font-size:10px;color:#94a3b8;font-weight:600;margin-bottom:4px">DENSITY · R²=0.995</div>
        <div style="font-size:22px;font-weight:700;color:#f1f5f9">{dens_pred:.1f} <span style="font-size:12px;color:#64748b">kg/m³</span></div>
      </div>
      <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px 14px">
        <div style="font-size:10px;color:#94a3b8;font-weight:600;margin-bottom:4px">pH · R²=0.989</div>
        <div style="font-size:22px;font-weight:700;color:#f1f5f9">{ph_pred:.2f}</div>
      </div>
      <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px 14px">
        <div style="font-size:10px;color:#94a3b8;font-weight:600;margin-bottom:4px">REDOX · R²=0.973</div>
        <div style="font-size:22px;font-weight:700;color:#f1f5f9">{redox_pred:.3f} <span style="font-size:12px;color:#64748b">V</span></div>
      </div>
      <div style="background:#0f3460;border:1px solid #1d4ed8;border-radius:10px;padding:12px 14px">
        <div style="font-size:10px;color:#93c5fd;font-weight:600;margin-bottom:4px">LI SCORE · R²=0.868</div>
        <div style="font-size:22px;font-weight:700;color:#60a5fa">{li_score:.1f} <span style="font-size:12px;color:#93c5fd">/100</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        # Radar
        norm = [
            min(visc_pred/5000,1)*100,
            max(0,(dens_pred-1000)/400*100),
            ph_pred/10*100,
            abs(redox_pred)/2*100,
            li_score,
        ]
        cats = ['Viscosity','Density(rel)','pH/10','|Redox|/2V','Li Score']
        fig_r = go.Figure(go.Scatterpolar(
            r=norm+[norm[0]], theta=cats+[cats[0]],
            fill='toself', fillcolor='rgba(76,110,245,0.2)',
            line=dict(color='#4c6ef5',width=2.5)))
        # Li ideal zone overlay
        li_ideal = [20, 50, 80, 70, 80]
        fig_r.add_trace(go.Scatterpolar(
            r=li_ideal+[li_ideal[0]], theta=cats+[cats[0]],
            fill='toself', fillcolor='rgba(55,178,77,0.08)',
            line=dict(color='#37b24d', width=1.5, dash='dash'),
            name='Li ideal zone'))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100])),
                             height=320, margin=dict(l=20,r=20,t=30,b=20),
                             paper_bgcolor='rgba(0,0,0,0)',
                             legend=dict(orientation='h', y=-0.15))
        st.plotly_chart(fig_r, use_container_width=True)
    with col2:
        # Accuracy table
        st.markdown("### Model accuracy")
        perf_rows = [
            {'Property': 'viscosity_cP',                 'R²': 0.8959, 'MAE': '315 cP',    'Train N': 4920},
            {'Property': 'density_kg_m3',                'R²': 0.9948, 'MAE': '1.04 kg/m³','Train N': 1264},
            {'Property': 'pH_acidity',                   'R²': 0.9888, 'MAE': '0.076',      'Train N': 1005},
            {'Property': 'reduction_potential_V_vs_SHE', 'R²': 0.9729, 'MAE': '0.029 V',   'Train N': 1737},
            {'Property': 'li_extraction_score (NEW)',    'R²': 0.8679, 'MAE': '1.94%',      'Train N':  878},
        ]
        st.dataframe(pd.DataFrame(perf_rows), hide_index=True, use_container_width=True)
        st.info("**New in v2:** A dedicated Li extraction score model (5th XGBoost) was trained on "
                "1,033 rows where all 3 key properties are simultaneously known, producing a "
                "combined suitability score calibrated against literature Li recovery benchmarks.")


# ════════ TAB 3 — FEATURE IMPORTANCE ════════
with tabs[2]:
    st.markdown("### XGBoost Feature Importance — All 5 Models")
    st.info("Two new binary features added: **Is Li-HBA** and **Is Li-HBD** — "
            "flagging compounds known in literature for Li battery leaching.")

    col1, col2 = st.columns(2)
    for idx, (target, fvals) in enumerate(fi.items()):
        col = col1 if idx % 2 == 0 else col2
        with col:
            r2 = MODEL_RESULTS[target]['r2']
            fig_fi = go.Figure(go.Bar(
                x=list(fvals.values()), y=list(fvals.keys()), orientation='h',
                marker_color=['#4c6ef5' if 'Li' not in k else '#37b24d' for k in fvals.keys()],
                text=[f"{v*100:.1f}%" for v in fvals.values()], textposition='outside'))
            fig_fi.update_layout(
                title=f"{target.replace('_',' ')} (R²={r2})",
                height=250, margin=dict(l=0,r=70,t=40,b=10),
                xaxis=dict(tickformat='.0%', title='Importance'),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_fi, use_container_width=True)


# ════════ TAB 4 — PROPERTY SWEEP ════════
with tabs[3]:
    st.markdown(f"### Temperature & Water Sweep — **{hba}** : **{hbd}** (ratio {ratio})")

    sweep = sweep_temps(hba, hbd, ratio, water)
    if sweep is None:
        st.error("Could not sweep — compound not in encoder.")
    else:
        tc = sweep['temps_C']
        col1, col2 = st.columns(2)

        def make_sweep_fig(x, y, title, ytitle, color, vline_x, highlight_range=None):
            fig = go.Figure()
            if highlight_range:
                fig.add_vrect(x0=highlight_range[0], x1=highlight_range[1],
                              fillcolor="#dcfce7", opacity=0.3, line_width=0,
                              annotation_text="Li optimal", annotation_position="top left")
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers',
                                     line=dict(color=color, width=2.5), marker=dict(size=4)))
            fig.add_vline(x=vline_x, line_dash='dash', line_color='#f03e3e',
                          annotation_text=f"{vline_x:.0f}°C")
            fig.update_layout(title=title, height=250, margin=dict(l=0,r=0,t=40,b=10),
                               xaxis_title="Temperature (°C)", yaxis_title=ytitle,
                               plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            return fig

        with col1:
            st.plotly_chart(make_sweep_fig(tc, sweep['viscosity_cP'], "Viscosity vs Temperature",
                "Viscosity (cP)", "#4c6ef5", temp_C_val, highlight_range=[60,90]),
                use_container_width=True)
            st.plotly_chart(make_sweep_fig(tc, sweep['pH_acidity'], "pH vs Temperature",
                "pH", "#f76707", temp_C_val, highlight_range=[60,90]),
                use_container_width=True)
        with col2:
            st.plotly_chart(make_sweep_fig(tc, sweep['li_extraction_score'],
                "Li Suitability Score vs Temperature",
                "Li score (0–100)", "#37b24d", temp_C_val, highlight_range=[60,90]),
                use_container_width=True)
            st.plotly_chart(make_sweep_fig(tc, sweep['density_kg_m3'], "Density vs Temperature",
                "Density (kg/m³)", "#7950f2", temp_C_val),
                use_container_width=True)

        # Water content sweep
        st.markdown("### Water content sweep (at current temperature)")
        water_vals = np.linspace(0, 0.5, 25)
        w_visc, w_li = [], []
        for w in water_vals:
            p2, _ = predict_all(hba, hbd, ratio, temp_K_val, float(w))
            if p2:
                w_visc.append(p2['viscosity_cP'])
                w_li.append(p2.get('li_extraction_score', 0))
        if w_visc:
            col_w1, col_w2 = st.columns(2)
            with col_w1:
                fig_wv = go.Figure(go.Scatter(x=water_vals, y=w_visc, mode='lines+markers',
                                               line=dict(color='#4c6ef5', width=2.5)))
                fig_wv.add_vline(x=water, line_dash='dash', line_color='#f03e3e')
                fig_wv.update_layout(title="Viscosity vs Water Content", height=230,
                                      xaxis_title="Water (mol fraction)", yaxis_title="Viscosity (cP)",
                                      margin=dict(l=0,r=0,t=40,b=10),
                                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_wv, use_container_width=True)
            with col_w2:
                fig_wl = go.Figure(go.Scatter(x=water_vals, y=w_li, mode='lines+markers',
                                               line=dict(color='#37b24d', width=2.5)))
                fig_wl.add_vline(x=water, line_dash='dash', line_color='#f03e3e')
                fig_wl.update_layout(title="Li Score vs Water Content", height=230,
                                      xaxis_title="Water (mol fraction)", yaxis_title="Li score (0–100)",
                                      margin=dict(l=0,r=0,t=40,b=10),
                                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_wl, use_container_width=True)


# ════════ TAB 5 — REUSE SIMULATOR ════════
with tabs[4]:
    col1, col2 = st.columns([1, 1.6])
    with col1:
        st.markdown("### Configuration")
        r_sys    = st.selectbox("DES system", [s['name'] for s in LI_DES_SYSTEMS])
        r_regen  = st.selectbox("Regeneration method", list(REGEN_METHODS.keys()))
        r_cycles = st.slider("Reuse cycles to simulate", 1, 15, 6)
        r_temp2  = st.slider("Operating temperature (°C)  ", 60, 180, 80, 5)
        r_purity = st.slider("Target Li purity (%)", 70, 99, 90, 1)

        rg  = REGEN_METHODS[r_regen]
        sys_data = next(s for s in LI_DES_SYSTEMS if s['name'] == r_sys)
        base_eff = sys_data['li_eff']
        # Loss per cycle depends on regen quality and temperature stress
        tf2 = 1.0 + max(0, r_temp2 - 100) / 500
        loss_cyc = (100 - base_eff) * 0.3 * tf2 * (1 - rg['li_recovery'] * 0.6)
        cyc_eff  = [max(55, base_eff - loss_cyc * i) for i in range(r_cycles)]
        purity_m = [max(50, r_purity - i * 0.5 * tf2) for i in range(r_cycles)]
        no_regen = [max(40, base_eff - (100-base_eff)*0.4*tf2*i) for i in range(r_cycles)]
        lifetime = next((i for i,e in enumerate(cyc_eff) if e < r_purity), r_cycles)

        m1,m2 = st.columns(2); m3,m4 = st.columns(2)
        m1.metric("Cycle 1 Li eff.", f"{cyc_eff[0]:.1f}%")
        m2.metric("Final cycle eff.", f"{cyc_eff[-1]:.1f}%")
        m3.metric("Est. useful life", f"{lifetime} cycles")
        m4.metric("DES viscosity", f"{sys_data['visc']} cP")

    with col2:
        cl = [f"C{i+1}" for i in range(r_cycles)]
        fig_ru = go.Figure()
        fig_ru.add_trace(go.Scatter(x=cl, y=cyc_eff, mode="lines+markers",
                                    name="With regen", line=dict(color="#4c6ef5", width=2.5)))
        fig_ru.add_trace(go.Scatter(x=cl, y=no_regen, mode="lines+markers",
                                    name="No regen", line=dict(color="#f03e3e", width=2, dash="dash")))
        fig_ru.add_hrect(y0=r_purity, y1=102, fillcolor="#dcfce7", opacity=0.25, line_width=0,
                          annotation_text=f"Target ≥{r_purity}%", annotation_position="top right")
        fig_ru.update_layout(title="Li Recovery Efficiency vs Cycle",
                              height=270, margin=dict(l=0,r=0,t=40,b=10),
                              yaxis=dict(range=[40,103], ticksuffix="%"),
                              legend=dict(orientation="h", y=-0.2),
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_ru, use_container_width=True)

        # Cumulative Li recovered
        cum_regen   = [sum(cyc_eff[:i+1]) for i in range(r_cycles)]
        cum_noregen = [sum(no_regen[:i+1]) for i in range(r_cycles)]
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=cl, y=cum_regen, fill="tozeroy", mode="lines",
                                     name="With regen", line=dict(color="#4c6ef5",width=2),
                                     fillcolor="rgba(76,110,245,.12)"))
        fig_cum.add_trace(go.Scatter(x=cl, y=cum_noregen, mode="lines",
                                     name="No regen", line=dict(color="#f03e3e",width=2,dash="dash")))
        fig_cum.update_layout(title="Cumulative Li recovered (relative)",
                               height=220, margin=dict(l=0,r=0,t=40,b=10),
                               legend=dict(orientation="h", y=-0.25),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cum, use_container_width=True)


# ════════ TAB 6 — DES RANKER ════════
with tabs[5]:
    st.markdown("### Li-Optimised DES Systems Ranker")
    col1, col2 = st.columns([1,1.5])

    with col1:
        st.markdown("#### Objective weights")
        pw_li  = st.slider("Li efficiency weight (%)", 0, 100, 50, 5)
        pw_co  = st.slider("Cost weight (%)",          0, 100, 25, 5)
        pw_gr  = st.slider("Green score weight (%)",   0, 100, 25, 5)
        tw = pw_li + pw_co + pw_gr
        we, wc, wg = (pw_li/100, pw_co/100, pw_gr/100) if tw > 0 else (0,0,0)

        pf = is_pareto(LI_DES_SYSTEMS)
        scored = sorted([dict(**s, score=round(
            we*(s["li_eff"]/100)*100 + wc*(1-s["cost"])*100 + wg*s["green"]*100, 1))
            for s in LI_DES_SYSTEMS], key=lambda x: x["score"], reverse=True)

        st.markdown("#### Top Li DES systems")
        for i, s in enumerate(scored[:7]):
            visc_icon = "💧" if s["visc"]<100 else ("⚠️" if s["visc"]<500 else "🔴")
            st.markdown(f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
              <span style="width:22px;color:#adb5bd;font-weight:700;font-size:14px">#{i+1}</span>
              <div style="flex:1">
                <div style="font-weight:600;font-size:13px">{s['name']}</div>
                <div style="font-size:11px;color:#6c757d">
                  Li {s['li_eff']}% · Cost {s['cost']:.2f} · Green {s['green']*100:.0f}% · {visc_icon}{s['visc']} cP</div>
                <div style="height:5px;background:#e9ecef;border-radius:3px;margin-top:3px">
                  <div style="height:5px;width:{min(int(s['score']),100)}%;background:#4c6ef5;border-radius:3px"></div></div>
              </div>
              <span class="badge-ok">{s['score']:.0f}pts</span>
            </div>""", unsafe_allow_html=True)

    with col2:
        po = [s for s,f in zip(LI_DES_SYSTEMS, pf) if f]
        pd2= [s for s,f in zip(LI_DES_SYSTEMS, pf) if not f]
        fig_p = go.Figure()
        if pd2:
            fig_p.add_trace(go.Scatter(
                x=[s["cost"] for s in pd2], y=[s["li_eff"] for s in pd2],
                mode="markers", name="Sub-optimal",
                marker=dict(color="#adb5bd", size=10),
                text=[s["name"] for s in pd2], hoverinfo="text+x+y"))
        if po:
            fig_p.add_trace(go.Scatter(
                x=[s["cost"] for s in po], y=[s["li_eff"] for s in po],
                mode="markers+text", name="Pareto-optimal",
                marker=dict(color="#4c6ef5", size=14, symbol="star"),
                text=[s["name"].split(":")[0] for s in po],
                textposition="top center", hoverinfo="text+x+y"))
        fig_p.update_layout(
            title="Pareto frontier — Li efficiency vs cost",
            height=320, margin=dict(l=0,r=0,t=40,b=10),
            xaxis=dict(title="Relative cost (lower=better)", autorange="reversed"),
            yaxis=dict(title="Li Leaching Efficiency (%)", range=[83, 103]),
            legend=dict(orientation="h", y=-0.25),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_p, use_container_width=True)

        # Viscosity vs Li efficiency scatter
        fig_vl = go.Figure()
        colors = ['#4c6ef5' if f else '#adb5bd' for f in pf]
        fig_vl.add_trace(go.Scatter(
            x=[s["visc"] for s in LI_DES_SYSTEMS],
            y=[s["li_eff"] for s in LI_DES_SYSTEMS],
            mode="markers+text",
            marker=dict(color=colors, size=12),
            text=[s["name"].split(":")[0] for s in LI_DES_SYSTEMS],
            textposition="top center", hoverinfo="text+x+y"))
        fig_vl.add_vline(x=200, line_dash="dash", line_color="#f03e3e",
                          annotation_text="200 cP threshold")
        fig_vl.update_layout(
            title="Li Efficiency vs Viscosity",
            height=280, margin=dict(l=0,r=0,t=40,b=10),
            xaxis_title="Viscosity (cP)", yaxis_title="Li Efficiency (%)",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_vl, use_container_width=True)


# ════════ TAB 7 — MODEL INFO ════════
with tabs[6]:
    st.markdown("""
    <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:16px">
      <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px 14px;text-align:center">
        <div style="font-size:10px;color:#94a3b8;font-weight:600;margin-bottom:4px">TRAINING ROWS</div>
        <div style="font-size:26px;font-weight:700;color:#f1f5f9">5,790</div>
      </div>
      <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px 14px;text-align:center">
        <div style="font-size:10px;color:#94a3b8;font-weight:600;margin-bottom:4px">MODELS</div>
        <div style="font-size:26px;font-weight:700;color:#60a5fa">5</div>
        <div style="font-size:11px;color:#64748b">XGBoost</div>
      </div>
      <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px 14px;text-align:center">
        <div style="font-size:10px;color:#94a3b8;font-weight:600;margin-bottom:4px">UNIQUE HBAs</div>
        <div style="font-size:26px;font-weight:700;color:#f1f5f9">144</div>
      </div>
      <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px 14px;text-align:center">
        <div style="font-size:10px;color:#94a3b8;font-weight:600;margin-bottom:4px">UNIQUE HBDs</div>
        <div style="font-size:26px;font-weight:700;color:#f1f5f9">167</div>
      </div>
      <div style="background:#0f3460;border:1px solid #1d4ed8;border-radius:10px;padding:12px 14px;text-align:center">
        <div style="font-size:10px;color:#93c5fd;font-weight:600;margin-bottom:4px">Li FEATURES</div>
        <div style="font-size:26px;font-weight:700;color:#60a5fa">2</div>
        <div style="font-size:11px;color:#93c5fd">new</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Model Architecture — v2 Li-specific")
        st.markdown("""
| Property | R² | MAE | Train N | Notes |
|---|---|---|---|---|
| viscosity_cP | 0.8959 | 315 cP | 4,920 | +Is_Li_HBA/HBD features |
| density_kg_m3 | 0.9948 | 1.04 kg/m³ | 1,264 | |
| pH_acidity | 0.9888 | 0.076 | 1,005 | |
| reduction_potential | 0.9729 | 0.029 V | 1,737 | |
| **li_extraction_score** | **0.8679** | **1.94%** | **878** | **NEW — Li-specific model** |

**New features added to all models:**
- `is_li_hba` — 1 if HBA is in the 11 literature-curated Li-relevant HBAs
- `is_li_hbd` — 1 if HBD is in the 14 literature-curated Li-relevant HBDs

**Li extraction score** is trained on rows where viscosity + pH + redox are all
simultaneously measured, computing a weighted suitability score calibrated to
Li recovery benchmarks from the literature.
        """)

    with col2:
        st.markdown("### Li-curated Compounds (from literature)")
        li_df = pd.DataFrame({
            'Li HBAs (11)': BEST_LI_HBAS + ['']*(max(0,len(BEST_LI_HBDS)-len(BEST_LI_HBAS))),
            'Li HBDs (13)': BEST_LI_HBDS + ['']*(max(0,len(BEST_LI_HBAS)-len(BEST_LI_HBDS))),
        })
        st.dataframe(li_df, hide_index=True, use_container_width=True)

        st.markdown("### Key Li extraction thresholds")
        st.markdown("""
| Property | Optimal range | Why it matters |
|---|---|---|
| Viscosity | < 200 cP | Li⁺ ion mobility |
| pH | 1.5 – 3.5 | Dissolves Li₂CO₃, LiCoO₂ |
| Redox potential | < −0.9 V | Reduces Co³⁺ → Co²⁺ releasing Li⁺ |
| Temperature | 60 – 90 °C | Balances rate vs DES stability |
| Water content | 0.0 – 0.2 | Maintains H-bond network |
        """)
        st.info("**Model storage:** All 5 XGBoost models are embedded as base64 in "
                "`model_data.py` — no external pkl/json files needed.")
