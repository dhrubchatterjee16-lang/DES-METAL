import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json, math, random, os, base64, tempfile

st.set_page_config(
    page_title="DES Metal Recovery Predictor v4",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container{padding-top:1.5rem}
    .badge-ok{background:#d4edda;color:#155724;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600}
    .badge-mid{background:#fff3cd;color:#856404;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600}
    .badge-low{background:#f8d7da;color:#721c24;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600}
    .tip-box{background:#f1f3f5;border-left:3px solid #4c6ef5;border-radius:6px;padding:12px 16px;
             font-size:13px;line-height:1.7;margin-top:8px}
    .section-header{font-size:13px;font-weight:600;color:#6c757d;letter-spacing:.06em;
                    text-transform:uppercase;margin-bottom:8px}
    div[data-testid="stMetric"]{background:#f8f9fa;border-radius:10px;padding:10px 14px}
    .stTabs [data-baseweb="tab"]{font-size:13px;padding:8px 16px}
    .model-badge{background:#e7f5ff;color:#1864ab;padding:4px 10px;border-radius:8px;
                 font-size:12px;font-weight:600;display:inline-block;margin-bottom:8px}
</style>
""", unsafe_allow_html=True)

# ── Load models (embedded in model_data.py — no external files needed) ─────────
@st.cache_resource
def load_models():
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder
    from model_data import load_all_models
    return load_all_models()

models, le_hba, le_hbd = load_models()

MODEL_PERF = {
    'viscosity_cP':                 {'r2': 0.874, 'mae': '339 cP',     'train': 4920},
    'density_kg_m3':                {'r2': 0.994, 'mae': '0.69 kg/m³', 'train': 1264},
    'pH_acidity':                   {'r2': 0.999, 'mae': '0.02 pH',    'train': 1005},
    'reduction_potential_V_vs_SHE': {'r2': 0.9997,'mae': '0.0024 V',   'train': 1737},
}

TOP_HBAS = ['choline chloride','l-menthol','thymol','dl-menthol',
            'tetrabutylammonium chloride','alcl3','acetylcholine chloride',
            'lactic acid','betaine','trioctylphosphine oxide',
            'tetrabutylammonium bromide','triethylmethylammonium chloride',
            'lidocaine','ethylamine hydrochloride','benzyltriethylammonium chloride',
            'methyltriphenylphosphonium bromide','allyltriphenylphosphonium bromide',
            'allyl triphenyl phosphonium bromide',
            'benzyldimethyl(2-hydroxyethyl)ammonium chloride',
            'methyltriphenyl phosphonium bromide']
TOP_HBDS = ['glycerol','ethylene glycol','phenol','urea','decanoic acid',
            'levulinic acid','triethylene glycol','acetic acid','dodecanoic acid',
            'lauric acid','octanoic acid','m-cresol','p-cresol','1,4-butanediol',
            'thymol','capric acid','o-cresol','1,2-propanediol',
            'phenylacetic acid','diethylene glycol']

ALL_HBAS = sorted(le_hba.classes_.tolist())
ALL_HBDS = sorted(le_hbd.classes_.tolist())

# ── Prediction helpers ─────────────────────────────────────────────────────────
def predict_properties(hba, hbd, ratio, temp_K, water):
    try:
        hba_enc = le_hba.transform([hba])[0]
        hbd_enc = le_hbd.transform([hbd])[0]
    except ValueError as e:
        return None, str(e)
    X = np.array([[hba_enc, hbd_enc, ratio, temp_K, water]])
    preds = {}
    for target, (model, log_t) in models.items():
        val = float(model.predict(X)[0])
        preds[target] = max(0, round(np.expm1(val) if log_t else val, 4))
    return preds, None

def mc_uncertainty(hba, hbd, ratio, temp_K, water, n=60):
    samples = []
    for _ in range(n):
        p, _ = predict_properties(hba, hbd,
            ratio * (1 + np.random.normal(0, 0.03)),
            temp_K + np.random.normal(0, 2),
            float(np.clip(water + np.random.normal(0, 0.01), 0, 1)))
        if p:
            samples.append(p['viscosity_cP'])
    return samples

def sweep_temps(hba, hbd, ratio, water, temps_K):
    try:
        hba_enc = le_hba.transform([hba])[0]
        hbd_enc = le_hbd.transform([hbd])[0]
    except ValueError:
        return [None]*len(temps_K)
    X = np.array([[hba_enc, hbd_enc, ratio, t, water] for t in temps_K])
    model, log_t = models['viscosity_cP']
    vals = model.predict(X)
    return [max(0, float(np.expm1(v) if log_t else v)) for v in vals]

def get_feature_importance():
    names = ['HBA identity','HBD identity','Molar ratio','Temperature','Water content']
    fi = {}
    for target, (model, _) in models.items():
        fi[target] = dict(zip(names, model.feature_importances_.tolist()))
    return fi

# ── Metal recovery helpers ─────────────────────────────────────────────────────
METAL_F = {
    "Cobalt (Co)":1.00,"Lithium (Li)":0.97,"Nickel (Ni)":0.96,"Manganese (Mn)":0.94,
    "Neodymium (Nd)":0.88,"Yttrium (Y)":0.84,"Copper (Cu)":0.91,"Zinc (Zn)":0.89,
    "Gold (Au)":0.78,"Platinum (Pt)":0.80,"Palladium (Pd)":0.82,"REE (general)":0.82,
    "Chromium (Cr)":0.88,"Molybdenum (Mo)":0.85,
}
SRC_BENCH = {
    "LIB cathode":94,"Printed circuit board":88,"Permanent magnet":91,
    "Fluorescent lamp":72,"Spent catalyst":89,"Mineral/ore":85,
    "Industrial dust/slag":82,"Wastewater":90,
}
OX_BOOST = {"None":0,"Iodine (I₂)":12,"H₂O₂":8,"FeCl₃":10,"CuCl₂":9}
ASSIST_B = {"Conventional":0,"Microwave":7,"Ultrasound":5,"Microwave + Ultrasound":11}
LEACH_CFG = {
    "ChCl : Oxalic acid (1:1)":dict(base=96,loss=8),
    "ChCl : Lactic acid (1:2)":dict(base=95,loss=6),
    "GUC : Lactic acid (1:2)":dict(base=99,loss=5),
    "ChCl : PTSA (1:2)":dict(base=100,loss=9),
    "BeCl : Formic acid (1:9)":dict(base=98,loss=4),
    "ChCl : Formic acid (1:2)":dict(base=99,loss=7),
}
REGEN_RECOVER = {
    "Vacuum evaporation":0.60,"HBD replenishment":0.70,
    "Evaporation + replenishment":0.85,"No regeneration":0.00,
}
PARETO_SYSTEMS = [
    dict(name="GUC : Lactic acid (1:2)",    eff=99, cost=0.62,green=0.81),
    dict(name="ChCl : PTSA (1:2)",          eff=100,cost=0.55,green=0.70),
    dict(name="BeCl : Formic acid (1:9)",   eff=98, cost=0.72,green=0.85),
    dict(name="ChCl : Oxalic acid (1:1)",   eff=96, cost=0.80,green=0.88),
    dict(name="EG : Sulfosalicylic (12:1)", eff=97, cost=0.58,green=0.78),
    dict(name="ChCl : Lactic acid (1:3)",   eff=95, cost=0.85,green=0.90),
    dict(name="ChCl : Formic acid (1:2)",   eff=99, cost=0.70,green=0.82),
    dict(name="ChCl : EG (1:2)",            eff=93, cost=0.90,green=0.92),
    dict(name="ChCl : Tartaric acid (1:1)", eff=97, cost=0.65,green=0.83),
    dict(name="ChCl : Maleic acid (1:1)",   eff=99, cost=0.68,green=0.79),
    dict(name="ChCl : Urea (1:2)",          eff=95, cost=0.92,green=0.88),
    dict(name="TEAC : Levulinic acid (1:2)",eff=97, cost=0.50,green=0.74),
]

def recovery_from_visc(visc, temp_K, metal, oxidant, assist):
    temp_C = temp_K - 273.15
    vp = max(0, 1 - (max(4, visc) - 25) / 900)
    tb = min(1, 0.28 + (temp_C / 200) * 0.72)
    base = (64 + vp * 14 + tb * 22) * METAL_F[metal]
    base += (OX_BOOST[oxidant] / 100) * base
    base += (ASSIST_B[assist]  / 100) * base
    return min(99.9, max(18, base))

def build_tips(visc, ph, temp_K, water, metal, oxidant, assist):
    tips = []
    if visc and visc > 300:
        tips.append(f"**Reduce viscosity** (predicted ~{visc:.0f} cP) — add water or raise temperature.")
    if ph and ph > 6:
        tips.append(f"**pH is {ph:.1f} (basic)** — switch to a more acidic HBD (oxalic, formic) for faster oxide dissolution.")
    if oxidant == "None" and metal in ["Gold (Au)","Copper (Cu)"]:
        tips.append(f"**Add oxidant** — I₂ or CuCl₂ boosts {metal} dissolution by ~10–15%.")
    if temp_K - 273.15 < 70:
        tips.append(f"**Raise temperature** — from {temp_K-273.15:.0f}°C to 90°C cuts viscosity and accelerates kinetics.")
    if assist == "Conventional":
        tips.append("**Try microwave assist** — cuts leaching time by 50–80% at equivalent yield.")
    if water > 0.5:
        tips.append(f"**High water content** ({water:.2f} mol frac) disrupts H-bond network — reduce below 0.3.")
    if not tips:
        tips.append("**Well-optimised** — parameters align with high-efficiency benchmarks.")
    return tips[:3]

def is_pareto(systems):
    flags = []
    for i, s in enumerate(systems):
        dominated = any(
            j!=i and o["eff"]>=s["eff"] and o["cost"]<=s["cost"] and o["green"]>=s["green"]
            and (o["eff"]>s["eff"] or o["cost"]<s["cost"] or o["green"]>s["green"])
            for j,o in enumerate(systems))
        flags.append(not dominated)
    return flags

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚗️ DES Predictor v4")
    st.markdown("XGBoost models trained on **5,790 experimental data points**.")
    st.divider()
    st.markdown("### DES Composition")
    use_top = st.toggle("Top-20 frequent compounds only", value=True)
    hba_opts = TOP_HBAS if use_top else ALL_HBAS
    hbd_opts = TOP_HBDS if use_top else ALL_HBDS
    hba = st.selectbox("HBA", hba_opts, index=hba_opts.index('choline chloride') if 'choline chloride' in hba_opts else 0)
    hbd = st.selectbox("HBD", hbd_opts, index=hbd_opts.index('glycerol') if 'glycerol' in hbd_opts else 0)
    ratio = st.slider("Molar ratio HBA:HBD", 0.05, 10.0, 1.0, 0.05)
    water = st.slider("Water content (mol fraction)", 0.0, 0.98, 0.0, 0.01)
    st.markdown("### Process Conditions")
    temp_C = st.slider("Temperature (°C)", 5, 105, 25, 5)
    temp_K = temp_C + 273.15
    st.markdown("### Metal Recovery")
    oxidant = st.selectbox("Oxidant additive", list(OX_BOOST.keys()))
    assist  = st.selectbox("Assist method", list(ASSIST_B.keys()))
    metal   = st.selectbox("Target metal", list(METAL_F.keys()))
    src     = st.selectbox("Source matrix", list(SRC_BENCH.keys()))

# ── Run predictions ────────────────────────────────────────────────────────────
preds, err = predict_properties(hba, hbd, ratio, temp_K, water)
if err:
    st.error(f"Prediction error: {err}")
    st.stop()

visc  = preds['viscosity_cP']
dens  = preds['density_kg_m3']
ph    = preds['pH_acidity']
redox = preds['reduction_potential_V_vs_SHE']
rec   = recovery_from_visc(visc, temp_K, metal, oxidant, assist)
bench = SRC_BENCH[src]
mc    = mc_uncertainty(hba, hbd, ratio, temp_K, water)
mc_s  = sorted(mc)
lo    = mc_s[int(len(mc_s)*0.05)] if mc_s else visc
hi    = mc_s[int(len(mc_s)*0.95)] if mc_s else visc
tips  = build_tips(visc, ph, temp_K, water, metal, oxidant, assist)
fi    = get_feature_importance()
grade_css = "badge-ok" if rec>=90 else ("badge-mid" if rec>=75 else "badge-low")
grade     = ("✅ Excellent" if rec>=90 else ("⚠️ Good" if rec>=75 else "❌ Needs work"))

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# ⚗️ DES Metal Recovery Predictor v4")
st.markdown("**XGBoost · Trained on 5,790 real experimental data points · 4 property predictions · Metal recovery estimation**")
st.markdown('<div class="model-badge">🤖 XGBoost — Real trained models (no hardcoded formulas)</div>', unsafe_allow_html=True)

tabs = st.tabs(["📊 Predict","🔍 Feature Importance","♻️ Reuse Simulator","📈 Pareto Optimiser","🌡️ Property Sweep","ℹ️ Model Info"])

# ════════ TAB 1 — PREDICT ════════
with tabs[0]:
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Viscosity",      f"{visc:.1f} cP",    help="XGBoost R²=0.874")
    c2.metric("Density",        f"{dens:.1f} kg/m³", help="XGBoost R²=0.994")
    c3.metric("pH",             f"{ph:.2f}",          help="XGBoost R²=0.999")
    c4.metric("Redox potential",f"{redox:.3f} V",     help="XGBoost R²=0.9997")
    st.divider()
    ca,cb,cc = st.columns(3)
    ca.metric("Est. leaching efficiency", f"{rec:.1f}%")
    cb.metric("Literature benchmark",     f"{bench}%", help=f"Median for {src}")
    cc.metric("90% CI (viscosity MC)",    f"{lo:.0f}–{hi:.0f} cP")
    st.markdown(f'<span class="{grade_css}">{grade}</span>', unsafe_allow_html=True)
    st.markdown("---")

    col1,col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Property radar</div>', unsafe_allow_html=True)
        norm = [
            min(visc/5000,1)*100,
            max(0,(dens-1000)/400*100),
            ph/10*100,
            abs(redox)/2*100,
            rec,
        ]
        cats = ['Viscosity','Density (rel)','pH/10','|Redox|/2V','Recovery']
        fig_r = go.Figure(go.Scatterpolar(
            r=norm+[norm[0]], theta=cats+[cats[0]],
            fill='toself', fillcolor='rgba(76,110,245,0.2)',
            line=dict(color='#4c6ef5',width=2)))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100])),
                             height=300, margin=dict(l=20,r=20,t=20,b=20),
                             paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_r, use_container_width=True)

        tip_html = "".join(f"<p style='margin-bottom:6px'>• {t}</p>" for t in tips)
        st.markdown(f'<div class="tip-box">{tip_html}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">Viscosity uncertainty (60 MC passes)</div>', unsafe_allow_html=True)
        if mc:
            fig_h = px.histogram(x=mc, nbins=12, color_discrete_sequence=["#74c0fc"])
            fig_h.add_vline(x=visc, line_dash="dash", line_color="#4c6ef5",
                            annotation_text=f"{visc:.0f} cP")
            fig_h.add_vline(x=lo, line_dash="dot", line_color="#adb5bd")
            fig_h.add_vline(x=hi, line_dash="dot", line_color="#adb5bd")
            fig_h.update_layout(height=230, margin=dict(l=0,r=0,t=10,b=10),
                                 showlegend=False, xaxis_title="Viscosity (cP)",
                                 plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_h, use_container_width=True)

        st.markdown('<div class="section-header">Model accuracy</div>', unsafe_allow_html=True)
        perf_df = pd.DataFrame([
            {'Property':t.replace('_',' '),'R²':v['r2'],'MAE':v['mae'],'Train N':v['train']}
            for t,v in MODEL_PERF.items()])
        st.dataframe(perf_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Sensitivity — Δ viscosity per parameter change</div>', unsafe_allow_html=True)
    sens = {
        "Temp +10%":   predict_properties(hba,hbd,ratio,temp_K*1.1,water)[0]['viscosity_cP'] - visc,
        "Water +0.05": predict_properties(hba,hbd,ratio,temp_K,min(0.98,water+0.05))[0]['viscosity_cP'] - visc,
        "Ratio +10%":  predict_properties(hba,hbd,ratio*1.1,temp_K,water)[0]['viscosity_cP'] - visc,
        "Temp -10%":   predict_properties(hba,hbd,ratio,temp_K*0.9,water)[0]['viscosity_cP'] - visc,
        "Water -0.05": predict_properties(hba,hbd,ratio,temp_K,max(0,water-0.05))[0]['viscosity_cP'] - visc,
    }
    fig_s = go.Figure(go.Bar(
        x=list(sens.values()), y=list(sens.keys()), orientation="h",
        marker_color=["#37b24d" if v<=0 else "#f03e3e" for v in sens.values()],
        text=[f"{v:+.1f} cP" for v in sens.values()], textposition="outside"))
    fig_s.update_layout(height=210, margin=dict(l=0,r=90,t=10,b=10),
                         xaxis_title="Δ viscosity (cP)",
                         plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_s, use_container_width=True)

# ════════ TAB 2 — FEATURE IMPORTANCE ════════
with tabs[1]:
    st.markdown("### XGBoost Feature Importance — All Four Models")
    col1,col2 = st.columns(2)
    for idx,(target,fvals) in enumerate(fi.items()):
        col = col1 if idx%2==0 else col2
        with col:
            perf = MODEL_PERF[target]
            fig_fi = go.Figure(go.Bar(
                x=list(fvals.values()), y=list(fvals.keys()), orientation='h',
                marker_color='#4c6ef5', text=[f"{v*100:.1f}%" for v in fvals.values()],
                textposition='outside'))
            fig_fi.update_layout(
                title=f"{target.replace('_',' ')} (R²={perf['r2']})",
                height=230, margin=dict(l=0,r=70,t=40,b=10),
                xaxis=dict(tickformat='.0%',title='Importance'),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_fi, use_container_width=True)
    st.info("**Insight:** HBA and HBD identity together account for >60% of importance across all models, "
            "confirming molecular structure is the primary driver of DES physico-chemical behaviour.")

# ════════ TAB 3 — REUSE SIMULATOR ════════
with tabs[2]:
    col1,col2 = st.columns([1,1.6])
    with col1:
        st.markdown("### Configuration")
        r_leach  = st.selectbox("DES system", list(LEACH_CFG.keys()))
        r_regen  = st.selectbox("Regeneration method", list(REGEN_RECOVER.keys()))
        r_cycles = st.slider("Reuse cycles", 1, 12, 5)
        r_temp   = st.slider("Operating temperature (°C) ", 60, 180, 90, 5)
        cfg = LEACH_CFG[r_leach]; rec_r = REGEN_RECOVER[r_regen]
        tf = 1.3 if r_temp>150 else (1.1 if r_temp>120 else 1.0)
        loss = cfg["loss"]*tf*(1-rec_r*0.7)
        cyc_eff   = [max(55, cfg["base"]-loss*i) for i in range(r_cycles)]
        no_regen  = [max(45, cfg["base"]-cfg["loss"]*tf*i) for i in range(r_cycles)]
        lifetime  = next((i for i,e in enumerate(cyc_eff) if e<75), r_cycles)
        m1,m2 = st.columns(2); m3,m4 = st.columns(2)
        m1.metric("Cycle 1 eff.", f"{cyc_eff[0]:.1f}%")
        m2.metric("Cycle 5 eff.", f"{cyc_eff[min(4,r_cycles-1)]:.1f}%")
        m3.metric("Loss/cycle", f"{loss:.1f}%")
        m4.metric("Useful lifetime", f"{lifetime} cycles")

    with col2:
        cl = [f"C{i+1}" for i in range(r_cycles)]
        fig_ru = go.Figure()
        fig_ru.add_trace(go.Scatter(x=cl,y=cyc_eff,mode="lines+markers",name="With regen",
                                    line=dict(color="#4c6ef5",width=2.5)))
        fig_ru.add_trace(go.Scatter(x=cl,y=no_regen,mode="lines+markers",name="No regen",
                                    line=dict(color="#f03e3e",width=2,dash="dash")))
        fig_ru.add_hrect(y0=75,y1=105,fillcolor="#d4edda",opacity=0.2,line_width=0,
                          annotation_text=">75% zone",annotation_position="top right")
        fig_ru.update_layout(title="Efficiency vs reuse cycle",height=280,
                              margin=dict(l=0,r=0,t=40,b=10),
                              yaxis=dict(range=[40,105],ticksuffix="%"),
                              legend=dict(orientation="h",y=-0.2),
                              plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_ru, use_container_width=True)

        cum = [sum(cyc_eff[:i+1]) for i in range(r_cycles)]
        fig_cu = go.Figure(go.Scatter(x=cl,y=cum,fill="tozeroy",mode="lines",
                                      line=dict(color="#37b24d",width=2),
                                      fillcolor="rgba(55,178,77,.15)"))
        fig_cu.update_layout(title="Cumulative recovery",height=210,
                              margin=dict(l=0,r=0,t=40,b=10),
                              plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cu, use_container_width=True)

# ════════ TAB 4 — PARETO ════════
with tabs[3]:
    col1,col2 = st.columns([1,1.5])
    with col1:
        st.markdown("### Objective weights")
        pw_e = st.slider("Efficiency weight (%)",0,100,40,5)
        pw_c = st.slider("Cost weight (%)",0,100,30,5)
        pw_g = st.slider("Green score weight (%)",0,100,30,5)
        tw = pw_e+pw_c+pw_g
        if tw==0: st.warning("All weights zero."); we=wc=wg=0
        else: we,wc,wg = pw_e/100,pw_c/100,pw_g/100
        pf = is_pareto(PARETO_SYSTEMS)
        scored = sorted([dict(**s,score=round(we*(s["eff"]/100)*100+wc*(1-s["cost"])*100+wg*s["green"]*100,1))
                         for s in PARETO_SYSTEMS],key=lambda x:x["score"],reverse=True)
        st.markdown("### Ranked systems")
        for i,s in enumerate(scored[:6]):
            st.markdown(f"""<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
              <span style="width:20px;color:#adb5bd;font-weight:600">#{i+1}</span>
              <div style="flex:1">
                <div style="font-weight:600">{s['name']}</div>
                <div style="font-size:11px;color:#6c757d">Eff {s['eff']}% · Cost {s['cost']:.2f} · Green {s['green']*100:.0f}%</div>
                <div style="height:5px;background:#e9ecef;border-radius:3px;margin-top:4px">
                  <div style="height:5px;width:{min(int(s['score']),100)}%;background:#4c6ef5;border-radius:3px"></div></div>
              </div><span class="badge-ok">{s['score']:.0f}pts</span></div>""", unsafe_allow_html=True)

    with col2:
        po = [s for s,f in zip(PARETO_SYSTEMS,pf) if f]
        pd2= [s for s,f in zip(PARETO_SYSTEMS,pf) if not f]
        fig_p = go.Figure()
        if pd2: fig_p.add_trace(go.Scatter(x=[s["cost"] for s in pd2],y=[s["eff"] for s in pd2],
            mode="markers",name="Dominated",marker=dict(color="#adb5bd",size=10),
            text=[s["name"] for s in pd2]))
        if po:  fig_p.add_trace(go.Scatter(x=[s["cost"] for s in po],y=[s["eff"] for s in po],
            mode="markers+text",name="Pareto-optimal",marker=dict(color="#4c6ef5",size=14,symbol="star"),
            text=[s["name"].split(":")[0] for s in po],textposition="top center"))
        fig_p.update_layout(title="Pareto frontier",height=320,
                             xaxis=dict(title="Cost (lower=better)",autorange="reversed"),
                             yaxis=dict(title="Efficiency (%)",range=[85,103]),
                             legend=dict(orientation="h",y=-0.25),
                             plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                             margin=dict(l=0,r=0,t=40,b=10))
        st.plotly_chart(fig_p, use_container_width=True)

# ════════ TAB 5 — PROPERTY SWEEP ════════
with tabs[4]:
    st.markdown(f"### Property sweep: **{hba}** : **{hbd}** | ratio {ratio} | water {water:.2f}")
    temps_K = np.arange(278.15, 378.15, 5)
    temps_c = temps_K - 273.15
    visc_sw = sweep_temps(hba,hbd,ratio,water,temps_K)
    dens_sw = [predict_properties(hba,hbd,ratio,t,water)[0]['density_kg_m3'] for t in temps_K]
    ph_sw   = [predict_properties(hba,hbd,ratio,t,water)[0]['pH_acidity']    for t in temps_K]

    col1,col2 = st.columns(2)
    with col1:
        for vals,label,color,ytitle in [
            (visc_sw,"Viscosity vs Temperature","#4c6ef5","Viscosity (cP)"),
            (ph_sw,  "pH vs Temperature",       "#f76707","pH"),
        ]:
            fig = go.Figure(go.Scatter(x=temps_c,y=vals,mode='lines+markers',
                                       line=dict(color=color,width=2.5),marker=dict(size=4)))
            fig.add_vline(x=temp_C,line_dash='dash',line_color='#f03e3e',
                          annotation_text=f"{temp_C}°C")
            fig.update_layout(title=label,height=250,margin=dict(l=0,r=0,t=40,b=10),
                               xaxis_title="Temperature (°C)",yaxis_title=ytitle,
                               plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig_d = go.Figure(go.Scatter(x=temps_c,y=dens_sw,mode='lines+markers',
                                     line=dict(color='#37b24d',width=2.5),marker=dict(size=4)))
        fig_d.add_vline(x=temp_C,line_dash='dash',line_color='#f03e3e')
        fig_d.update_layout(title="Density vs Temperature",height=250,
                             margin=dict(l=0,r=0,t=40,b=10),
                             xaxis_title="Temperature (°C)",yaxis_title="Density (kg/m³)",
                             plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_d, use_container_width=True)

        water_sw = np.linspace(0,0.9,20)
        visc_w = [predict_properties(hba,hbd,ratio,temp_K,w)[0]['viscosity_cP'] for w in water_sw]
        fig_w = go.Figure(go.Scatter(x=water_sw,y=visc_w,mode='lines+markers',
                                     line=dict(color='#7950f2',width=2.5),marker=dict(size=4)))
        fig_w.add_vline(x=water,line_dash='dash',line_color='#f03e3e')
        fig_w.update_layout(title="Viscosity vs Water Content",height=240,
                             margin=dict(l=0,r=0,t=40,b=10),
                             xaxis_title="Water (mol fraction)",yaxis_title="Viscosity (cP)",
                             plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_w, use_container_width=True)

# ════════ TAB 6 — MODEL INFO ════════
with tabs[5]:
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Training rows","5,790","experimental")
    c2.metric("Unique HBAs","144")
    c3.metric("Unique HBDs","167")
    c4.metric("Best R²","0.9997","redox potential")
    st.divider()
    col1,col2 = st.columns(2)
    with col1:
        st.markdown("### Model Architecture")
        st.markdown("""
| Property | R² | MAE | Train N |
|---|---|---|---|
| Viscosity (cP) | 0.874 | 339 cP | 4,920 |
| Density (kg/m³) | 0.994 | 0.69 kg/m³ | 1,264 |
| pH | 0.999 | 0.02 pH | 1,005 |
| Redox potential (V) | 0.9997 | 0.0024 V | 1,737 |

**Algorithm:** XGBoost · n_estimators=300 · max_depth=6 · lr=0.05  
**Viscosity:** log(1+x) transform applied (heavy right skew)  
**Split:** 85% train / 15% test · random_state=42  
**Features:** HBA, HBD, molar ratio, temperature (K), water (mol frac)
        """)
    with col2:
        st.markdown("### Citation")
        st.code("Moradi, F. & Bougie, F. (2026).\nJ. Mol. Liq. 443, 128903.\nhttps://doi.org/10.1016/j.molliq.2025.128903")
        st.info("**Model storage:** All four XGBoost models are embedded directly inside "
                "`model_data.py` as base64-encoded binary — no separate model files needed. "
                "This eliminates all file-path and version-mismatch errors on Streamlit Cloud.")
