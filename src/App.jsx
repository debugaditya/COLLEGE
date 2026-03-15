import { useState, useEffect, useRef } from "react";

// ─── Model (exact weights from notebook training) ─────────────────────────────
const INTERCEPT = 0.0509;
const W = {
  ca_0:1.47,ca_4:1.36,cp_3:0.92,sex_0:0.90,thal_2:0.66,thal_1:0.66,
  slope_2:0.61,cp_2:0.53,thalach:0.46,exang_0:0.30,age:0.25,restecg_1:0.17,
  fbs_1:0.16,slope_0:0.12,restecg_0:0.01,fbs_0:-0.11,cp_1:-0.13,
  restecg_2:-0.14,exang_1:-0.25,trestbps:-0.34,thal_0:-0.52,ca_3:-0.58,
  chol:-0.61,oldpeak:-0.61,slope_1:-0.68,ca_1:-0.74,thal_3:-0.76,
  sex_1:-0.85,cp_0:-1.27,ca_2:-1.46,
};
// StandardScaler stats from notebook (mean, std of 1025 patients)
const SCALER = {
  age:     { mean: 54.43, std: 9.07  },
  trestbps:{ mean: 131.61,std: 17.52 },
  chol:    { mean: 246.00,std: 51.59 },
  thalach: { mean: 149.11,std: 23.01 },
  oldpeak: { mean: 1.05,  std: 1.16  },
};
const scale  = (v, k) => (v - SCALER[k].mean) / SCALER[k].std;
const sigmoid = z => 1 / (1 + Math.exp(-z));

function predict(f) {
  return sigmoid(
    INTERCEPT
    + W.age      * scale(f.age,      "age")
    + W.thalach  * scale(f.thalach,  "thalach")
    + W.trestbps * scale(f.trestbps, "trestbps")
    + W.chol     * scale(f.chol,     "chol")
    + W.oldpeak  * scale(f.oldpeak,  "oldpeak")
    + [W.sex_0,    W.sex_1   ][f.sex]
    + [W.cp_0, W.cp_1, W.cp_2, W.cp_3][f.cp]
    + [W.fbs_0,    W.fbs_1   ][f.fbs]
    + [W.restecg_0,W.restecg_1,W.restecg_2][f.restecg]
    + [W.exang_0,  W.exang_1 ][f.exang]
    + [W.slope_0,  W.slope_1, W.slope_2][f.slope]
    + [W.ca_0, W.ca_1, W.ca_2, W.ca_3, W.ca_4][f.ca]   // ca: 0–4 per notebook
    + [W.thal_0,W.thal_1,W.thal_2,W.thal_3][f.thal]
  );
}

// Dataset means as defaults (from notebook df.describe())
const DEFAULTS = {
  age:54, sex:1, cp:1, trestbps:131, chol:246,
  fbs:0, restecg:1, thalach:149, exang:0,
  oldpeak:1.0, slope:1, ca:0, thal:2,
};

const STEPS = [
  { id:"onboard",  label:"Welcome",   icon:"💙" },
  { id:"about",    label:"About You", icon:"👤" },
  { id:"cardiac",  label:"Heart",     icon:"🫀" },
  { id:"lab",      label:"Lab Tests", icon:"🧪" },
  { id:"imaging",  label:"Imaging",   icon:"🔬" },
];

// ─── Risk dimension scores for radar (0–100) ──────────────────────────────────
function getRadarData(f, prob) {
  const p = prob * 100;
  return [
    { label: "Chest Pain",    score: [0,35,65,90][f.cp] },
    { label: "Heart Rate",    score: Math.max(0, Math.min(100, 100 - ((f.thalach - 71) / (202-71)) * 100)) },
    { label: "ST Stress",     score: Math.min(100, (f.oldpeak / 6.2) * 100) },
    { label: "Vessels",       score: (f.ca / 4) * 100 },
    { label: "Thalium",       score: [10, 70, 20, 85][f.thal] },
    { label: "BP",            score: Math.max(0, Math.min(100, ((f.trestbps - 94) / (200-94)) * 100)) },
  ];
}

function getTopFactors(f) {
  return [
    { name:"Vessel blockage",  contrib: [W.ca_0,W.ca_1,W.ca_2,W.ca_3,W.ca_4][f.ca] },
    { name:"Chest pain type",  contrib: [W.cp_0,W.cp_1,W.cp_2,W.cp_3][f.cp] },
    { name:"Thalium result",   contrib: [W.thal_0,W.thal_1,W.thal_2,W.thal_3][f.thal] },
    { name:"Sex",              contrib: [W.sex_0,W.sex_1][f.sex] },
    { name:"ST slope",         contrib: [W.slope_0,W.slope_1,W.slope_2][f.slope] },
    { name:"Max heart rate",   contrib: W.thalach * scale(f.thalach,"thalach") },
    { name:"ST depression",    contrib: W.oldpeak * scale(f.oldpeak,"oldpeak") },
    { name:"Exercise angina",  contrib: [W.exang_0,W.exang_1][f.exang] },
  ].sort((a,b) => Math.abs(b.contrib) - Math.abs(a.contrib)).slice(0,5);
}

// ─── Radar / Spider Chart SVG ─────────────────────────────────────────────────
function RadarChart({ data, color }) {
  const cx = 110, cy = 110, r = 75;
  const n = data.length;
  const angleStep = (2 * Math.PI) / n;

  const toXY = (idx, radius) => {
    const a = idx * angleStep - Math.PI / 2;
    return { x: cx + radius * Math.cos(a), y: cy + radius * Math.sin(a) };
  };

  // Grid rings
  const rings = [0.25, 0.5, 0.75, 1].map(f => {
    const pts = Array.from({ length: n }, (_, i) => toXY(i, r * f));
    return pts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ") + " Z";
  });

  // Data polygon
  const dataPts = data.map((d, i) => toXY(i, r * (d.score / 100)));
  const dataPath = dataPts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ") + " Z";

  // Axis lines
  const axes = Array.from({ length: n }, (_, i) => {
    const end = toXY(i, r);
    return { x1: cx, y1: cy, x2: end.x, y2: end.y };
  });

  // Labels
  const labels = data.map((d, i) => {
    const pt = toXY(i, r + 18);
    return { ...pt, text: d.label, score: d.score };
  });

  return (
    <svg width="220" height="220" viewBox="0 0 220 220" style={{ overflow: "visible" }}>
      {/* Grid rings */}
      {rings.map((d, i) => (
        <path key={i} d={d} fill="none" stroke="#1e2d42" strokeWidth="1" />
      ))}
      {/* Axes */}
      {axes.map((a, i) => (
        <line key={i} x1={a.x1} y1={a.y1} x2={a.x2} y2={a.y2}
          stroke="#1e2d42" strokeWidth="1" />
      ))}
      {/* Data fill */}
      <path d={dataPath} fill={`${color}22`} stroke={color} strokeWidth="2"
        strokeLinejoin="round" />
      {/* Data dots */}
      {dataPts.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r="4"
          fill={color} stroke="#0d1623" strokeWidth="2" />
      ))}
      {/* Labels */}
      {labels.map((l, i) => {
        const anchor = l.x < cx - 5 ? "end" : l.x > cx + 5 ? "start" : "middle";
        return (
          <text key={i} x={l.x} y={l.y + 4} textAnchor={anchor}
            style={{ fontSize: 9, fill: "#64748b", fontFamily: "'Sora',sans-serif" }}>
            {l.text}
          </text>
        );
      })}
    </svg>
  );
}

// ─── Reusable form atoms ──────────────────────────────────────────────────────
function OptionCard({ label, sub, selected, onClick }) {
  return (
    <button onClick={onClick} style={{
      padding: "13px 11px", borderRadius: 12, cursor: "pointer", textAlign: "left", width: "100%",
      border: selected ? "2px solid #c8a97e" : "1.5px solid #1e2d42",
      background: selected ? "rgba(200,169,126,0.1)" : "rgba(255,255,255,0.02)",
      transition: "all .16s ease", outline: "none",
      transform: selected ? "scale(1.02)" : "scale(1)",
    }}>
      <div style={{ fontSize: 13, fontWeight: 600, color: selected ? "#c8a97e" : "#94a3b8", marginBottom: sub ? 3 : 0 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: "#334155", lineHeight: 1.4 }}>{sub}</div>}
    </button>
  );
}

function SliderField({ label, id, value, onChange, min, max, step = 1, unit, hint, normalRange }) {
  const pct = ((value - min) / (max - min)) * 100;
  const inRange = normalRange ? (value >= normalRange[0] && value <= normalRange[1]) : null;
  return (
    <div style={{ marginBottom: 4 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 10 }}>
        <div>
          <label style={{ fontSize: 13, fontWeight: 600, color: "#94a3b8", letterSpacing: "0.03em" }}>{label}</label>
          {inRange !== null && (
            <span style={{
              marginLeft: 8, fontSize: 10,
              color: inRange ? "#4ade80" : "#fbbf24",
              background: inRange ? "rgba(74,222,128,.08)" : "rgba(251,191,36,.08)",
              border: `1px solid ${inRange ? "rgba(74,222,128,.2)" : "rgba(251,191,36,.2)"}`,
              borderRadius: 20, padding: "1px 7px"
            }}>{inRange ? "Normal" : "Abnormal"}</span>
          )}
        </div>
        <span style={{ fontSize: 21, fontWeight: 700, color: "#e2e8f0", fontFamily: "'Sora',sans-serif", lineHeight: 1 }}>
          {step < 1 ? value.toFixed(1) : value}
          <span style={{ fontSize: 11, color: "#475569", marginLeft: 3, fontWeight: 400 }}>{unit}</span>
        </span>
      </div>
      <div style={{ position: "relative", height: 6, borderRadius: 99 }}>
        <div style={{ position: "absolute", inset: 0, borderRadius: 99, background: "#1e2d42" }} />
        <div style={{ position: "absolute", left: 0, top: 0, height: "100%", borderRadius: 99,
          width: `${pct}%`, background: "linear-gradient(90deg,#7c9cbf,#c8a97e)", transition: "width .1s" }} />
        <input type="range" min={min} max={max} step={step} value={value}
          onChange={e => onChange(id, parseFloat(e.target.value))}
          style={{ position: "absolute", inset: 0, width: "100%", opacity: 0, cursor: "pointer", height: "100%", margin: 0 }} />
        <div style={{
          position: "absolute", top: "50%", left: `${pct}%`,
          transform: "translate(-50%,-50%)", width: 18, height: 18, borderRadius: "50%",
          background: "#c8a97e", border: "3px solid #0d1623", pointerEvents: "none",
          transition: "left .1s", boxShadow: "0 0 0 4px rgba(200,169,126,0.2)"
        }} />
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
        {hint && <span style={{ fontSize: 10, color: "#334155" }}>{hint}</span>}
        {normalRange && <span style={{ fontSize: 10, color: "#1e2d42" }}>Normal: {normalRange[0]}–{normalRange[1]} {unit}</span>}
      </div>
    </div>
  );
}

function SectionLabel({ text }) {
  return (
    <label style={{ fontSize: 13, fontWeight: 600, color: "#94a3b8", display: "block", marginBottom: 10 }}>{text}</label>
  );
}

function PageHeader({ title, sub }) {
  return (
    <div style={{ marginBottom: 22 }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, color: "#e2e8f0", letterSpacing: "-0.02em", marginBottom: 4 }}>{title}</h2>
      <p style={{ fontSize: 13, color: "#475569" }}>{sub}</p>
    </div>
  );
}

// ─── Step pages ───────────────────────────────────────────────────────────────
function StepOnboard() {
  return (
    <div style={{ textAlign: "center", padding: "4px 0 12px" }}>
      <div style={{ fontSize: 48, marginBottom: 14 }}>🫀</div>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: "#e2e8f0", marginBottom: 10 }}>Welcome to CardioSense AI</h2>
      <p style={{ fontSize: 13, color: "#64748b", lineHeight: 1.7, marginBottom: 22 }}>
        Estimates heart disease risk using a logistic regression model trained on the
        <strong style={{ color: "#c8a97e" }}> UCI Heart Disease Dataset</strong> (1,025 patients).
        Test accuracy: <strong style={{ color: "#c8a97e" }}>81.82%</strong>.
      </p>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, textAlign: "left", marginBottom: 20 }}>
        {[
          { icon: "📋", t: "4 short sections", s: "Takes ~2 minutes" },
          { icon: "📊", t: "Rich results", s: "Gauge, radar chart & risk factors" },
          { icon: "🔬", t: "UCI Dataset", s: "1,025 patients · 14 features" },
          { icon: "📄", t: "PDF report", s: "Download your results" },
        ].map(({ icon, t, s }) => (
          <div key={t} style={{ background: "rgba(255,255,255,.03)", border: "1px solid #1e2d42",
            borderRadius: 12, padding: "12px 13px", display: "flex", gap: 10 }}>
            <span style={{ fontSize: 18 }}>{icon}</span>
            <div>
              <div style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8", marginBottom: 2 }}>{t}</div>
              <div style={{ fontSize: 11, color: "#334155" }}>{s}</div>
            </div>
          </div>
        ))}
      </div>
      <div style={{ background: "rgba(200,169,126,.06)", border: "1px solid rgba(200,169,126,.18)",
        borderRadius: 12, padding: "11px 15px", fontSize: 12, color: "#94a3b8", lineHeight: 1.6 }}>
        ⚠️ Research tool only. Not a substitute for professional medical diagnosis.
      </div>
    </div>
  );
}

function StepAbout({ form, set }) {
  return (
    <div>
      <PageHeader title="About the patient" sub="Basic demographic information" />
      <SliderField label="Age" id="age" value={form.age} onChange={set} min={29} max={77}
        unit="years" hint="Dataset range: 29–77 yrs" normalRange={[29, 65]} />
      <div style={{ marginTop: 22 }}>
        <SectionLabel text="Biological Sex" />
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
          <OptionCard label="Female" sub="🍑" selected={form.sex === 0} onClick={() => set("sex", 0)} />
          <OptionCard label="Male"   sub="🍌"   selected={form.sex === 1} onClick={() => set("sex", 1)} />
        </div>
      </div>
    </div>
  );
}

function StepCardiac({ form, set }) {
  return (
    <div>
      <PageHeader title="Cardiac parameters" sub="Blood pressure, cholesterol & heart rate" />
      <SliderField label="Resting Blood Pressure" id="trestbps" value={form.trestbps} onChange={set}
        min={94} max={200} unit="mm Hg" hint="At hospital admission" normalRange={[90, 120]} />
      <div style={{ margin: "20px 0" }} />
      <SliderField label="Serum Cholesterol" id="chol" value={form.chol} onChange={set}
        min={126} max={564} unit="mg/dl" normalRange={[125, 200]} />
      <div style={{ margin: "20px 0" }} />
      <SliderField label="Max Heart Rate" id="thalach" value={form.thalach} onChange={set}
        min={71} max={202} unit="bpm" hint="During exercise stress test" normalRange={[100, 170]} />
      <div style={{ marginTop: 22 }}>
        <SectionLabel text="Chest Pain Type (cp)" />
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
          {[
            { v: 0, l: "Typical Angina",   s: "Classic cardiac chest pain" },
            { v: 1, l: "Atypical Angina",  s: "Atypical characteristics" },
            { v: 2, l: "Non-Anginal Pain", s: "Not related to heart" },
            { v: 3, l: "Asymptomatic",     s: "No chest pain symptoms" },
          ].map(o => <OptionCard key={o.v} label={o.l} sub={o.s} selected={form.cp === o.v} onClick={() => set("cp", o.v)} />)}
        </div>
      </div>
    </div>
  );
}

function StepLab({ form, set }) {
  return (
    <div>
      <PageHeader title="Lab & exercise data" sub="ECG, blood sugar and exercise tests" />
      <div style={{ marginBottom: 20 }}>
        <SectionLabel text="Fasting Blood Sugar > 120 mg/dl (fbs)" />
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
          <OptionCard label="No"  sub="FBS ≤ 120 mg/dl" selected={form.fbs === 0} onClick={() => set("fbs", 0)} />
          <OptionCard label="Yes" sub="FBS > 120 mg/dl"  selected={form.fbs === 1} onClick={() => set("fbs", 1)} />
        </div>
      </div>
      <div style={{ marginBottom: 20 }}>
        <SectionLabel text="Resting ECG (restecg)" />
        <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 8 }}>
          {[
            { v: 0, l: "Normal",            s: "No abnormalities detected" },
            { v: 1, l: "ST-T Abnormality",  s: "T-wave inversion or ST depression · most common in dataset" },
            { v: 2, l: "LV Hypertrophy",    s: "Left ventricular hypertrophy signs" },
          ].map(o => <OptionCard key={o.v} label={o.l} sub={o.s} selected={form.restecg === o.v} onClick={() => set("restecg", o.v)} />)}
        </div>
      </div>
      <div style={{ marginBottom: 20 }}>
        <SectionLabel text="Exercise-Induced Angina (exang)" />
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
          <OptionCard label="No"  sub="No chest pain during exercise" selected={form.exang === 0} onClick={() => set("exang", 0)} />
          <OptionCard label="Yes" sub="Angina triggered by exercise"  selected={form.exang === 1} onClick={() => set("exang", 1)} />
        </div>
      </div>
      <SliderField label="ST Depression (oldpeak)" id="oldpeak" value={form.oldpeak} onChange={set}
        min={0} max={6.2} step={0.1} unit="mm" hint="Relative to rest · higher = more stress" normalRange={[0, 1]} />
    </div>
  );
}

function StepImaging({ form, set }) {
  return (
    <div>
      <PageHeader title="Imaging & haematology" sub="Fluoroscopy vessels and thalium stress test" />
      <div style={{ marginBottom: 20 }}>
        <SectionLabel text="Peak Exercise ST Slope (slope)" />
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
          {[
            { v: 0, l: "Upsloping",   s: "Positive sign" },
            { v: 1, l: "Flat",        s: "Borderline" },
            { v: 2, l: "Downsloping", s: "Concerning" },
          ].map(o => <OptionCard key={o.v} label={o.l} sub={o.s} selected={form.slope === o.v} onClick={() => set("slope", o.v)} />)}
        </div>
      </div>
      <div style={{ marginBottom: 20 }}>
        <SectionLabel text="Major Vessels by Fluoroscopy (ca) — Normal: 0" />
        <div style={{ display: "grid", gridTemplateColumns: "repeat(5,1fr)", gap: 8 }}>
          {[0, 1, 2, 3, 4].map(v => (
            <OptionCard key={v} label={`${v}`} sub={["None","One","Two","Three","Four"][v]}
              selected={form.ca === v} onClick={() => set("ca", v)} />
          ))}
        </div>
      </div>
      <div>
        <SectionLabel text="Thalium Stress Test (thal)" />
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
          {[
            { v: 0, l: "No Result",         s: "Test not performed" },
            { v: 1, l: "Fixed Defect",      s: "Permanent blood flow issue" },
            { v: 2, l: "Normal",            s: "Normal blood flow" },
            { v: 3, l: "Reversible Defect", s: "Stress-induced only" },
          ].map(o => <OptionCard key={o.v} label={o.l} sub={o.s} selected={form.thal === o.v} onClick={() => set("thal", o.v)} />)}
        </div>
      </div>
    </div>
  );
}

// ─── PDF Generator ────────────────────────────────────────────────────────────
function downloadPDF(form, result, factors, radarData) {
  const pct = Math.round(result * 100);
  const label = pct < 35 ? "Low Risk" : pct < 65 ? "Moderate Risk" : "High Risk";
  const verdict = result >= 0.5 ? "Heart Disease Likely" : "No Disease Detected";
  const date = new Date().toLocaleDateString("en-GB", { day:"2-digit", month:"short", year:"numeric" });

  const cpLabels  = ["Typical Angina","Atypical Angina","Non-Anginal Pain","Asymptomatic"];
  const ecgLabels = ["Normal","ST-T Abnormality","LV Hypertrophy"];
  const slopeLabels = ["Upsloping","Flat","Downsloping"];
  const thalLabels  = ["No Result","Fixed Defect","Normal","Reversible Defect"];

  const html = `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body { font-family: Arial, sans-serif; margin: 0; padding: 40px; color: #1e293b; background: #fff; font-size: 13px; }
  .header { border-bottom: 3px solid #c8a97e; padding-bottom: 18px; margin-bottom: 24px; display: flex; justify-content: space-between; align-items: flex-end; }
  .brand { font-size: 22px; font-weight: 700; color: #0d1623; }
  .brand span { color: #c8a97e; }
  .date { font-size: 11px; color: #64748b; }
  .verdict-box { padding: 18px 22px; border-radius: 10px; margin-bottom: 24px; border-left: 5px solid ${pct<35?"#22c55e":pct<65?"#f59e0b":"#ef4444"}; background: ${pct<35?"#f0fdf4":pct<65?"#fffbeb":"#fef2f2"}; }
  .verdict-title { font-size: 20px; font-weight: 700; color: ${pct<35?"#15803d":pct<65?"#b45309":"#b91c1c"}; margin-bottom: 4px; }
  .verdict-sub { font-size: 13px; color: #64748b; }
  .prob-row { display: flex; gap: 16px; margin-bottom: 24px; }
  .prob-box { flex: 1; padding: 14px; border-radius: 8px; background: #f8fafc; border: 1px solid #e2e8f0; text-align: center; }
  .prob-val { font-size: 22px; font-weight: 700; }
  .prob-lbl { font-size: 11px; color: #64748b; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.06em; }
  h3 { font-size: 14px; font-weight: 600; color: #0d1623; margin: 20px 0 10px; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; }
  table { width: 100%; border-collapse: collapse; margin-bottom: 18px; }
  td, th { padding: 7px 10px; border: 1px solid #e2e8f0; font-size: 12px; }
  th { background: #f1f5f9; font-weight: 600; text-align: left; }
  .bar-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
  .bar-name { font-size: 12px; color: #475569; width: 140px; flex-shrink: 0; }
  .bar-track { flex: 1; height: 8px; background: #e2e8f0; border-radius: 99px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 99px; }
  .bar-val { font-size: 11px; font-weight: 600; width: 40px; text-align: right; }
  .footer { margin-top: 32px; padding-top: 16px; border-top: 1px solid #e2e8f0; font-size: 10px; color: #94a3b8; text-align: center; }
  .badge-row { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }
  .badge { padding: 6px 14px; border-radius: 20px; font-size: 11px; font-weight: 600; background: #f1f5f9; color: #475569; border: 1px solid #e2e8f0; }
</style>
</head>
<body>
<div class="header">
  <div class="brand">Cardio<span>Sense</span> AI</div>
  <div class="date">Generated: ${date}</div>
</div>

<div class="verdict-box">
  <div class="verdict-title">${verdict}</div>
  <div class="verdict-sub">${label} · ${pct}% disease probability</div>
</div>

<div class="prob-row">
  <div class="prob-box"><div class="prob-val" style="color:${result>0.5?"#dc2626":"#16a34a"}">${pct}%</div><div class="prob-lbl">Disease Risk</div></div>
  <div class="prob-box"><div class="prob-val" style="color:#2563eb">${100-pct}%</div><div class="prob-lbl">Healthy Odds</div></div>
  <div class="prob-box"><div class="prob-val" style="color:${result>=0.5?"#dc2626":"#16a34a"}">${result>=0.5?"Positive":"Negative"}</div><div class="prob-lbl">Classification</div></div>
</div>

<div class="badge-row">
  <div class="badge">📊 Model: Logistic Regression</div>
  <div class="badge">🔬 UCI Dataset · 1,025 patients</div>
  <div class="badge">🎯 Test Accuracy: 81.82%</div>
</div>

<h3>Patient Parameters</h3>
<table>
  <tr><th>Parameter</th><th>Value</th><th>Parameter</th><th>Value</th></tr>
  <tr><td>Age</td><td>${form.age} years</td><td>Sex</td><td>${form.sex===1?"Male":"Female"}</td></tr>
  <tr><td>Resting BP</td><td>${form.trestbps} mm Hg</td><td>Cholesterol</td><td>${form.chol} mg/dl</td></tr>
  <tr><td>Max Heart Rate</td><td>${form.thalach} bpm</td><td>ST Depression</td><td>${form.oldpeak.toFixed(1)} mm</td></tr>
  <tr><td>Chest Pain Type</td><td>${cpLabels[form.cp]}</td><td>Fasting Blood Sugar</td><td>${form.fbs===1?"Yes (>120 mg/dl)":"No (≤120 mg/dl)"}</td></tr>
  <tr><td>Resting ECG</td><td>${ecgLabels[form.restecg]}</td><td>Exercise Angina</td><td>${form.exang===1?"Yes":"No"}</td></tr>
  <tr><td>ST Slope</td><td>${slopeLabels[form.slope]}</td><td>Major Vessels (ca)</td><td>${form.ca}</td></tr>
  <tr><td>Thalium Test</td><td>${thalLabels[form.thal]}</td><td></td><td></td></tr>
</table>

<h3>Top Contributing Risk Factors</h3>
${factors.map(f => {
  const maxW = 1.5;
  const w = Math.abs(f.contrib);
  const pctBar = Math.round((w / maxW) * 100);
  const isRisk = f.contrib > 0;
  return `<div class="bar-row">
    <div class="bar-name">${f.name}</div>
    <div class="bar-track"><div class="bar-fill" style="width:${pctBar}%;background:${isRisk?"#ef4444":"#22c55e"}"></div></div>
    <div class="bar-val" style="color:${isRisk?"#dc2626":"#16a34a"}">${isRisk?"↑":"↓"}</div>
  </div>`;
}).join("")}

<h3>Risk Dimension Scores</h3>
<table>
  <tr>${radarData.map(d => `<th style="text-align:center">${d.label}</th>`).join("")}</tr>
  <tr>${radarData.map(d => `<td style="text-align:center;font-weight:600;color:${d.score>60?"#dc2626":d.score>35?"#d97706":"#16a34a"}">${Math.round(d.score)}%</td>`).join("")}</tr>
</table>

<div class="footer">
  CardioSense AI · Research prototype only · Not for clinical diagnosis · Consult a qualified physician<br>
  Logistic Regression · UCI Heart Disease Dataset · 1,025 patients · Test Accuracy 81.82% · Training Accuracy 89.54%
</div>
</body>
</html>`;

  const blob = new Blob([html], { type: "text/html" });
  const url  = URL.createObjectURL(blob);
  const win  = window.open(url, "_blank");
  if (win) {
    win.addEventListener("load", () => {
      setTimeout(() => { win.print(); URL.revokeObjectURL(url); }, 400);
    });
  }
}

// ─── Result Card ──────────────────────────────────────────────────────────────
function ResultCard({ result, form, onClose, onRetake }) {
  const [vis, setVis] = useState(false);
  const [animPct, setAnimPct] = useState(0);
  const pct       = Math.round(result * 100);
  const color     = pct < 35 ? "#4ade80" : pct < 65 ? "#fbbf24" : "#f87171";
  const bgAccent  = pct < 35 ? "rgba(74,222,128,.07)" : pct < 65 ? "rgba(251,191,36,.07)" : "rgba(248,113,113,.07)";
  const label     = pct < 35 ? "Low Risk" : pct < 65 ? "Moderate Risk" : "High Risk";
  const verdict   = result >= 0.5 ? "Heart Disease Likely" : "No Disease Detected";
  const circ      = Math.PI * 70;
  const factors   = getTopFactors(form);
  const radarData = getRadarData(form, result);
  const maxFactor = Math.max(...factors.map(f => Math.abs(f.contrib)));

  useEffect(() => {
    setTimeout(() => setVis(true), 10);
    let start = null;
    const go = ts => {
      if (!start) start = ts;
      const p = Math.min((ts - start) / 1100, 1);
      setAnimPct(Math.round((1 - Math.pow(1 - p, 3)) * pct));
      if (p < 1) requestAnimationFrame(go);
    };
    setTimeout(() => requestAnimationFrame(go), 200);
  }, []);

  const close = () => { setVis(false); setTimeout(onClose, 320); };

  return (
    <div style={{
      position: "fixed", inset: 0, zIndex: 1000,
      background: vis ? "rgba(5,10,20,0.8)" : "rgba(5,10,20,0)",
      backdropFilter: vis ? "blur(8px)" : "blur(0)",
      transition: "all .3s",
      display: "flex", alignItems: "flex-start", justifyContent: "center",
      padding: "24px 16px", overflowY: "auto",
    }}>
      <div style={{
        background: "#0d1623", border: "1px solid #1e2d42", borderRadius: 24,
        width: "100%", maxWidth: 500,
        boxShadow: "0 40px 100px rgba(0,0,0,.8)",
        transform: vis ? "translateY(0) scale(1)" : "translateY(30px) scale(.95)",
        opacity: vis ? 1 : 0,
        transition: "all .35s cubic-bezier(.22,1,.36,1)",
        overflow: "hidden",
      }}>
        {/* Top accent bar */}
        <div style={{ height: 4, background: `linear-gradient(90deg,${color},${color}66)` }} />

        <div style={{ padding: "26px 24px 22px" }}>
          {/* Header */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 20 }}>
            <div>
              <div style={{ fontSize: 11, color: "#334155", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 3 }}>Analysis Complete</div>
              <div style={{ fontSize: 20, fontWeight: 700, color: "#e2e8f0", letterSpacing: "-0.02em" }}>{verdict}</div>
            </div>
            <button onClick={close} style={{
              width: 30, height: 30, borderRadius: "50%", border: "1px solid #1e2d42",
              background: "#111927", color: "#475569", fontSize: 17, cursor: "pointer",
              display: "flex", alignItems: "center", justifyContent: "center",
              transition: "all .15s", flexShrink: 0,
            }}
              onMouseEnter={e => { e.currentTarget.style.background = "#1e2d42"; e.currentTarget.style.color = "#e2e8f0"; }}
              onMouseLeave={e => { e.currentTarget.style.background = "#111927"; e.currentTarget.style.color = "#475569"; }}
            >×</button>
          </div>

          {/* Gauge row */}
          <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 20,
            padding: "16px", background: bgAccent, borderRadius: 14, border: `1px solid ${color}18` }}>
            <svg width="100" height="60" viewBox="0 0 160 92" style={{ flexShrink: 0 }}>
              <path d="M14 80 A70 70 0 0 1 146 80" fill="none" stroke="#1a2638" strokeWidth="13" strokeLinecap="round" />
              <path d="M14 80 A70 70 0 0 1 146 80" fill="none" stroke={color} strokeWidth="13"
                strokeLinecap="round" strokeDasharray={`${(animPct / 100) * circ} ${circ}`} />
              <text x="80" y="76" textAnchor="middle" fill={color}
                style={{ fontSize: 30, fontWeight: 700, fontFamily: "monospace" }}>{animPct}%</text>
            </svg>
            <div style={{ flex: 1 }}>
              <div style={{ display: "inline-flex", alignItems: "center", gap: 6, padding: "4px 11px",
                borderRadius: 20, background: `${color}15`, border: `1px solid ${color}40`, marginBottom: 10 }}>
                <div style={{ width: 6, height: 6, borderRadius: "50%", background: color }} />
                <span style={{ fontSize: 12, fontWeight: 700, color }}>{label}</span>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                {[
                  { l: "Disease risk", v: `${(result * 100).toFixed(1)}%`, c: result > 0.5 ? "#f87171" : "#4ade80" },
                  { l: "Healthy odds", v: `${((1 - result) * 100).toFixed(1)}%`, c: "#60a5fa" },
                ].map(({ l, v, c }) => (
                  <div key={l} style={{ background: "rgba(0,0,0,.2)", borderRadius: 9, padding: "8px 10px" }}>
                    <div style={{ fontSize: 9, color: "#334155", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 3 }}>{l}</div>
                    <div style={{ fontSize: 16, fontWeight: 700, color: c }}>{v}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Radar + Factors side by side */}
          <div style={{ display: "grid", gridTemplateColumns: "auto 1fr", gap: 16, marginBottom: 20,
            background: "rgba(255,255,255,.02)", borderRadius: 12, padding: "16px", border: "1px solid #1a2638" }}>
            <div>
              <div style={{ fontSize: 10, color: "#475569", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 8 }}>Risk Radar</div>
              <RadarChart data={radarData} color={color} />
            </div>
            <div>
              <div style={{ fontSize: 10, color: "#475569", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 10 }}>Top Factors</div>
              {factors.map((f, i) => (
                <div key={f.name} style={{ marginBottom: 9 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                    <span style={{ fontSize: 11, color: "#94a3b8" }}>{f.name}</span>
                    <span style={{ fontSize: 10, fontWeight: 600, color: f.contrib > 0 ? "#f87171" : "#4ade80" }}>
                      {f.contrib > 0 ? "↑ Risk" : "↓ Risk"}
                    </span>
                  </div>
                  <div style={{ height: 5, borderRadius: 99, background: "#1a2638", overflow: "hidden" }}>
                    <div style={{
                      height: "100%", borderRadius: 99,
                      width: `${(Math.abs(f.contrib) / maxFactor) * 100}%`,
                      background: f.contrib > 0 ? "linear-gradient(90deg,#7f1d1d,#f87171)" : "linear-gradient(90deg,#14532d,#4ade80)",
                      transition: `width 0.9s ease ${i * 0.08}s`,
                    }} />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Trust badge */}
          <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 13px",
            background: "rgba(255,255,255,.02)", border: "1px solid #1e2d42", borderRadius: 10, marginBottom: 18 }}>
            <div style={{ width: 32, height: 32, borderRadius: 8, background: "rgba(200,169,126,.1)",
              border: "1px solid rgba(200,169,126,.2)", display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 16, flexShrink: 0 }}>🏆</div>
            <div>
              <div style={{ fontSize: 12, fontWeight: 600, color: "#c8a97e" }}>81.82% Test Accuracy</div>
              <div style={{ fontSize: 10, color: "#334155" }}>UCI Dataset · 1,025 patients · 70/30 train-test split</div>
            </div>
          </div>

          {/* Actions */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
            <button onClick={onRetake} style={{
              padding: "10px 6px", borderRadius: 10, border: "1px solid #1e2d42",
              background: "transparent", color: "#64748b", fontSize: 12, fontWeight: 600,
              cursor: "pointer", fontFamily: "inherit", transition: "all .15s",
            }}
              onMouseEnter={e => { e.currentTarget.style.borderColor = "#334155"; e.currentTarget.style.color = "#94a3b8"; }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = "#1e2d42"; e.currentTarget.style.color = "#64748b"; }}
            >Start Over</button>
            <button onClick={() => downloadPDF(form, result, factors, radarData)} style={{
              padding: "10px 6px", borderRadius: 10, border: "1px solid rgba(200,169,126,.3)",
              background: "rgba(200,169,126,.07)", color: "#c8a97e", fontSize: 12, fontWeight: 600,
              cursor: "pointer", fontFamily: "inherit", transition: "all .15s",
            }}
              onMouseEnter={e => { e.currentTarget.style.background = "rgba(200,169,126,.15)"; }}
              onMouseLeave={e => { e.currentTarget.style.background = "rgba(200,169,126,.07)"; }}
            >📄 Download PDF</button>
            <button onClick={close} style={{
              padding: "10px 6px", borderRadius: 10, border: "none",
              background: "linear-gradient(135deg,#7c9cbf,#c8a97e)",
              color: "#0d1623", fontSize: 12, fontWeight: 700,
              cursor: "pointer", fontFamily: "inherit", transition: "opacity .15s",
            }}
              onMouseEnter={e => e.currentTarget.style.opacity = ".85"}
              onMouseLeave={e => e.currentTarget.style.opacity = "1"}
            >Done</button>
          </div>

          <p style={{ textAlign: "center", fontSize: 10, color: "#1a2638", marginTop: 14, lineHeight: 1.6 }}>
            Research tool only · Not for clinical diagnosis · Consult a qualified physician
          </p>
        </div>
      </div>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [step,       setStep      ] = useState(0);
  const [dir,        setDir       ] = useState(1);
  const [animating,  setAnimating ] = useState(false);
  const [form,       setForm      ] = useState(DEFAULTS);
  const [result,     setResult    ] = useState(null);
  const [showResult, setShowResult] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const set = (id, val) => setForm(f => ({ ...f, [id]: val }));

  const navigate = next => {
    if (animating) return;
    setDir(next > step ? 1 : -1);
    setAnimating(true);
    setTimeout(() => { setStep(next); setAnimating(false); }, 260);
  };

  const handleSubmit = async () => {
    setSubmitting(true);
    await new Promise(r => setTimeout(r, 900));
    setResult(predict(form));
    setSubmitting(false);
    setShowResult(true);
  };

  const handleRetake = () => { setShowResult(false); setResult(null); setForm(DEFAULTS); setStep(0); };

  const pages = [
    <StepOnboard />,
    <StepAbout   form={form} set={set} />,
    <StepCardiac form={form} set={set} />,
    <StepLab     form={form} set={set} />,
    <StepImaging form={form} set={set} />,
  ];

  const isLast   = step === STEPS.length - 1;
  const progress = step === 0 ? 0 : ((step - 1) / (STEPS.length - 2)) * 100;

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
        *, *::before, *::after { box-sizing:border-box; margin:0; padding:0; }
        html, body { height:100%; }
        body { font-family:'Sora',sans-serif; color:#e2e8f0; background:#0d1623; }
        #root, #__next { background:#0d1623; min-height:100vh; }
        input[type=range] { -webkit-appearance:none; appearance:none; background:transparent; }
        ::-webkit-scrollbar { width:4px; }
        ::-webkit-scrollbar-track { background:#0d1623; }
        ::-webkit-scrollbar-thumb { background:#1e2d42; border-radius:2px; }

        @keyframes slideInFwd  { from{opacity:0;transform:translateX(36px)} to{opacity:1;transform:none} }
        @keyframes slideInBwd  { from{opacity:0;transform:translateX(-36px)} to{opacity:1;transform:none} }
        @keyframes slideOutFwd { from{opacity:1;transform:none} to{opacity:0;transform:translateX(-36px)} }
        @keyframes slideOutBwd { from{opacity:1;transform:none} to{opacity:0;transform:translateX(36px)} }
        @keyframes spin { to{transform:rotate(360deg)} }
        .sif { animation:slideInFwd  .26s cubic-bezier(.4,0,.2,1) both }
        .sib { animation:slideInBwd  .26s cubic-bezier(.4,0,.2,1) both }
        .sof { animation:slideOutFwd .26s cubic-bezier(.4,0,.2,1) both }
        .sob { animation:slideOutBwd .26s cubic-bezier(.4,0,.2,1) both }

        .next-btn { padding:13px 28px; border:none; border-radius:12px; cursor:pointer;
          background:#c8a97e; color:#0d1623; font-size:14px; font-weight:700;
          font-family:'Sora',sans-serif; transition:all .15s; }
        .next-btn:hover { background:#d4b98e; transform:translateY(-1px); }
        .next-btn:active { transform:scale(.98); }
        .submit-btn { padding:14px 32px; border:none; border-radius:12px; cursor:pointer;
          background:linear-gradient(135deg,#7c9cbf,#c8a97e); color:#0d1623;
          font-size:14px; font-weight:700; font-family:'Sora',sans-serif; transition:all .15s; }
        .submit-btn:hover { opacity:.9; transform:translateY(-1px); }
        .submit-btn:active { transform:scale(.98); }
        .submit-btn:disabled { opacity:.5; cursor:default; transform:none; }
        .back-btn { padding:13px 20px; border:1.5px solid #1e2d42; border-radius:12px; cursor:pointer;
          background:transparent; color:#64748b; font-size:14px; font-weight:600;
          font-family:'Sora',sans-serif; transition:all .15s; }
        .back-btn:hover { border-color:#334155; color:#94a3b8; }
      `}</style>

      <div style={{ background:"#0d1623", position:"fixed", inset:0, overflowY:"auto",
        display:"flex", flexDirection:"column", alignItems:"center",
        justifyContent:"flex-start", padding:"32px 16px 48px" }}>
        <div style={{ width:"100%", maxWidth:520 }}>

          {/* Brand */}
          <div style={{ marginBottom:26, textAlign:"center" }}>
            <div style={{ display:"inline-flex", alignItems:"center", gap:8, marginBottom:10,
              background:"rgba(200,169,126,.07)", border:"1px solid rgba(200,169,126,.15)",
              borderRadius:20, padding:"5px 14px" }}>
              <span style={{ fontSize:13 }}>🫀</span>
              <span style={{ fontSize:11, fontWeight:600, color:"#c8a97e", letterSpacing:"0.1em", textTransform:"uppercase" }}>
                CardioSense AI
              </span>
            </div>
            <h1 style={{ fontSize:"clamp(22px,4vw,30px)", fontWeight:700, letterSpacing:"-0.03em", lineHeight:1.15,
              background:"linear-gradient(135deg,#94b8d4,#c8a97e)", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }}>
              Heart Disease Risk Assessment
            </h1>
          </div>

          {/* Progress (hidden on onboard step) */}
          {step > 0 && (
            <div style={{ marginBottom:22 }}>
              <div style={{ height:3, borderRadius:99, background:"#1a2435", marginBottom:14, overflow:"hidden" }}>
                <div style={{ height:"100%", borderRadius:99,
                  background:"linear-gradient(90deg,#7c9cbf,#c8a97e)",
                  width:`${progress}%`, transition:"width .4s cubic-bezier(.4,0,.2,1)" }} />
              </div>
              <div style={{ display:"flex", justifyContent:"space-between" }}>
                {STEPS.slice(1).map((s, i) => {
                  const ri = i + 1, done = ri < step, active = ri === step;
                  return (
                    <button key={s.id} onClick={() => ri < step && navigate(ri)}
                      style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:4,
                        background:"none", border:"none",
                        cursor: ri < step ? "pointer" : "default",
                        padding:"2px 4px", opacity: ri > step ? 0.3 : 1, transition:"opacity .2s" }}>
                      <div style={{
                        width:32, height:32, borderRadius:"50%",
                        border: active ? "2px solid #c8a97e" : done ? "2px solid #4a6a8a" : "1.5px solid #1e2d42",
                        background: done ? "#4a6a8a" : active ? "rgba(200,169,126,.12)" : "#111927",
                        display:"flex", alignItems:"center", justifyContent:"center", fontSize:14,
                        boxShadow: active ? "0 0 0 4px rgba(200,169,126,.12)" : "none",
                        transition:"all .25s",
                      }}>
                        {done ? <span style={{ fontSize:12, color:"#e2e8f0" }}>✓</span> : s.icon}
                      </div>
                      <span style={{ fontSize:9, fontWeight: active?600:400,
                        color: active?"#c8a97e":done?"#64748b":"#334155",
                        letterSpacing:"0.04em", whiteSpace:"nowrap" }}>{s.label}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* Form card */}
          <div style={{ background:"#0d1623", border:"1px solid #1a2638", borderRadius:20,
            padding:"26px 22px", marginBottom:12,
            boxShadow:"0 20px 60px rgba(0,0,0,.4), inset 0 1px 0 rgba(255,255,255,.03)" }}>
            <div style={{ overflow:"hidden" }}>
              <div className={animating ? (dir>0?"sof":"sob") : (dir>0?"sif":"sib")}>
                {pages[step]}
              </div>
            </div>
          </div>

          {/* Step label */}
          <div style={{ display:"flex", justifyContent:"space-between", marginBottom:12, padding:"0 2px" }}>
            <span style={{ fontSize:11, color:"#1e2d42", fontFamily:"'JetBrains Mono',monospace" }}>
              {step === 0 ? "Get started" : `Step ${step} of ${STEPS.length - 1}`}
            </span>
            <span style={{ fontSize:11, color:"#1e2d42" }}>{STEPS[step].label}</span>
          </div>

          {/* Nav */}
          <div style={{ display:"grid", gridTemplateColumns: step>0 ? "auto 1fr" : "1fr", gap:10 }}>
            {step > 0 && <button className="back-btn" onClick={() => navigate(step - 1)}>← Back</button>}
            {isLast ? (
              <button className="submit-btn" onClick={handleSubmit} disabled={submitting}>
                {submitting
                  ? <span style={{ display:"flex", alignItems:"center", justifyContent:"center", gap:10 }}>
                      <span style={{ width:14, height:14, border:"2px solid rgba(13,22,35,.3)",
                        borderTopColor:"#0d1623", borderRadius:"50%", display:"inline-block",
                        animation:"spin .7s linear infinite" }} />
                      Analysing…
                    </span>
                  : "Get My Results →"}
              </button>
            ) : (
              <button className="next-btn" onClick={() => navigate(step + 1)}>
                {step === 0 ? "Start Assessment →" : "Continue →"}
              </button>
            )}
          </div>

        </div>
      </div>

      {showResult && result !== null && (
        <ResultCard result={result} form={form}
          onClose={() => setShowResult(false)}
          onRetake={handleRetake} />
      )}
    </>
  );
}