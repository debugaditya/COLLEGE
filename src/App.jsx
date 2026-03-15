import { useState, useEffect } from "react";

// ── Model ─────────────────────────────────────────────────────────────────────
const INTERCEPT = 0.0509;
const W = {
  ca_0:1.47, ca_4:1.36, cp_3:0.92, sex_0:0.90, thal_2:0.66, thal_1:0.66,
  slope_2:0.61, cp_2:0.53, thalach:0.46, exang_0:0.30, age:0.25, restecg_1:0.17,
  fbs_1:0.16, slope_0:0.12, restecg_0:0.01, fbs_0:-0.11, cp_1:-0.13,
  restecg_2:-0.14, exang_1:-0.25, trestbps:-0.34, thal_0:-0.52, ca_3:-0.58,
  chol:-0.61, oldpeak:-0.61, slope_1:-0.68, ca_1:-0.74, thal_3:-0.76,
  sex_1:-0.85, cp_0:-1.27, ca_2:-1.46,
};

const sigmoid = z => 1 / (1 + Math.exp(-z));

function predict(f) {
  return sigmoid(
    INTERCEPT
    + W.age      * ((f.age - 54) / 9)
    + W.thalach  * ((f.thalach - 149) / 22)
    + W.trestbps * ((f.trestbps - 131) / 18)
    + W.chol     * ((f.chol - 246) / 51)
    + W.oldpeak  * ((f.oldpeak - 1.05) / 1.16)
    + [W.sex_0, W.sex_1][f.sex]
    + [W.cp_0, W.cp_1, W.cp_2, W.cp_3][f.cp]
    + [W.fbs_0, W.fbs_1][f.fbs]
    + [W.restecg_0, W.restecg_1, W.restecg_2][f.restecg]
    + [W.exang_0, W.exang_1][f.exang]
    + [W.slope_0, W.slope_1, W.slope_2][f.slope]
    + [W.ca_0, W.ca_1, W.ca_2, W.ca_3][f.ca]   // ca: 0–3 per dataset
    + [W.thal_0, W.thal_1, W.thal_2, W.thal_3][f.thal]
  );
}

// ── Defaults (dataset means) ───────────────────────────────────────────────────
const DEFAULTS = {
  age: 54, sex: 1, cp: 0, trestbps: 131, chol: 246,
  fbs: 0, restecg: 0, thalach: 149, exang: 0,
  oldpeak: 1.0, slope: 1, ca: 0, thal: 2,
};

// ── Gauge SVG ──────────────────────────────────────────────────────────────────
function Gauge({ prob }) {
  const pct = Math.round(prob * 100);
  const r = 54, circ = Math.PI * r;
  const fill = (pct / 100) * circ;
  const color = pct < 35 ? "#22c55e" : pct < 65 ? "#f59e0b" : "#ef4444";
  const label = pct < 35 ? "Low Risk" : pct < 65 ? "Moderate Risk" : "High Risk";
  return (
    <div style={{ textAlign: "center" }}>
      <svg width="140" height="80" viewBox="0 0 140 80">
        <path d="M 16 70 A 54 54 0 0 1 124 70" fill="none" stroke="#1e293b" strokeWidth="10" strokeLinecap="round" />
        <path d="M 16 70 A 54 54 0 0 1 124 70" fill="none" stroke={color} strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={`${fill} ${circ}`}
          style={{ transition: "stroke-dasharray 0.7s ease" }} />
        <text x="70" y="66" textAnchor="middle" fill={color}
          style={{ fontSize: 22, fontWeight: 700, fontFamily: "monospace" }}>{pct}%</text>
      </svg>
      <div style={{ fontSize: 12, fontWeight: 600, color, marginTop: -4 }}>{label}</div>
    </div>
  );
}

// ── Field components ───────────────────────────────────────────────────────────
const labelStyle = {
  display: "block", fontSize: 11, fontWeight: 600,
  color: "#94a3b8", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 5,
};
const inputStyle = {
  width: "100%", padding: "9px 12px", borderRadius: 8,
  background: "#0f172a", border: "1px solid #1e3a5c",
  color: "#f1f5f9", fontSize: 14, outline: "none", boxSizing: "border-box",
  fontFamily: "inherit",
};

function NumInput({ label, id, value, onChange, min, max, step = 1, unit }) {
  return (
    <div>
      <label style={labelStyle}>{label}{unit && <span style={{ color: "#334155", fontWeight: 400 }}> ({unit})</span>}</label>
      <input
        type="number" min={min} max={max} step={step} value={value}
        onChange={e => onChange(id, parseFloat(e.target.value))}
        style={inputStyle}
        onFocus={e => (e.target.style.borderColor = "#3b82f6")}
        onBlur={e => (e.target.style.borderColor = "#1e3a5c")}
      />
    </div>
  );
}

function SelectInput({ label, id, value, onChange, options }) {
  return (
    <div>
      <label style={labelStyle}>{label}</label>
      <select
        value={value} onChange={e => onChange(id, parseInt(e.target.value))}
        style={{ ...inputStyle, cursor: "pointer", appearance: "none" }}
        onFocus={e => (e.target.style.borderColor = "#3b82f6")}
        onBlur={e => (e.target.style.borderColor = "#1e3a5c")}
      >
        {options.map(o => (
          <option key={o.v} value={o.v} style={{ background: "#0f172a" }}>{o.l}</option>
        ))}
      </select>
    </div>
  );
}

// ── Section ────────────────────────────────────────────────────────────────────
function Section({ title, children }) {
  return (
    <div style={{ marginBottom: 24 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
        <span style={{ fontSize: 11, fontWeight: 700, color: "#3b82f6", textTransform: "uppercase", letterSpacing: "0.1em" }}>{title}</span>
        <div style={{ flex: 1, height: "1px", background: "#1e3a5c" }} />
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        {children}
      </div>
    </div>
  );
}

// ── Result Card ────────────────────────────────────────────────────────────────
function ResultCard({ result, onClose }) {
  const [visible, setVisible] = useState(false);
  const pct = Math.round(result * 100);
  const color = pct < 35 ? "#22c55e" : pct < 65 ? "#f59e0b" : "#ef4444";
  const label = pct < 35 ? "Low Risk" : pct < 65 ? "Moderate Risk" : "High Risk";
  const circ = Math.PI * 62;

  useEffect(() => { const t = setTimeout(() => setVisible(true), 10); return () => clearTimeout(t); }, []);

  const close = () => { setVisible(false); setTimeout(onClose, 350); };

  return (
    <div style={{
      position: "fixed", inset: 0, zIndex: 1000,
      display: "flex", alignItems: "center", justifyContent: "center",
      padding: 16, pointerEvents: "none",
    }}>
      <div style={{
        background: "#0a1628",
        border: `1px solid ${color}55`,
        borderTop: `3px solid ${color}`,
        borderRadius: 20,
        padding: "28px 26px 24px",
        width: "100%", maxWidth: 420,
        boxShadow: `0 32px 80px rgba(0,0,0,0.6), 0 0 0 1px ${color}22`,
        transform: visible ? "translateY(0) scale(1)" : "translateY(-32px) scale(0.95)",
        opacity: visible ? 1 : 0,
        transition: "transform 0.4s cubic-bezier(.22,1,.36,1), opacity 0.35s ease",
        pointerEvents: "all",
      }}>

        {/* Header row */}
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 22 }}>
          <div>
            <div style={{ fontSize: 11, fontWeight: 600, color: "#475569", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 4 }}>
              Prediction Result
            </div>
            <div style={{ fontSize: 20, fontWeight: 700, color: "#f1f5f9", letterSpacing: "-0.02em" }}>
              {result >= 0.5 ? "Heart Disease Likely" : "No Disease Detected"}
            </div>
          </div>
          <button onClick={close} style={{
            width: 30, height: 30, borderRadius: "50%",
            background: "#0f172a", border: "1px solid #1e3a5c",
            color: "#475569", fontSize: 17, cursor: "pointer",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontFamily: "inherit", lineHeight: 1, flexShrink: 0, marginLeft: 12,
            transition: "background 0.15s, color 0.15s",
          }}
            onMouseEnter={e => { e.currentTarget.style.background = "#1e3a5c"; e.currentTarget.style.color = "#f1f5f9"; }}
            onMouseLeave={e => { e.currentTarget.style.background = "#0f172a"; e.currentTarget.style.color = "#475569"; }}
          >×</button>
        </div>

        {/* Gauge + badge */}
        <div style={{ display: "flex", alignItems: "center", gap: 20, marginBottom: 22 }}>
          <div style={{ position: "relative", flexShrink: 0 }}>
            <svg width="120" height="70" viewBox="0 0 160 92">
              <path d="M 18 80 A 62 62 0 0 1 142 80" fill="none" stroke="#1e293b" strokeWidth="12" strokeLinecap="round" />
              <path d="M 18 80 A 62 62 0 0 1 142 80" fill="none" stroke={color} strokeWidth="12"
                strokeLinecap="round"
                strokeDasharray={`${(pct / 100) * circ} ${circ}`}
                style={{ transition: "stroke-dasharray 0.9s cubic-bezier(.4,0,.2,1)" }} />
              <text x="80" y="74" textAnchor="middle" fill={color}
                style={{ fontSize: 28, fontWeight: 700, fontFamily: "monospace" }}>{pct}%</text>
            </svg>
          </div>
          <div style={{ flex: 1 }}>
            <div style={{
              display: "inline-block", padding: "5px 14px", borderRadius: 20, marginBottom: 10,
              background: `${color}15`, border: `1px solid ${color}40`,
              color, fontSize: 12, fontWeight: 700,
            }}>{label}</div>
          </div>
        </div>

        {/* Stats */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8, marginBottom: 18 }}>
          {[
            { l: "P(Disease)", v: `${(result * 100).toFixed(1)}%`, c: result > 0.5 ? "#ef4444" : "#22c55e" },
            { l: "P(Healthy)", v: `${((1 - result) * 100).toFixed(1)}%`, c: "#60a5fa" },
            { l: "Decision", v: result >= 0.5 ? "Positive" : "Negative", c: result >= 0.5 ? "#ef4444" : "#22c55e" },
          ].map(({ l, v, c }) => (
            <div key={l} style={{ background: "#0f172a", border: "1px solid #1e3a5c", borderRadius: 10, padding: "11px 8px", textAlign: "center" }}>
              <div style={{ fontSize: 9, color: "#475569", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 5 }}>{l}</div>
              <div style={{ fontSize: 15, fontWeight: 700, color: c }}>{v}</div>
            </div>
          ))}
        </div>

        <p style={{ fontSize: 10, color: "#1e3a5c", textAlign: "center" }}>
          Research prototype only · Not for clinical use · Consult a physician
        </p>
      </div>
    </div>
  );
}

// ── App ────────────────────────────────────────────────────────────────────────
export default function App() {
  const [form, setForm] = useState(DEFAULTS);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showCard, setShowCard] = useState(false);

  const set = (id, val) => setForm(f => ({ ...f, [id]: val }));

  const handleSubmit = async () => {
    setLoading(true);
    setShowCard(false);
    await new Promise(r => setTimeout(r, 600));
    setResult(predict(form));
    setLoading(false);
    setShowCard(true);
  };

  const handleReset = () => { setForm(DEFAULTS); setResult(null); setShowCard(false); };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #020917; font-family: 'Inter', sans-serif; }
        select option { background: #0f172a; }
        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-track { background: #020917; }
        ::-webkit-scrollbar-thumb { background: #1e3a5c; border-radius: 3px; }
        .card { background: #0a1628; border: 1px solid #1e3a5c; border-radius: 16px; }
        .btn-primary {
          width: 100%; padding: 12px; border: none; border-radius: 10px; cursor: pointer;
          background: #3b82f6; color: #fff; font-size: 14px; font-weight: 700;
          font-family: 'Inter', sans-serif; letter-spacing: 0.04em;
          transition: background 0.15s, transform 0.1s;
        }
        .btn-primary:hover { background: #2563eb; }
        .btn-primary:active { transform: scale(0.98); }
        .btn-primary:disabled { background: #1e3a5c; color: #334155; cursor: default; }
        .btn-ghost {
          padding: 12px 20px; border: 1px solid #1e3a5c; border-radius: 10px; cursor: pointer;
          background: transparent; color: #64748b; font-size: 13px; font-weight: 500;
          font-family: 'Inter', sans-serif; transition: border-color 0.15s, color 0.15s;
        }
        .btn-ghost:hover { border-color: #334155; color: #94a3b8; }
      `}</style>

      <div style={{ minHeight: "100vh", padding: "36px 16px 60px" }}>
        <div style={{ maxWidth: 620, margin: "0 auto" }}>

          {/* Header */}
          <div style={{ marginBottom: 28 }}>
            <h1 style={{ fontSize: 28, fontWeight: 700, color: "#f1f5f9", letterSpacing: "-0.02em" }}>
              Heart Disease <span style={{ color: "#3b82f6" }}>Predictor</span>
            </h1>
            <p style={{ marginTop: 6, fontSize: 13, color: "#475569", lineHeight: 1.6 }}>
              Enter patient clinical parameters.
            </p>
          </div>

          {/* Form card */}
          <div className="card" style={{ padding: "28px 24px", marginBottom: 16 }}>

            <Section title="Demographics">
              <NumInput label="Age" id="age" value={form.age} onChange={set} min={1} max={100} unit="years" />
              <SelectInput label="Sex" id="sex" value={form.sex} onChange={set} options={[
                { v: 1, l: "1 — Male" }, { v: 0, l: "0 — Female" }
              ]} />
            </Section>

            <Section title="Cardiac">
              <SelectInput label="Chest Pain Type" id="cp" value={form.cp} onChange={set} options={[
                { v: 0, l: "0 — Typical angina" }, { v: 1, l: "1 — Atypical angina" },
                { v: 2, l: "2 — Non-anginal pain" }, { v: 3, l: "3 — Asymptomatic" },
              ]} />
              <NumInput label="Resting Blood Pressure" id="trestbps" value={form.trestbps} onChange={set} min={60} max={250} unit="mm Hg" />
              <NumInput label="Serum Cholesterol" id="chol" value={form.chol} onChange={set} min={80} max={600} unit="mg/dl" />
              <NumInput label="Max Heart Rate" id="thalach" value={form.thalach} onChange={set} min={60} max={220} unit="bpm" />
            </Section>

            <Section title="Lab & Exercise">
              <SelectInput label="Fasting Blood Sugar > 120 mg/dl" id="fbs" value={form.fbs} onChange={set} options={[
                { v: 0, l: "0 — No" }, { v: 1, l: "1 — Yes" }
              ]} />
              <SelectInput label="Resting ECG" id="restecg" value={form.restecg} onChange={set} options={[
                { v: 0, l: "0 — Normal" }, { v: 1, l: "1 — ST-T wave abnormality" },
                { v: 2, l: "2 — LV hypertrophy" },
              ]} />
              <SelectInput label="Exercise-Induced Angina" id="exang" value={form.exang} onChange={set} options={[
                { v: 0, l: "0 — No" }, { v: 1, l: "1 — Yes" }
              ]} />
              <NumInput label="ST Depression" id="oldpeak" value={form.oldpeak} onChange={set} min={0} max={7} step={0.1} unit="mm" />
            </Section>

            <Section title="Imaging">
              <SelectInput label="Slope of Peak Exercise ST" id="slope" value={form.slope} onChange={set} options={[
                { v: 0, l: "0 — Upsloping" }, { v: 1, l: "1 — Flat" }, { v: 2, l: "2 — Downsloping" },
              ]} />
              <SelectInput label="Major Vessels Coloured" id="ca" value={form.ca} onChange={set} options={[
                { v: 0, l: "0" }, { v: 1, l: "1" }, { v: 2, l: "2" }, { v: 3, l: "3" },
              ]} />
              <div style={{ gridColumn: "1 / -1" }}>
                <SelectInput label="Thalium Stress Result" id="thal" value={form.thal} onChange={set} options={[
                  { v: 0, l: "0 — No result" }, { v: 1, l: "1 — Fixed defect" },
                  { v: 2, l: "2 — Normal" }, { v: 3, l: "3 — Reversible defect" },
                ]} />
              </div>
            </Section>

            {/* Buttons */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 10 }}>
              <button className="btn-primary" onClick={handleSubmit} disabled={loading}>
                {loading ? "Calculating…" : "Predict"}
              </button>
              <button className="btn-ghost" onClick={handleReset}>Reset</button>
            </div>
          </div>
        </div>
      </div>

      {/* Result Card */}
      {showCard && result !== null && (
        <ResultCard result={result} onClose={() => setShowCard(false)} />
      )}
    </>
  );
}