import { useState, useEffect, useRef } from "react";

// ─── Model ────────────────────────────────────────────────────────────────────
const INTERCEPT = 0.0509;
const W = {
  ca_0:1.47,ca_4:1.36,cp_3:0.92,sex_0:0.90,thal_2:0.66,thal_1:0.66,
  slope_2:0.61,cp_2:0.53,thalach:0.46,exang_0:0.30,age:0.25,restecg_1:0.17,
  fbs_1:0.16,slope_0:0.12,restecg_0:0.01,fbs_0:-0.11,cp_1:-0.13,
  restecg_2:-0.14,exang_1:-0.25,trestbps:-0.34,thal_0:-0.52,ca_3:-0.58,
  chol:-0.61,oldpeak:-0.61,slope_1:-0.68,ca_1:-0.74,thal_3:-0.76,
  sex_1:-0.85,cp_0:-1.27,ca_2:-1.46,
};
const sigmoid = z => 1 / (1 + Math.exp(-z));
function predict(f) {
  return sigmoid(
    INTERCEPT
    + W.age*((f.age-54)/9) + W.thalach*((f.thalach-149)/22)
    + W.trestbps*((f.trestbps-131)/18) + W.chol*((f.chol-246)/51)
    + W.oldpeak*((f.oldpeak-1.05)/1.16)
    + [W.sex_0,W.sex_1][f.sex]
    + [W.cp_0,W.cp_1,W.cp_2,W.cp_3][f.cp]
    + [W.fbs_0,W.fbs_1][f.fbs]
    + [W.restecg_0,W.restecg_1,W.restecg_2][f.restecg]
    + [W.exang_0,W.exang_1][f.exang]
    + [W.slope_0,W.slope_1,W.slope_2][f.slope]
    + [W.ca_0,W.ca_1,W.ca_2,W.ca_3][f.ca]
    + [W.thal_0,W.thal_1,W.thal_2,W.thal_3][f.thal]
  );
}

const DEFAULTS = { age:54,sex:1,cp:0,trestbps:131,chol:246,fbs:0,restecg:0,thalach:149,exang:0,oldpeak:1.0,slope:1,ca:0,thal:2 };

// ─── Steps config ─────────────────────────────────────────────────────────────
const STEPS = [
  { id:"demographics", label:"About You",     icon:"👤" },
  { id:"cardiac",      label:"Heart Profile", icon:"🫀" },
  { id:"lab",          label:"Lab Results",   icon:"🧪" },
  { id:"imaging",      label:"Imaging",       icon:"🔬" },
];

// ─── Small reusable atoms ─────────────────────────────────────────────────────
function OptionCard({ label, sub, selected, onClick }) {
  return (
    <button onClick={onClick} style={{
      padding:"14px 12px", borderRadius:12, cursor:"pointer", textAlign:"left",
      border: selected ? "2px solid #c8a97e" : "1.5px solid #2a3448",
      background: selected ? "rgba(200,169,126,0.1)" : "rgba(255,255,255,0.02)",
      transition:"all .18s cubic-bezier(.4,0,.2,1)", outline:"none",
      transform: selected ? "scale(1.02)" : "scale(1)",
    }}>
      <div style={{ fontSize:13, fontWeight:600, color: selected?"#c8a97e":"#94a3b8", marginBottom: sub?3:0, lineHeight:1.3 }}>{label}</div>
      {sub && <div style={{ fontSize:11, color:"#475569", lineHeight:1.4 }}>{sub}</div>}
    </button>
  );
}

function SliderField({ label, id, value, onChange, min, max, step=1, unit, hint }) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div style={{ marginBottom:8 }}>
      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"baseline", marginBottom:10 }}>
        <label style={{ fontSize:13, fontWeight:600, color:"#94a3b8", letterSpacing:"0.04em" }}>{label}</label>
        <span style={{ fontSize:22, fontWeight:700, color:"#e2e8f0", fontFamily:"'Sora',sans-serif" }}>
          {step < 1 ? value.toFixed(1) : value}
          <span style={{ fontSize:12, color:"#475569", marginLeft:4, fontWeight:400 }}>{unit}</span>
        </span>
      </div>
      <div style={{ position:"relative", height:6, borderRadius:99 }}>
        <div style={{ position:"absolute", inset:0, borderRadius:99, background:"#1e2d42" }} />
        <div style={{ position:"absolute", left:0, top:0, height:"100%", borderRadius:99,
          width:`${pct}%`, background:"linear-gradient(90deg,#7c9cbf,#c8a97e)",
          transition:"width .15s ease" }} />
        <input type="range" min={min} max={max} step={step} value={value}
          onChange={e => onChange(id, parseFloat(e.target.value))}
          style={{ position:"absolute", inset:0, width:"100%", opacity:0, cursor:"pointer", height:"100%", margin:0 }} />
        <div style={{
          position:"absolute", top:"50%", left:`${pct}%`,
          transform:"translate(-50%,-50%)", width:18, height:18,
          borderRadius:"50%", background:"#c8a97e", border:"3px solid #0d1623",
          pointerEvents:"none", transition:"left .15s ease",
          boxShadow:"0 0 0 4px rgba(200,169,126,0.2)"
        }} />
      </div>
      {hint && <div style={{ fontSize:11, color:"#334155", marginTop:8 }}>{hint}</div>}
    </div>
  );
}

// ─── Step pages ───────────────────────────────────────────────────────────────
function StepDemographics({ form, set }) {
  return (
    <div>
      <PageHeader title="Tell us about the patient" sub="Basic demographic information" />
      <SliderField label="Age" id="age" value={form.age} onChange={set} min={18} max={100} unit="years"
        hint="Patient age in years" />
      <div style={{ marginTop:28 }}>
        <label style={{ fontSize:13, fontWeight:600, color:"#94a3b8", letterSpacing:"0.04em", display:"block", marginBottom:12 }}>
          Biological Sex
        </label>
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:10 }}>
          <OptionCard label="Female" sub="Sex = 0" selected={form.sex===0} onClick={() => set("sex",0)} />
          <OptionCard label="Male"   sub="Sex = 1" selected={form.sex===1} onClick={() => set("sex",1)} />
        </div>
      </div>
    </div>
  );
}

function StepCardiac({ form, set }) {
  const cpOptions = [
    { v:0, l:"Typical Angina",    s:"Classic chest pain from reduced blood flow" },
    { v:1, l:"Atypical Angina",   s:"Chest pain with atypical characteristics" },
    { v:2, l:"Non-Anginal Pain",  s:"Chest pain not related to heart" },
    { v:3, l:"Asymptomatic",      s:"No chest pain symptoms present" },
  ];
  return (
    <div>
      <PageHeader title="Cardiac parameters" sub="Blood pressure, cholesterol and heart rate" />
      <SliderField label="Resting Blood Pressure" id="trestbps" value={form.trestbps} onChange={set}
        min={80} max={220} unit="mm Hg" hint="Measured at hospital admission" />
      <div style={{ marginBottom:28 }} />
      <SliderField label="Serum Cholesterol" id="chol" value={form.chol} onChange={set}
        min={100} max={500} unit="mg/dl" />
      <div style={{ marginBottom:28 }} />
      <SliderField label="Max Heart Rate Achieved" id="thalach" value={form.thalach} onChange={set}
        min={60} max={210} unit="bpm" hint="Maximum heart rate during stress test" />
      <div style={{ marginTop:28 }}>
        <label style={{ fontSize:13, fontWeight:600, color:"#94a3b8", letterSpacing:"0.04em", display:"block", marginBottom:12 }}>
          Chest Pain Type
        </label>
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:10 }}>
          {cpOptions.map(o => (
            <OptionCard key={o.v} label={o.l} sub={o.s} selected={form.cp===o.v} onClick={() => set("cp",o.v)} />
          ))}
        </div>
      </div>
    </div>
  );
}

function StepLab({ form, set }) {
  const ecgOptions = [
    { v:0, l:"Normal",            s:"No ST-T wave changes" },
    { v:1, l:"ST-T Abnormality",  s:"T-wave inversion or ST depression" },
    { v:2, l:"LV Hypertrophy",    s:"Probable or definite left ventricular hypertrophy" },
  ];
  return (
    <div>
      <PageHeader title="Lab & exercise data" sub="ECG, blood sugar and exercise test results" />
      <div style={{ marginBottom:24 }}>
        <label style={{ fontSize:13, fontWeight:600, color:"#94a3b8", letterSpacing:"0.04em", display:"block", marginBottom:12 }}>
          Fasting Blood Sugar {">"} 120 mg/dl
        </label>
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:10 }}>
          <OptionCard label="No"  sub="FBS ≤ 120 mg/dl" selected={form.fbs===0} onClick={() => set("fbs",0)} />
          <OptionCard label="Yes" sub="FBS > 120 mg/dl"  selected={form.fbs===1} onClick={() => set("fbs",1)} />
        </div>
      </div>
      <div style={{ marginBottom:24 }}>
        <label style={{ fontSize:13, fontWeight:600, color:"#94a3b8", letterSpacing:"0.04em", display:"block", marginBottom:12 }}>
          Resting ECG Results
        </label>
        <div style={{ display:"grid", gridTemplateColumns:"1fr", gap:10 }}>
          {ecgOptions.map(o => (
            <OptionCard key={o.v} label={o.l} sub={o.s} selected={form.restecg===o.v} onClick={() => set("restecg",o.v)} />
          ))}
        </div>
      </div>
      <div style={{ marginBottom:24 }}>
        <label style={{ fontSize:13, fontWeight:600, color:"#94a3b8", letterSpacing:"0.04em", display:"block", marginBottom:12 }}>
          Exercise-Induced Angina
        </label>
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:10 }}>
          <OptionCard label="No"  sub="No angina during exercise" selected={form.exang===0} onClick={() => set("exang",0)} />
          <OptionCard label="Yes" sub="Angina triggered by exercise" selected={form.exang===1} onClick={() => set("exang",1)} />
        </div>
      </div>
      <SliderField label="ST Depression (Oldpeak)" id="oldpeak" value={form.oldpeak} onChange={set}
        min={0} max={6.2} step={0.1} unit="mm" hint="ST depression induced by exercise relative to rest" />
    </div>
  );
}

function StepImaging({ form, set }) {
  const slopeOptions = [
    { v:0, l:"Upsloping",   s:"ST segment slopes upward" },
    { v:1, l:"Flat",        s:"No change in ST segment slope" },
    { v:2, l:"Downsloping", s:"ST segment slopes downward" },
  ];
  const thalOptions = [
    { v:0, l:"No Result",          s:"Test not performed" },
    { v:1, l:"Fixed Defect",       s:"Permanent blood flow defect" },
    { v:2, l:"Normal",             s:"Normal blood flow" },
    { v:3, l:"Reversible Defect",  s:"Defect under stress only" },
  ];
  return (
    <div>
      <PageHeader title="Imaging & haematology" sub="Fluoroscopy, ST slope and thalium stress test" />
      <div style={{ marginBottom:24 }}>
        <label style={{ fontSize:13, fontWeight:600, color:"#94a3b8", letterSpacing:"0.04em", display:"block", marginBottom:12 }}>
          Slope of Peak Exercise ST Segment
        </label>
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:10 }}>
          {slopeOptions.map(o => (
            <OptionCard key={o.v} label={o.l} sub={o.s} selected={form.slope===o.v} onClick={() => set("slope",o.v)} />
          ))}
        </div>
      </div>
      <div style={{ marginBottom:24 }}>
        <label style={{ fontSize:13, fontWeight:600, color:"#94a3b8", letterSpacing:"0.04em", display:"block", marginBottom:12 }}>
          Major Vessels Coloured by Fluoroscopy
        </label>
        <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:10 }}>
          {[0,1,2,3].map(v => (
            <OptionCard key={v} label={`${v}`} sub={v===0?"None":v===1?"One":v===2?"Two":"Three"} selected={form.ca===v} onClick={() => set("ca",v)} />
          ))}
        </div>
      </div>
      <div>
        <label style={{ fontSize:13, fontWeight:600, color:"#94a3b8", letterSpacing:"0.04em", display:"block", marginBottom:12 }}>
          Thalium Stress Test Result
        </label>
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:10 }}>
          {thalOptions.map(o => (
            <OptionCard key={o.v} label={o.l} sub={o.s} selected={form.thal===o.v} onClick={() => set("thal",o.v)} />
          ))}
        </div>
      </div>
    </div>
  );
}

function PageHeader({ title, sub }) {
  return (
    <div style={{ marginBottom:28 }}>
      <h2 style={{ fontSize:22, fontWeight:700, color:"#e2e8f0", letterSpacing:"-0.03em", marginBottom:4, fontFamily:"'Sora',sans-serif" }}>{title}</h2>
      <p style={{ fontSize:13, color:"#475569" }}>{sub}</p>
    </div>
  );
}

// ─── Result Card Overlay ──────────────────────────────────────────────────────
function ResultCard({ result, onClose, onRetake }) {
  const [vis, setVis] = useState(false);
  const [animPct, setAnimPct] = useState(0);
  const pct = Math.round(result * 100);
  const color = pct < 35 ? "#4ade80" : pct < 65 ? "#fbbf24" : "#f87171";
  const bgAccent = pct < 35 ? "rgba(74,222,128,.08)" : pct < 65 ? "rgba(251,191,36,.08)" : "rgba(248,113,113,.08)";
  const label = pct < 35 ? "Low Risk" : pct < 65 ? "Moderate Risk" : "High Risk";
  const verdict = result >= 0.5 ? "Heart Disease Likely" : "No Disease Detected";
  const circ = Math.PI * 70;

  useEffect(() => {
    setTimeout(() => setVis(true), 10);
    let start = null;
    const animate = ts => {
      if (!start) start = ts;
      const progress = Math.min((ts - start) / 1000, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      setAnimPct(Math.round(ease * pct));
      if (progress < 1) requestAnimationFrame(animate);
    };
    setTimeout(() => requestAnimationFrame(animate), 300);
  }, []);

  const close = () => { setVis(false); setTimeout(onClose, 320); };

  return (
    <div style={{
      position:"fixed", inset:0, zIndex:1000,
      background: vis ? "rgba(5,10,20,0.7)" : "rgba(5,10,20,0)",
      backdropFilter: vis ? "blur(6px)" : "blur(0px)",
      transition:"background .3s, backdrop-filter .3s",
      display:"flex", alignItems:"center", justifyContent:"center", padding:20,
    }}>
      <div style={{
        background:"#0d1623", border:"1px solid #1e2d42",
        borderRadius:24, width:"100%", maxWidth:440,
        padding:"32px 28px 28px",
        boxShadow:"0 40px 100px rgba(0,0,0,0.7), 0 0 0 1px rgba(200,169,126,.1)",
        transform: vis ? "translateY(0) scale(1)" : "translateY(30px) scale(.95)",
        opacity: vis ? 1 : 0,
        transition:"all .35s cubic-bezier(.22,1,.36,1)",
      }}>
        {/* Top row */}
        <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start", marginBottom:24 }}>
          <div>
            <div style={{ fontSize:11, color:"#334155", letterSpacing:"0.12em", textTransform:"uppercase", marginBottom:4 }}>Analysis Result</div>
            <div style={{ fontSize:21, fontWeight:700, color:"#e2e8f0", fontFamily:"'Sora',sans-serif", letterSpacing:"-0.02em" }}>{verdict}</div>
          </div>
          <button onClick={close} style={{
            width:32, height:32, borderRadius:"50%", border:"1px solid #1e2d42",
            background:"#111927", color:"#475569", fontSize:18, cursor:"pointer",
            display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0,
            transition:"all .15s",
          }}
            onMouseEnter={e=>{e.currentTarget.style.background="#1e2d42";e.currentTarget.style.color="#e2e8f0";}}
            onMouseLeave={e=>{e.currentTarget.style.background="#111927";e.currentTarget.style.color="#475569";}}
          >×</button>
        </div>

        {/* Gauge */}
        <div style={{ display:"flex", alignItems:"center", gap:24, marginBottom:24, padding:"20px", background:"rgba(255,255,255,.02)", borderRadius:16, border:"1px solid #1a2638" }}>
          <div style={{ position:"relative", flexShrink:0 }}>
            <svg width="110" height="65" viewBox="0 0 160 92">
              <path d="M14 80 A70 70 0 0 1 146 80" fill="none" stroke="#1a2638" strokeWidth="13" strokeLinecap="round"/>
              <path d="M14 80 A70 70 0 0 1 146 80" fill="none" stroke={color} strokeWidth="13"
                strokeLinecap="round"
                strokeDasharray={`${(animPct/100)*circ} ${circ}`}/>
              <text x="80" y="76" textAnchor="middle" fill={color}
                style={{ fontSize:30, fontWeight:700, fontFamily:"monospace" }}>{animPct}%</text>
            </svg>
          </div>
          <div>
            <div style={{ display:"inline-flex", alignItems:"center", gap:6, padding:"5px 12px", borderRadius:20,
              background:bgAccent, border:`1px solid ${color}40`, marginBottom:8 }}>
              <div style={{ width:7, height:7, borderRadius:"50%", background:color }} />
              <span style={{ fontSize:12, fontWeight:700, color }}>{label}</span>
            </div>
          </div>
        </div>

        {/* Stats */}
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:8, marginBottom:20 }}>
          {[
            { l:"P(Disease)",  v:`${(result*100).toFixed(1)}%`,   c: result>0.5?"#f87171":"#4ade80" },
            { l:"P(Healthy)",  v:`${((1-result)*100).toFixed(1)}%`, c:"#60a5fa" },
            { l:"Decision",    v: result>=0.5?"Positive":"Negative", c: result>=0.5?"#f87171":"#4ade80" },
          ].map(({l,v,c}) => (
            <div key={l} style={{ background:"#111927", border:"1px solid #1e2d42", borderRadius:12, padding:"12px 10px", textAlign:"center" }}>
              <div style={{ fontSize:9, color:"#334155", textTransform:"uppercase", letterSpacing:"0.1em", marginBottom:6 }}>{l}</div>
              <div style={{ fontSize:15, fontWeight:700, color:c }}>{v}</div>
            </div>
          ))}
        </div>

        {/* Actions */}
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:10 }}>
          <button onClick={onRetake} style={{
            padding:"11px", borderRadius:10, border:"1px solid #1e2d42",
            background:"transparent", color:"#64748b", fontSize:13, fontWeight:600,
            cursor:"pointer", fontFamily:"inherit", transition:"all .15s",
          }}
            onMouseEnter={e=>{e.currentTarget.style.borderColor="#334155";e.currentTarget.style.color="#94a3b8";}}
            onMouseLeave={e=>{e.currentTarget.style.borderColor="#1e2d42";e.currentTarget.style.color="#64748b";}}
          >Start Over</button>
          <button onClick={close} style={{
            padding:"11px", borderRadius:10, border:"none",
            background:"linear-gradient(135deg,#7c9cbf,#c8a97e)",
            color:"#0d1623", fontSize:13, fontWeight:700,
            cursor:"pointer", fontFamily:"inherit", transition:"opacity .15s",
          }}
            onMouseEnter={e=>e.currentTarget.style.opacity=".85"}
            onMouseLeave={e=>e.currentTarget.style.opacity="1"}
          >Close</button>
        </div>

        <p style={{ textAlign:"center", fontSize:10, color:"#1e2d42", marginTop:16 }}>
          Research prototype · Not for clinical use · Consult a physician
        </p>
      </div>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [step, setStep] = useState(0);
  const [dir, setDir] = useState(1); // 1=forward, -1=back
  const [animating, setAnimating] = useState(false);
  const [form, setForm] = useState(DEFAULTS);
  const [result, setResult] = useState(null);
  const [showResult, setShowResult] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const contentRef = useRef(null);

  const set = (id, val) => setForm(f => ({ ...f, [id]: val }));

  const navigate = (nextStep) => {
    if (animating) return;
    setDir(nextStep > step ? 1 : -1);
    setAnimating(true);
    setTimeout(() => {
      setStep(nextStep);
      setAnimating(false);
      contentRef.current?.scrollTo({ top: 0, behavior: "smooth" });
    }, 260);
  };

  const handleNext = () => step < STEPS.length - 1 && navigate(step + 1);
  const handleBack = () => step > 0 && navigate(step - 1);

  const handleSubmit = async () => {
    setSubmitting(true);
    await new Promise(r => setTimeout(r, 900));
    setResult(predict(form));
    setSubmitting(false);
    setShowResult(true);
  };

  const handleRetake = () => {
    setShowResult(false);
    setResult(null);
    setForm(DEFAULTS);
    setStep(0);
  };

  const stepComponents = [
    <StepDemographics form={form} set={set} />,
    <StepCardiac      form={form} set={set} />,
    <StepLab          form={form} set={set} />,
    <StepImaging      form={form} set={set} />,
  ];

  const isLast = step === STEPS.length - 1;
  const progress = ((step) / (STEPS.length - 1)) * 100;

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
        *, *::before, *::after { box-sizing:border-box; margin:0; padding:0; }
        html { height:100%; }
        body {
          font-family:'Sora',sans-serif;
          color:#e2e8f0;
          background:#0d1623;
          min-height:100vh;
          margin:0;
        }
        #root, #__next { background:#0d1623; min-height:100vh; }
        ::-webkit-scrollbar { width:4px; }
        ::-webkit-scrollbar-track { background:#0d1623; }
        ::-webkit-scrollbar-thumb { background:#1e2d42; border-radius:2px; }
        input[type=range] { -webkit-appearance:none; appearance:none; background:transparent; }
        select option { background:#0d1623; }

        @keyframes slideInFwd  { from { opacity:0; transform:translateX(40px);  } to { opacity:1; transform:translateX(0);   } }
        @keyframes slideInBwd  { from { opacity:0; transform:translateX(-40px); } to { opacity:1; transform:translateX(0);   } }
        @keyframes slideOutFwd { from { opacity:1; transform:translateX(0);     } to { opacity:0; transform:translateX(-40px); } }
        @keyframes slideOutBwd { from { opacity:1; transform:translateX(0);     } to { opacity:0; transform:translateX(40px);  } }
        @keyframes spin { to { transform:rotate(360deg); } }
        @keyframes shimmer {
          0%   { background-position:200% center; }
          100% { background-position:-200% center; }
        }

        .slide-in-fwd  { animation: slideInFwd  .26s cubic-bezier(.4,0,.2,1) both; }
        .slide-in-bwd  { animation: slideInBwd  .26s cubic-bezier(.4,0,.2,1) both; }
        .slide-out-fwd { animation: slideOutFwd .26s cubic-bezier(.4,0,.2,1) both; }
        .slide-out-bwd { animation: slideOutBwd .26s cubic-bezier(.4,0,.2,1) both; }

        .submit-btn {
          position:relative; overflow:hidden;
          padding:14px 32px; border:none; border-radius:12px; cursor:pointer;
          background: linear-gradient(135deg,#7c9cbf 0%,#c8a97e 50%,#7c9cbf 100%);
          background-size:200% auto;
          color:#080e1a; font-size:14px; font-weight:700;
          font-family:'Sora',sans-serif; letter-spacing:0.05em;
          transition:background-position .5s, transform .15s;
        }
        .submit-btn:hover { background-position:right center; transform:translateY(-1px); }
        .submit-btn:active { transform:scale(.98); }
        .submit-btn:disabled { opacity:.5; cursor:default; transform:none; }

        .next-btn {
          padding:13px 28px; border:none; border-radius:12px; cursor:pointer;
          background:#c8a97e; color:#080e1a; font-size:14px; font-weight:700;
          font-family:'Sora',sans-serif; letter-spacing:0.03em;
          transition:background .15s, transform .15s;
        }
        .next-btn:hover { background:#d4b98e; transform:translateY(-1px); }
        .next-btn:active { transform:scale(.98); }

        .back-btn {
          padding:13px 20px; border:1.5px solid #1e2d42; border-radius:12px; cursor:pointer;
          background:transparent; color:#64748b; font-size:14px; font-weight:600;
          font-family:'Sora',sans-serif; transition:all .15s;
        }
        .back-btn:hover { border-color:#334155; color:#94a3b8; }
      `}</style>

      <div style={{ background:"#0d1623", minHeight:"100vh", width:"100%", display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"flex-start", padding:"32px 16px 48px" }}>
        <div style={{ width:"100%", maxWidth:520 }}>

          {/* ── Brand header ── */}
          <div style={{ marginBottom:32, textAlign:"center" }}>
            <h1 style={{ fontSize:"clamp(26px,5vw,36px)", fontWeight:700, letterSpacing:"-0.03em", lineHeight:1.1,
              background:"linear-gradient(135deg,#94b8d4,#c8a97e)", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }}>
              CardioRisk Assessment
            </h1>
            <p style={{ fontSize:12, color:"#334155", marginTop:8 }}>Heart disease risk assessment tool</p>
          </div>

          {/* ── Step indicators ── */}
          <div style={{ marginBottom:28 }}>
            {/* Progress bar */}
            <div style={{ height:3, borderRadius:99, background:"#1a2435", marginBottom:20, overflow:"hidden" }}>
              <div style={{ height:"100%", borderRadius:99,
                background:"linear-gradient(90deg,#7c9cbf,#c8a97e)",
                width:`${progress}%`, transition:"width .4s cubic-bezier(.4,0,.2,1)" }} />
            </div>
            {/* Step dots */}
            <div style={{ display:"flex", justifyContent:"space-between" }}>
              {STEPS.map((s, i) => {
                const done = i < step;
                const active = i === step;
                return (
                  <button key={s.id} onClick={() => i <= step && navigate(i)}
                    style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:6,
                      background:"none", border:"none", cursor: i <= step ? "pointer":"default",
                      padding:"4px 8px", opacity: i > step ? 0.35 : 1, transition:"opacity .2s" }}>
                    <div style={{
                      width:36, height:36, borderRadius:"50%",
                      border: active ? "2px solid #c8a97e" : done ? "2px solid #4a6a8a" : "1.5px solid #1e2d42",
                      background: done ? "#4a6a8a" : active ? "rgba(200,169,126,.12)" : "#111927",
                      display:"flex", alignItems:"center", justifyContent:"center",
                      fontSize:16, transition:"all .25s",
                      boxShadow: active ? "0 0 0 4px rgba(200,169,126,.15)" : "none"
                    }}>
                      {done ? <span style={{ fontSize:13, color:"#e2e8f0" }}>✓</span> : s.icon}
                    </div>
                    <span style={{ fontSize:10, fontWeight:active?600:400, color:active?"#c8a97e":done?"#64748b":"#334155",
                      letterSpacing:"0.04em", whiteSpace:"nowrap" }}>{s.label}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* ── Form card ── */}
          <div style={{ background:"#0d1623", border:"1px solid #1a2638", borderRadius:20,
            padding:"28px 26px", marginBottom:16,
            boxShadow:"0 20px 60px rgba(0,0,0,.5), 0 1px 0 rgba(255,255,255,.03) inset" }}>
            <div ref={contentRef} style={{ overflow:"hidden" }}>
              <div className={animating ? (dir > 0 ? "slide-out-fwd" : "slide-out-bwd") : (dir > 0 ? "slide-in-fwd" : "slide-in-bwd")}>
                {stepComponents[step]}
              </div>
            </div>
          </div>

          {/* ── Step counter ── */}
          <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:16, padding:"0 2px" }}>
            <span style={{ fontSize:12, color:"#334155", fontFamily:"'JetBrains Mono',monospace" }}>
              Step {step + 1} of {STEPS.length}
            </span>
            <span style={{ fontSize:12, color:"#334155" }}>{STEPS[step].label}</span>
          </div>

          {/* ── Navigation ── */}
          <div style={{ display:"grid", gridTemplateColumns: step > 0 ? "auto 1fr" : "1fr", gap:10 }}>
            {step > 0 && (
              <button className="back-btn" onClick={handleBack}>← Back</button>
            )}
            {isLast ? (
              <button className="submit-btn" onClick={handleSubmit} disabled={submitting}>
                {submitting
                  ? <span style={{ display:"flex", alignItems:"center", justifyContent:"center", gap:10 }}>
                      <span style={{ width:14, height:14, border:"2px solid rgba(8,14,26,.3)", borderTopColor:"#080e1a", borderRadius:"50%", display:"inline-block", animation:"spin .7s linear infinite" }} />
                      Analysing…
                    </span>
                  : "Get Prediction →"}
              </button>
            ) : (
              <button className="next-btn" onClick={handleNext}>Continue →</button>
            )}
          </div>

          {/* ── Footer ── */}
          <p style={{ textAlign:"center", fontSize:11, color:"#1e2d42", marginTop:24 }}>
            For research purposes only · Not for clinical use
          </p>
        </div>
      </div>

      {/* ── Result overlay ── */}
      {showResult && result !== null && (
        <ResultCard
          result={result}
          onClose={() => setShowResult(false)}
          onRetake={handleRetake}
        />
      )}
    </>
  );
}