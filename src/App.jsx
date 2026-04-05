import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Heart, Sun, Moon, Check, ChevronRight, ChevronLeft,
  User, Activity, FileText, TrendingUp, TrendingDown,
  Award, AlertCircle, ArrowRight, Zap, Wind, Flame,
  Waves, Brain, Cigarette, Scale, Dumbbell, Users,
  ThermometerSun, Stethoscope, ShieldAlert,
} from "lucide-react";

// ─── Themes ───────────────────────────────────────────────────────────────────
const THEMES = {
  light: {
    bg: "#f8f4f0", surface: "#ffffff", card: "#ffffff",
    border: "#e8ddd4", border2: "#d4c4b4",
    text: "#1a1208", textSub: "#4a3728", textMute: "#8a6a54", textDim: "#c4a48c",
    accent: "#c2410c", accent2: "#92400e",
    grad: "linear-gradient(135deg,#c2410c,#ea580c)",
    sliderTrack: "#f0e8e0", progressBg: "#f0e8e0",
    stepDone: "#c2410c", stepActive: "rgba(194,65,12,.08)",
    stepInactive: "#f8f4f0", isLight: true,
  },
  dark: {
    bg: "#0f0d0b", surface: "#1a1612", card: "#0f0d0b",
    border: "#2a2018", border2: "#3a3028",
    text: "#f0e8e0", textSub: "#c4a48c", textMute: "#8a6a54", textDim: "#3a3028",
    accent: "#fb923c", accent2: "#f97316",
    grad: "linear-gradient(135deg,#c2410c,#fb923c)",
    sliderTrack: "#2a2018", progressBg: "#1a1612",
    stepDone: "#c2410c", stepActive: "rgba(251,146,60,.1)",
    stepInactive: "#1a1612", isLight: false,
  },
};

// ─── API ──────────────────────────────────────────────────────────────────────
const API_URL = "https://mlbackend-ww05.onrender.com/predict";

// Formats form state → backend payload (all values as floats)
function toPayload(form) {
  return {
    Chest_Pain: parseFloat(form.Chest_Pain),
    Shortness_of_Breath: parseFloat(form.Shortness_of_Breath),
    Fatigue: parseFloat(form.Fatigue),
    Palpitations: parseFloat(form.Palpitations),
    Dizziness: parseFloat(form.Dizziness),
    Swelling: parseFloat(form.Swelling),
    Pain_Arms_Jaw_Back: parseFloat(form.Pain_Arms_Jaw_Back),
    Cold_Sweats_Nausea: parseFloat(form.Cold_Sweats_Nausea),
    High_BP: parseFloat(form.High_BP),
    High_Cholesterol: parseFloat(form.High_Cholesterol),
    Diabetes: parseFloat(form.Diabetes),
    Smoking: parseFloat(form.Smoking),
    Obesity: parseFloat(form.Obesity),
    Sedentary_Lifestyle: parseFloat(form.Sedentary_Lifestyle),
    Family_History: parseFloat(form.Family_History),
    Chronic_Stress: parseFloat(form.Chronic_Stress),
    Gender: parseFloat(form.Gender),
    Age: parseFloat(form.Age),
  };
}

async function callAPI(form) {
  const res = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(toPayload(form)),
  });
  if (!res.ok) throw new Error(`API error ${res.status}`);
  return await res.json(); // { probability, prediction }
}

// Perturbation-based factor attribution — flips each binary feature,
// calls the real API for each, measures delta vs base probability
async function getTopFactors(form, baseProb) {
  const keys = [
    { key: "Chest_Pain", name: "Chest Pain" },
    { key: "Shortness_of_Breath", name: "Shortness of Breath" },
    { key: "Palpitations", name: "Palpitations" },
    { key: "High_BP", name: "High Blood Pressure" },
    { key: "High_Cholesterol", name: "High Cholesterol" },
    { key: "Family_History", name: "Family History" },
    { key: "Fatigue", name: "Fatigue" },
    { key: "Diabetes", name: "Diabetes" },
    { key: "Smoking", name: "Smoking" },
    { key: "Chronic_Stress", name: "Chronic Stress" },
  ];

  const results = await Promise.all(
    keys.map(async ({ key, name }) => {
      try {
        const tweaked = { ...form, [key]: form[key] ? 0 : 1 };
        const data = await callAPI(tweaked);
        const contrib = baseProb - data.probability;
        return { name, contrib, raw: Math.abs(contrib) };
      } catch {
        return { name, contrib: 0, raw: 0 };
      }
    })
  );

  return results.sort((a, b) => b.raw - a.raw).slice(0, 5);
}

function getRadarData(form) {
  const symCount = [form.Chest_Pain, form.Shortness_of_Breath, form.Fatigue,
  form.Palpitations, form.Dizziness, form.Swelling,
  form.Pain_Arms_Jaw_Back, form.Cold_Sweats_Nausea].reduce((a, b) => a + b, 0);
  const riskCount = [form.High_BP, form.High_Cholesterol, form.Diabetes].reduce((a, b) => a + b, 0);
  const lifeCount = [form.Smoking, form.Obesity, form.Sedentary_Lifestyle, form.Chronic_Stress].reduce((a, b) => a + b, 0);
  const ageScore = Math.max(0, Math.min(100, ((form.Age - 20) / 60) * 100));
  return [
    { label: "Symptoms", score: (symCount / 8) * 100 },
    { label: "Age Risk", score: ageScore },
    { label: "Medical", score: (riskCount / 3) * 100 },
    { label: "Lifestyle", score: (lifeCount / 4) * 100 },
    { label: "Hereditary", score: form.Family_History * 100 },
    { label: "Stress", score: form.Chronic_Stress * 100 },
  ];
}

// ─── Steps ────────────────────────────────────────────────────────────────────
const STEPS = [
  { id: "onboard", label: "Welcome", Icon: Heart },
  { id: "profile", label: "Profile", Icon: User },
  { id: "symptoms", label: "Symptoms", Icon: Activity },
  { id: "risk", label: "Risk Factors", Icon: ShieldAlert },
  { id: "lifestyle", label: "Lifestyle", Icon: Dumbbell },
];

const DEFAULTS = {
  Age: 45, Gender: 1,
  Chest_Pain: 0, Shortness_of_Breath: 0, Fatigue: 0, Palpitations: 0,
  Dizziness: 0, Swelling: 0, Pain_Arms_Jaw_Back: 0, Cold_Sweats_Nausea: 0,
  High_BP: 0, High_Cholesterol: 0, Diabetes: 0, Smoking: 0,
  Obesity: 0, Sedentary_Lifestyle: 0, Family_History: 0, Chronic_Stress: 0,
};

// ─── Animation variants ───────────────────────────────────────────────────────
const pageVariants = {
  enter: (dir) => ({ opacity: 0, x: dir > 0 ? 40 : -40, filter: "blur(4px)" }),
  center: { opacity: 1, x: 0, filter: "blur(0px)", transition: { type: "spring", stiffness: 340, damping: 30 } },
  exit: (dir) => ({ opacity: 0, x: dir > 0 ? -40 : 40, filter: "blur(4px)", transition: { duration: 0.15 } }),
};
const stagger = { hidden: {}, show: { transition: { staggerChildren: 0.07, delayChildren: 0.04 } } };
const fadeUp = { hidden: { opacity: 0, y: 16 }, show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 380, damping: 28 } } };

// ─── Shared sub-components ────────────────────────────────────────────────────
function FadeUp({ children }) {
  return <motion.div variants={fadeUp}>{children}</motion.div>;
}
function Stagger({ children, style }) {
  return <motion.div variants={stagger} initial="hidden" animate="show" style={style}>{children}</motion.div>;
}
function PageHeader({ title, sub, Icon, t }) {
  return (
    <motion.div variants={fadeUp} style={{ marginBottom: 24, display: "flex", gap: 14, alignItems: "flex-start" }}>
      <motion.div whileHover={{ rotate: 8, scale: 1.08 }}
        style={{
          width: 42, height: 42, borderRadius: 12, background: `${t.accent}14`, border: `1px solid ${t.accent}28`,
          display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, marginTop: 3
        }}>
        <Icon size={19} color={t.accent} strokeWidth={2} />
      </motion.div>
      <div>
        <h2 style={{ fontSize: 20, fontWeight: 700, color: t.text, marginBottom: 4, letterSpacing: "-0.02em" }}>{title}</h2>
        <p style={{ fontSize: 12, color: t.textMute, lineHeight: 1.5 }}>{sub}</p>
      </div>
    </motion.div>
  );
}
function Divider({ t }) {
  return <motion.div variants={fadeUp} style={{ height: 1, background: t.border, margin: "18px 0" }} />;
}

// ─── Toggle card ──────────────────────────────────────────────────────────────
function ToggleCard({ id, label, sub, icon: Icon, value, onChange, t }) {
  return (
    <motion.div variants={fadeUp} onClick={() => onChange(id, value ? 0 : 1)}
      whileHover={{ scale: 1.018 }} whileTap={{ scale: 0.97 }}
      style={{
        display: "flex", alignItems: "center", gap: 11, padding: "13px 14px", borderRadius: 12, cursor: "pointer",
        border: value ? `2px solid ${t.accent}` : `1.5px solid ${t.border}`,
        background: value ? `${t.accent}10` : `${t.accent}03`,
        transition: "border-color .15s, background .15s", position: "relative", overflow: "hidden"
      }}>
      <AnimatePresence>
        {value && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            style={{ position: "absolute", inset: 0, background: `${t.accent}06`, borderRadius: 10 }} />
        )}
      </AnimatePresence>
      {Icon && (
        <motion.div animate={{ color: value ? t.accent : t.textMute }} style={{ flexShrink: 0, position: "relative" }}>
          <Icon size={17} strokeWidth={2} />
        </motion.div>
      )}
      <div style={{ flex: 1, position: "relative" }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: value ? t.accent : t.textSub }}>{label}</div>
        {sub && <div style={{ fontSize: 11, color: t.textMute, marginTop: 1 }}>{sub}</div>}
      </div>
      <div style={{
        flexShrink: 0, width: 40, height: 22, borderRadius: 99,
        background: value ? t.accent : t.sliderTrack,
        position: "relative", transition: "background .2s", border: `1px solid ${value ? t.accent : t.border}`
      }}>
        <motion.div animate={{ x: value ? 19 : 2 }} transition={{ type: "spring", stiffness: 500, damping: 30 }}
          style={{ position: "absolute", top: 2, width: 16, height: 16, borderRadius: "50%", background: "#fff", boxShadow: "0 1px 4px rgba(0,0,0,.2)" }} />
      </div>
    </motion.div>
  );
}

// ─── Option card ──────────────────────────────────────────────────────────────
function OptionCard({ label, sub, selected, onClick, t }) {
  return (
    <motion.button variants={fadeUp} onClick={onClick} whileHover={{ scale: 1.025, y: -2 }} whileTap={{ scale: 0.97 }}
      style={{
        padding: "14px 12px", borderRadius: 12, cursor: "pointer", textAlign: "left", width: "100%",
        border: selected ? `2px solid ${t.accent}` : `1.5px solid ${t.border}`,
        background: selected ? `${t.accent}12` : `${t.accent}04`,
        outline: "none", position: "relative", overflow: "hidden"
      }}>
      <div style={{ fontSize: 13, fontWeight: 600, color: selected ? t.accent : t.textSub, marginBottom: 3 }}>{label}</div>
      <div style={{ fontSize: 11, color: t.textMute }}>{sub}</div>
    </motion.button>
  );
}

// ─── Radar Chart ──────────────────────────────────────────────────────────────
function RadarChart({ data, color }) {
  const cx = 110, cy = 110, r = 72, n = data.length;
  const ang = (i) => i * (2 * Math.PI / n) - Math.PI / 2;
  const pt = (i, rad) => ({ x: cx + rad * Math.cos(ang(i)), y: cy + rad * Math.sin(ang(i)) });
  const rings = [0.25, 0.5, 0.75, 1].map((f) => {
    const pts = Array.from({ length: n }, (_, i) => pt(i, r * f));
    return pts.map((p, i) => `${i ? "L" : "M"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ") + "Z";
  });
  const dataPts = data.map((d, i) => pt(i, r * (d.score / 100)));
  const dataPath = dataPts.map((p, i) => `${i ? "L" : "M"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ") + "Z";
  return (
    <svg width="220" height="220" viewBox="0 0 220 220" style={{ overflow: "visible" }}>
      {rings.map((d, i) => <path key={i} d={d} fill="none" stroke={color} strokeWidth=".5" strokeOpacity=".2" />)}
      {Array.from({ length: n }, (_, i) => { const e = pt(i, r); return <line key={i} x1={cx} y1={cy} x2={e.x.toFixed(1)} y2={e.y.toFixed(1)} stroke={color} strokeWidth=".5" strokeOpacity=".2" />; })}
      <motion.path d={dataPath} fill={color + "22"} stroke={color} strokeWidth="2" strokeLinejoin="round"
        initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: 1 }} transition={{ duration: 0.9 }} />
      {dataPts.map((p, i) => (
        <motion.circle key={i} cx={p.x} cy={p.y} r="4" fill={color} stroke="#fff" strokeWidth="2"
          initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: 0.6 + i * 0.08, type: "spring", stiffness: 400 }} />
      ))}
      {data.map((d, i) => {
        const l = pt(i, r + 17);
        const anchor = l.x < cx - 5 ? "end" : l.x > cx + 5 ? "start" : "middle";
        return <text key={i} x={l.x} y={l.y + 4} textAnchor={anchor} style={{ fontSize: 9, fill: "#64748b", fontFamily: "sans-serif" }}>{d.label}</text>;
      })}
    </svg>
  );
}

// ─── Beating Heart ────────────────────────────────────────────────────────────
function BeatingHeart({ color }) {
  return (
    <div style={{ position: "relative", display: "flex", alignItems: "center", justifyContent: "center", width: 48, height: 48 }}>
      <style>{`@keyframes hb{0%{transform:scale(1)}14%{transform:scale(1.35)}28%{transform:scale(1)}42%{transform:scale(1.22)}56%{transform:scale(1)}100%{transform:scale(1)}}@keyframes pr{0%{transform:scale(.75);opacity:.65}100%{transform:scale(2.5);opacity:0}}.hbr{position:absolute;inset:0;border-radius:50%;border:1.5px solid ${color};animation:pr 1.1s ease-out infinite}.hbr2{animation-delay:.4s!important}.hbi{animation:hb 1.1s ease-in-out infinite}`}</style>
      <div className="hbr" /><div className="hbr hbr2" />
      <div className="hbi" style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
        <Heart size={24} fill={color} color={color} strokeWidth={1} style={{ filter: `drop-shadow(0 0 6px ${color}88)` }} />
      </div>
    </div>
  );
}

// ─── Theme Toggle ─────────────────────────────────────────────────────────────
function ThemeToggle({ current, onChange }) {
  const t = THEMES[current], isDark = current === "dark";
  return (
    <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={() => onChange(isDark ? "light" : "dark")}
      style={{
        display: "flex", alignItems: "center", gap: 6, padding: "7px 13px", background: `${t.accent}12`,
        border: `1px solid ${t.accent}30`, borderRadius: 20, cursor: "pointer", fontFamily: "inherit",
        color: t.accent, fontSize: 11, fontWeight: 700, letterSpacing: "0.07em", textTransform: "uppercase"
      }}>
      <AnimatePresence mode="wait">
        <motion.span key={current} initial={{ rotate: -90, opacity: 0 }} animate={{ rotate: 0, opacity: 1 }} exit={{ rotate: 90, opacity: 0 }} transition={{ duration: 0.2 }}>
          {isDark ? <Sun size={13} strokeWidth={2.5} /> : <Moon size={13} strokeWidth={2.5} />}
        </motion.span>
      </AnimatePresence>
      {isDark ? "Light" : "Dark"}
    </motion.button>
  );
}

// ─── Animated counter ─────────────────────────────────────────────────────────
function AnimatedNumber({ target, suffix = "" }) {
  const [display, setDisplay] = useState(0);
  useEffect(() => {
    let start = null;
    const go = (ts) => {
      if (!start) start = ts;
      const p = Math.min((ts - start) / 1100, 1);
      const ease = 1 - Math.pow(1 - p, 3);
      setDisplay(Math.round(ease * target));
      if (p < 1) requestAnimationFrame(go);
    };
    const id = requestAnimationFrame(go);
    return () => cancelAnimationFrame(id);
  }, [target]);
  return <>{display}{suffix}</>;
}

// ─── Step 0: Onboard ──────────────────────────────────────────────────────────
function StepOnboard({ t }) {
  return (
    <Stagger>
      <FadeUp>
        <div style={{ textAlign: "center", marginBottom: 28 }}>
          <motion.div animate={{ scale: [1, 1.12, 1], rotate: [0, 5, -5, 0] }} transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut" }}
            style={{
              width: 64, height: 64, borderRadius: 20, background: `${t.accent}12`, border: `2px solid ${t.accent}28`,
              display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 18px"
            }}>
            <Heart size={30} color={t.accent} strokeWidth={2} />
          </motion.div>
          <h2 style={{ fontSize: 26, fontWeight: 700, color: t.text, marginBottom: 10 }}>Welcome to CardioSense AI</h2>
          <p style={{ fontSize: 13, color: t.textMute, lineHeight: 1.75 }}>
            A <strong style={{ color: t.accent }}>Deep Neural Network</strong> trained on 70,000 patients.<br />
            Model accuracy: <strong style={{ color: t.accent }}>99.21%</strong>
          </p>
        </div>
      </FadeUp>
      <motion.div variants={stagger} initial="hidden" animate="show"
        style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 20 }}>
        {[
          { Icon: Activity, label: "4 short sections", sub: "~2 minutes" },
          { Icon: Brain, label: "Live API backend", sub: "Real model inference" },
          { Icon: Users, label: "70,000 patients", sub: "18 clinical features" },
          { Icon: FileText, label: "Factor analysis", sub: "Perturbation attribution" },
        ].map(({ Icon, label, sub }) => (
          <motion.div key={label} variants={fadeUp} whileHover={{ y: -3, boxShadow: `0 8px 24px ${t.accent}14` }}
            style={{ background: `${t.accent}07`, border: `1px solid ${t.border}`, borderRadius: 14, padding: "14px 12px", display: "flex", gap: 10, alignItems: "flex-start" }}>
            <motion.div whileHover={{ rotate: 12 }} style={{ width: 32, height: 32, borderRadius: 9, background: `${t.accent}12`, border: `1px solid ${t.accent}20`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
              <Icon size={15} color={t.accent} strokeWidth={2} />
            </motion.div>
            <div>
              <div style={{ fontSize: 12, fontWeight: 600, color: t.textSub, marginBottom: 2 }}>{label}</div>
              <div style={{ fontSize: 11, color: t.textMute }}>{sub}</div>
            </div>
          </motion.div>
        ))}
      </motion.div>
      <FadeUp>
        <div style={{ background: `${t.accent}07`, border: `1px solid ${t.accent}22`, borderRadius: 12, padding: "11px 14px", fontSize: 12, color: t.textSub, lineHeight: 1.6, display: "flex", gap: 9, alignItems: "flex-start" }}>
          <AlertCircle size={14} color={t.accent} style={{ flexShrink: 0, marginTop: 1 }} />
          <span>Research tool only. Not a substitute for professional medical diagnosis.</span>
        </div>
      </FadeUp>
    </Stagger>
  );
}

// ─── Step 1: Profile ──────────────────────────────────────────────────────────
function StepProfile({ form, set, t }) {
  return (
    <Stagger>
      <PageHeader title="Personal Profile" sub="Your basic demographic information" Icon={User} t={t} />
      <FadeUp>
        <div style={{ background: t.surface, border: `1px solid ${t.border}`, borderRadius: 14, padding: "18px 20px", marginBottom: 10 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 14 }}>
            <label style={{ fontSize: 13, fontWeight: 600, color: t.textSub }}>Age</label>
            <motion.span key={form.Age} initial={{ y: -6, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ type: "spring", stiffness: 500 }}
              style={{ fontSize: 28, fontWeight: 700, color: t.text }}>
              {form.Age}<span style={{ fontSize: 12, color: t.textMute, marginLeft: 4 }}>years</span>
            </motion.span>
          </div>
          <div style={{ position: "relative", height: 6, borderRadius: 99 }}>
            <div style={{ position: "absolute", inset: 0, borderRadius: 99, background: t.sliderTrack }} />
            <div style={{
              position: "absolute", left: 0, top: 0, height: "100%", borderRadius: 99,
              width: `${((form.Age - 20) / 60) * 100}%`, background: t.grad, transition: "width .1s"
            }} />
            <input type="range" min={20} max={80} step={1} value={form.Age}
              onChange={(e) => set("Age", parseInt(e.target.value))}
              style={{ position: "absolute", inset: 0, width: "100%", opacity: 0, cursor: "pointer", height: "100%", margin: 0 }} />
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", marginTop: 8 }}>
            <span style={{ fontSize: 10, color: t.textMute }}>20 years</span>
            <span style={{ fontSize: 10, color: t.textMute }}>80 years</span>
          </div>
        </div>
      </FadeUp>
      <Divider t={t} />
      <FadeUp><label style={{ fontSize: 13, fontWeight: 600, color: t.textSub, display: "block", marginBottom: 10 }}>Biological Sex</label></FadeUp>
      <motion.div variants={stagger} style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
        <OptionCard label="Female" sub="Biological female" selected={form.Gender === 0} onClick={() => set("Gender", 0)} t={t} />
        <OptionCard label="Male" sub="Biological male" selected={form.Gender === 1} onClick={() => set("Gender", 1)} t={t} />
      </motion.div>
    </Stagger>
  );
}

// ─── Step 2: Symptoms ─────────────────────────────────────────────────────────
function StepSymptoms({ form, set, t }) {
  const items = [
    { id: "Chest_Pain", label: "Chest Pain", sub: "Tightness, pressure or pain", icon: Heart },
    { id: "Shortness_of_Breath", label: "Shortness of Breath", sub: "Difficulty breathing", icon: Wind },
    { id: "Fatigue", label: "Fatigue", sub: "Unusual tiredness", icon: Flame },
    { id: "Palpitations", label: "Palpitations", sub: "Irregular or racing heartbeat", icon: Activity },
    { id: "Dizziness", label: "Dizziness", sub: "Lightheadedness or faintness", icon: Waves },
    { id: "Swelling", label: "Swelling", sub: "Legs, ankles or feet", icon: ThermometerSun },
    { id: "Pain_Arms_Jaw_Back", label: "Radiating Pain", sub: "Arms, jaw or back discomfort", icon: Stethoscope },
    { id: "Cold_Sweats_Nausea", label: "Cold Sweats / Nausea", sub: "Sudden sweats or nausea", icon: ThermometerSun },
  ];
  return (
    <Stagger>
      <PageHeader title="Current Symptoms" sub="Toggle any symptoms you are currently experiencing" Icon={Activity} t={t} />
      <motion.div variants={stagger} style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
        {items.map(({ id, label, sub, icon }) => (
          <ToggleCard key={id} id={id} label={label} sub={sub} icon={icon} value={form[id]} onChange={set} t={t} />
        ))}
      </motion.div>
    </Stagger>
  );
}

// ─── Step 3: Medical Risk ─────────────────────────────────────────────────────
function StepRisk({ form, set, t }) {
  const items = [
    { id: "High_BP", label: "High Blood Pressure", sub: "BP above 130/80 mmHg", icon: Activity },
    { id: "High_Cholesterol", label: "High Cholesterol", sub: "Total > 200 mg/dL", icon: Zap },
    { id: "Diabetes", label: "Diabetes", sub: "Type 1 or Type 2 diagnosis", icon: ThermometerSun },
    { id: "Family_History", label: "Family History", sub: "Parent or sibling with HD", icon: Users },
  ];
  return (
    <Stagger>
      <PageHeader title="Medical Risk Factors" sub="Pre-existing conditions and family history" Icon={ShieldAlert} t={t} />
      <motion.div variants={stagger} style={{ display: "grid", gap: 8 }}>
        {items.map(({ id, label, sub, icon }) => (
          <ToggleCard key={id} id={id} label={label} sub={sub} icon={icon} value={form[id]} onChange={set} t={t} />
        ))}
      </motion.div>
    </Stagger>
  );
}

// ─── Step 4: Lifestyle ────────────────────────────────────────────────────────
function StepLifestyle({ form, set, t }) {
  const items = [
    { id: "Smoking", label: "Smoking", sub: "Current or recent smoker", icon: Cigarette },
    { id: "Obesity", label: "Obesity", sub: "BMI above 30 kg/m²", icon: Scale },
    { id: "Sedentary_Lifestyle", label: "Sedentary Lifestyle", sub: "< 30 min exercise per day", icon: Dumbbell },
    { id: "Chronic_Stress", label: "Chronic Stress", sub: "Persistent stress levels", icon: Brain },
  ];
  return (
    <Stagger>
      <PageHeader title="Lifestyle Factors" sub="Daily habits that affect cardiovascular health" Icon={Dumbbell} t={t} />
      <motion.div variants={stagger} style={{ display: "grid", gap: 8 }}>
        {items.map(({ id, label, sub, icon }) => (
          <ToggleCard key={id} id={id} label={label} sub={sub} icon={icon} value={form[id]} onChange={set} t={t} />
        ))}
      </motion.div>
    </Stagger>
  );
}

// ─── Result Modal ─────────────────────────────────────────────────────────────
function ResultCard({ result, factors, form, onClose, onRetake, t }) {
  const prob = result.probability;
  const pct = Math.round(prob * 100);
  const rColor = pct < 35 ? "#16a34a" : pct < 65 ? "#ca8a04" : "#dc2626";
  const label = pct < 35 ? "Low Risk" : pct < 65 ? "Moderate Risk" : "High Risk";
  const verdict = result.prediction === 1 ? "Heart Disease Likely" : "No Disease Detected";
  const circ = Math.PI * 70;
  const radarData = getRadarData(form);
  const maxF = Math.max(...factors.map((f) => Math.abs(f.contrib)), 0.001);

  return (
    <AnimatePresence>
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
        style={{
          position: "fixed", inset: 0, zIndex: 2000, background: "rgba(5,10,20,0.82)", backdropFilter: "blur(12px)",
          display: "flex", alignItems: "flex-start", justifyContent: "center", padding: "24px 16px", overflowY: "auto"
        }}>
        <motion.div
          initial={{ y: 60, scale: 0.92, opacity: 0 }} animate={{ y: 0, scale: 1, opacity: 1 }} exit={{ y: 40, scale: 0.95, opacity: 0 }}
          transition={{ type: "spring", stiffness: 280, damping: 26 }}
          style={{
            background: t.surface, border: `1px solid ${t.border2}`, borderRadius: 24, width: "100%", maxWidth: 500,
            boxShadow: "0 40px 100px rgba(0,0,0,.5)", overflow: "hidden"
          }}>

          <motion.div initial={{ scaleX: 0 }} animate={{ scaleX: 1 }} transition={{ duration: 0.7 }}
            style={{ height: 4, background: `linear-gradient(90deg,${rColor},${rColor}88)`, transformOrigin: "left" }} />

          <div style={{ padding: "26px 24px 22px" }}>
            {/* Header */}
            <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
              style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 20 }}>
              <div>
                <div style={{ fontSize: 11, color: t.textMute, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 3 }}>Analysis Complete</div>
                <div style={{ fontSize: 22, fontWeight: 700, color: t.text, letterSpacing: "-0.02em" }}>{verdict}</div>
              </div>
              <motion.button whileHover={{ scale: 1.1, rotate: 90 }} whileTap={{ scale: 0.9 }} onClick={onClose}
                style={{
                  width: 32, height: 32, borderRadius: "50%", border: `1px solid ${t.border}`, background: t.card,
                  color: t.textMute, fontSize: 18, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center"
                }}>×</motion.button>
            </motion.div>

            {/* Gauge */}
            <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.18, type: "spring" }}
              style={{
                display: "flex", alignItems: "center", gap: 16, marginBottom: 20, padding: 16,
                background: `${rColor}08`, borderRadius: 14, border: `1px solid ${rColor}20`
              }}>
              <svg width="100" height="60" viewBox="0 0 160 92" style={{ flexShrink: 0 }}>
                <path d="M14 80 A70 70 0 0 1 146 80" fill="none" stroke={t.sliderTrack} strokeWidth="13" strokeLinecap="round" />
                <motion.path d="M14 80 A70 70 0 0 1 146 80" fill="none" stroke={rColor} strokeWidth="13" strokeLinecap="round"
                  initial={{ strokeDasharray: `0 ${circ}` }}
                  animate={{ strokeDasharray: `${(pct / 100) * circ} ${circ}` }}
                  transition={{ duration: 1.1, delay: 0.3, ease: "easeOut" }} />
                <text x="80" y="76" textAnchor="middle" fill={rColor} style={{ fontSize: 28, fontWeight: 700, fontFamily: "monospace" }}>
                  <AnimatedNumber target={pct} suffix="%" />
                </text>
              </svg>
              <div style={{ flex: 1 }}>
                <motion.div initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.4 }}
                  style={{
                    display: "inline-flex", alignItems: "center", gap: 6, padding: "4px 11px", borderRadius: 20,
                    background: `${rColor}12`, border: `1px solid ${rColor}35`, marginBottom: 10
                  }}>
                  <motion.div animate={{ scale: [1, 1.4, 1] }} transition={{ repeat: Infinity, duration: 1.4 }}
                    style={{ width: 6, height: 6, borderRadius: "50%", background: rColor }} />
                  <span style={{ fontSize: 12, fontWeight: 700, color: rColor }}>{label}</span>
                </motion.div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                  {[
                    { l: "Disease risk", v: `${(prob * 100).toFixed(1)}%`, c: result.prediction === 1 ? "#dc2626" : "#16a34a" },
                    { l: "Healthy odds", v: `${((1 - prob) * 100).toFixed(1)}%`, c: t.accent2 },
                  ].map(({ l, v, c }, i) => (
                    <motion.div key={l} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 + i * 0.1 }}
                      style={{ background: `${t.accent}09`, borderRadius: 9, padding: "8px 10px" }}>
                      <div style={{ fontSize: 9, color: t.textMute, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 3 }}>{l}</div>
                      <div style={{ fontSize: 16, fontWeight: 700, color: c }}>{v}</div>
                    </motion.div>
                  ))}
                </div>
              </div>
            </motion.div>

            {/* Radar + Top Factors */}
            <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.35 }}
              style={{
                display: "grid", gridTemplateColumns: "auto 1fr", gap: 16, marginBottom: 20,
                background: `${t.accent}05`, borderRadius: 12, padding: 16, border: `1px solid ${t.border}`
              }}>
              <div>
                <div style={{ fontSize: 10, color: t.textMute, textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 8 }}>Risk Radar</div>
                <RadarChart data={radarData} color={t.accent} />
              </div>
              <div>
                <div style={{ fontSize: 10, color: t.textMute, textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 10 }}>Top Factors</div>
                {factors.map((f, i) => (
                  <motion.div key={f.name} initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.45 + i * 0.09 }} style={{ marginBottom: 10 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                      <span style={{ fontSize: 11, color: t.textSub }}>{f.name}</span>
                      <span style={{ display: "flex", alignItems: "center", gap: 3, fontSize: 10, fontWeight: 600, color: f.contrib > 0 ? "#dc2626" : "#16a34a" }}>
                        {f.contrib > 0 ? <TrendingUp size={10} strokeWidth={2.5} /> : <TrendingDown size={10} strokeWidth={2.5} />}
                        {f.contrib > 0 ? "Risk" : "Healthy"}
                      </span>
                    </div>
                    <div style={{ height: 5, borderRadius: 99, background: t.sliderTrack, overflow: "hidden" }}>
                      <motion.div initial={{ width: 0 }} animate={{ width: `${(Math.abs(f.contrib) / maxF) * 100}%` }}
                        transition={{ delay: 0.5 + i * 0.09, duration: 0.7 }}
                        style={{
                          height: "100%", borderRadius: 99,
                          background: f.contrib > 0 ? "linear-gradient(90deg,#7f1d1d,#f87171)" : "linear-gradient(90deg,#14532d,#4ade80)"
                        }} />
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>

            {/* Trust badge */}
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.7 }}
              style={{
                display: "flex", alignItems: "center", gap: 10, padding: "10px 13px",
                background: `${t.accent}07`, border: `1px solid ${t.border}`, borderRadius: 10, marginBottom: 18
              }}>
              <Award size={16} color={t.accent} strokeWidth={2} />
              <div>
                <div style={{ fontSize: 12, fontWeight: 600, color: t.accent }}>99.21% Test Accuracy</div>
                <div style={{ fontSize: 10, color: t.textMute }}>Live API inference · 70,000 patients · Perturbation attribution</div>
              </div>
            </motion.div>

            {/* Actions */}
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.8 }}
              style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              <motion.button whileHover={{ scale: 1.04, y: -1 }} whileTap={{ scale: 0.96 }} onClick={onRetake}
                style={{
                  padding: "11px 6px", borderRadius: 10, border: `1px solid ${t.border}`, background: "transparent",
                  color: t.textMute, fontSize: 12, fontWeight: 600, cursor: "pointer", fontFamily: "inherit",
                  display: "flex", alignItems: "center", justifyContent: "center", gap: 5
                }}>
                <Zap size={12} strokeWidth={2} /> Start Over
              </motion.button>
              <motion.button whileHover={{ scale: 1.04, y: -1, opacity: 0.88 }} whileTap={{ scale: 0.96 }} onClick={onClose}
                style={{
                  padding: "11px 6px", borderRadius: 10, border: "none", background: t.grad, color: "#fff",
                  fontSize: 12, fontWeight: 700, cursor: "pointer", fontFamily: "inherit",
                  display: "flex", alignItems: "center", justifyContent: "center", gap: 5
                }}>
                <Check size={13} strokeWidth={3} /> Done
              </motion.button>
            </motion.div>

            <p style={{ textAlign: "center", fontSize: 10, color: t.textDim, marginTop: 14, lineHeight: 1.6 }}>
              Research tool only · Not for clinical diagnosis · Consult a qualified physician
            </p>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}

// ─── Error Banner ─────────────────────────────────────────────────────────────
function ErrorBanner({ message, onDismiss }) {
  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 20 }}
      style={{
        position: "fixed", bottom: 90, left: "50%", transform: "translateX(-50%)", zIndex: 3000,
        background: "#dc2626", color: "#fff", borderRadius: 12, padding: "12px 18px", fontSize: 13, fontWeight: 600,
        maxWidth: 400, width: "90%", display: "flex", alignItems: "center", gap: 10, boxShadow: "0 8px 32px rgba(220,38,38,.4)"
      }}>
      <AlertCircle size={16} style={{ flexShrink: 0 }} />
      <span style={{ flex: 1 }}>{message}</span>
      <button onClick={onDismiss} style={{ background: "none", border: "none", color: "#fff", cursor: "pointer", fontSize: 18, lineHeight: 1 }}>×</button>
    </motion.div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [themeKey, setThemeKey] = useState("light");
  const [step, setStep] = useState(0);
  const [dir, setDir] = useState(1);
  const [form, setForm] = useState(DEFAULTS);
  const [result, setResult] = useState(null);   // { probability, prediction }
  const [factors, setFactors] = useState([]);
  const [showResult, setShowResult] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const t = THEMES[themeKey];
  const set = useCallback((id, val) => setForm((f) => ({ ...f, [id]: val })), []);
  const navigate = (next) => { setDir(next > step ? 1 : -1); setStep(next); };

  const handleSubmit = async () => {
    setSubmitting(true);
    setError(null);
    try {
      // 1. Main prediction from backend
      const data = await callAPI(form);                              // { probability, prediction }

      // 2. Parallel perturbation calls for top factor attribution
      const topFactors = await getTopFactors(form, data.probability);

      setResult(data);
      setFactors(topFactors);
      setShowResult(true);
    } catch (err) {
      setError("Could not reach the backend. Check your connection and try again.");
    } finally {
      setSubmitting(false);
    }
  };

  const handleRetake = () => {
    setShowResult(false);
    setResult(null);
    setFactors([]);
    setForm(DEFAULTS);
    setStep(0);
  };

  const pages = [
    <StepOnboard t={t} />,
    <StepProfile form={form} set={set} t={t} />,
    <StepSymptoms form={form} set={set} t={t} />,
    <StepRisk form={form} set={set} t={t} />,
    <StepLifestyle form={form} set={set} t={t} />,
  ];

  const isLast = step === STEPS.length - 1;
  const progress = step === 0 ? 0 : ((step - 1) / (STEPS.length - 2)) * 100;

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
        *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
        html,body{height:100%}
        body{font-family:'Sora',sans-serif}
        input[type=range]{-webkit-appearance:none;appearance:none;background:transparent}
        ::-webkit-scrollbar{width:4px}
        ::-webkit-scrollbar-thumb{background:${t.border2};border-radius:2px}
      `}</style>

      <motion.div animate={{ background: t.bg }} transition={{ duration: 0.4 }}
        style={{ position: "fixed", inset: 0, display: "flex", flexDirection: "column" }}>

        {/* Top bar */}
        <motion.div initial={{ y: -56, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ type: "spring", stiffness: 320, damping: 28 }}
          style={{
            flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "space-between",
            padding: "14px 24px", borderBottom: `1px solid ${t.border}`
          }}>
          <motion.div whileHover={{ scale: 1.04 }} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <motion.div animate={{ scale: [1, 1.18, 1] }} transition={{ duration: 1.6, repeat: Infinity, ease: "easeInOut" }}>
              <Heart size={15} color={t.accent} strokeWidth={2.5} fill={t.accent} />
            </motion.div>
            <span style={{ fontSize: 13, fontWeight: 700, color: t.accent, letterSpacing: "0.08em", textTransform: "uppercase" }}>CardioSense AI</span>
          </motion.div>
          <AnimatePresence mode="wait">
            {step > 0 && (
              <motion.span key="step" initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.8 }}
                style={{ fontSize: 11, color: t.textMute, fontFamily: "'JetBrains Mono',monospace" }}>
                Step {step} / {STEPS.length - 1}
              </motion.span>
            )}
          </AnimatePresence>
          <ThemeToggle current={themeKey} onChange={setThemeKey} />
        </motion.div>

        {/* Progress stepper */}
        <AnimatePresence>
          {step > 0 && (
            <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }}
              style={{ flexShrink: 0, overflow: "hidden" }}>
              <div style={{ padding: "12px 24px 0" }}>
                <div style={{ height: 3, borderRadius: 99, background: t.progressBg, marginBottom: 12, overflow: "hidden" }}>
                  <motion.div animate={{ width: `${progress}%` }} transition={{ type: "spring", stiffness: 180, damping: 22 }}
                    style={{ height: "100%", borderRadius: 99, background: t.grad }} />
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", paddingBottom: 12, borderBottom: `1px solid ${t.border}` }}>
                  {STEPS.slice(1).map((s, i) => {
                    const ri = i + 1, done = ri < step, active = ri === step;
                    const StepIcon = s.Icon;
                    return (
                      <motion.button key={s.id}
                        whileHover={ri < step ? { scale: 1.1 } : {}} whileTap={ri < step ? { scale: 0.94 } : {}}
                        onClick={() => ri < step && navigate(ri)}
                        style={{
                          display: "flex", flexDirection: "column", alignItems: "center", gap: 4,
                          background: "none", border: "none", cursor: ri < step ? "pointer" : "default",
                          padding: "2px 4px", opacity: ri > step ? 0.28 : 1
                        }}>
                        <motion.div
                          animate={{
                            borderColor: active ? t.accent : done ? t.stepDone : t.border,
                            background: done ? t.stepDone : active ? t.stepActive : t.stepInactive,
                            boxShadow: active ? `0 0 0 4px ${t.accent}20` : "0 0 0 0px transparent"
                          }}
                          transition={{ type: "spring", stiffness: 300, damping: 22 }}
                          style={{
                            width: 30, height: 30, borderRadius: "50%", border: "2px solid",
                            display: "flex", alignItems: "center", justifyContent: "center"
                          }}>
                          <AnimatePresence mode="wait">
                            {done
                              ? <motion.div key="check" initial={{ scale: 0, rotate: -30 }} animate={{ scale: 1, rotate: 0 }} exit={{ scale: 0 }}><Check size={12} color="#fff" strokeWidth={3} /></motion.div>
                              : <motion.div key="icon" initial={{ scale: 0 }} animate={{ scale: 1 }} exit={{ scale: 0 }}><StepIcon size={12} color={active ? t.accent : t.textMute} strokeWidth={2} /></motion.div>
                            }
                          </AnimatePresence>
                        </motion.div>
                        <motion.span animate={{ color: active ? t.accent : done ? t.textMute : t.textDim }}
                          style={{ fontSize: 9, fontWeight: active ? 600 : 400, letterSpacing: "0.04em", whiteSpace: "nowrap" }}>
                          {s.label}
                        </motion.span>
                      </motion.button>
                    );
                  })}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Content */}
        <div style={{ flex: 1, overflowY: "auto", position: "relative" }}>
          <div style={{ maxWidth: 600, width: "100%", margin: "0 auto", padding: "28px 24px 0" }}>
            <AnimatePresence>
              {step === 0 && (
                <motion.h1 initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -16 }}
                  transition={{ type: "spring", stiffness: 320, damping: 26 }}
                  style={{
                    fontSize: "clamp(38px,7vw,60px)", fontWeight: 700, letterSpacing: "-0.04em",
                    lineHeight: 1.08, color: t.text, textAlign: "center", marginBottom: 28
                  }}>
                  Heart Disease{" "}<br />
                  <motion.span animate={{ color: t.accent }} transition={{ duration: 0.4 }}>Risk Assessment</motion.span>
                </motion.h1>
              )}
            </AnimatePresence>
            <AnimatePresence mode="wait" custom={dir}>
              <motion.div key={step} custom={dir} variants={pageVariants} initial="enter" animate="center" exit="exit">
                {pages[step]}
              </motion.div>
            </AnimatePresence>
          </div>
        </div>

        {/* Bottom nav */}
        <motion.div initial={{ y: 56, opacity: 0 }} animate={{ y: 0, opacity: 1 }}
          transition={{ type: "spring", stiffness: 320, damping: 28, delay: 0.1 }}
          style={{
            flexShrink: 0, borderTop: `1px solid ${t.border}`, padding: "14px 24px",
            display: "grid", gridTemplateColumns: step > 0 ? "auto 1fr" : "1fr", gap: 10,
            maxWidth: 600, width: "100%", margin: "0 auto", alignSelf: "stretch"
          }}>
          <AnimatePresence>
            {step > 0 && (
              <motion.button initial={{ opacity: 0, x: -16 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -16 }}
                whileHover={{ scale: 1.03, x: -2 }} whileTap={{ scale: 0.97 }} onClick={() => navigate(step - 1)}
                style={{
                  display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
                  padding: "13px 20px", border: `1.5px solid ${t.border}`, borderRadius: 12, cursor: "pointer",
                  background: "transparent", color: t.textMute, fontSize: 14, fontWeight: 600, fontFamily: "inherit"
                }}>
                <ChevronLeft size={16} strokeWidth={2.5} /> Back
              </motion.button>
            )}
          </AnimatePresence>

          {isLast ? (
            <motion.button whileHover={!submitting ? { scale: 1.02, y: -1 } : {}} whileTap={!submitting ? { scale: 0.98 } : {}}
              onClick={handleSubmit} disabled={submitting}
              style={{
                border: "none", borderRadius: 12, cursor: submitting ? "default" : "pointer",
                background: t.grad, color: "#fff", fontSize: 14, fontWeight: 700, fontFamily: "inherit",
                height: 52, overflow: "hidden", display: "flex", alignItems: "center", justifyContent: "center", gap: 8
              }}>
              <AnimatePresence mode="wait">
                {!submitting
                  ? <motion.span key="label" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}
                    style={{ display: "flex", alignItems: "center", gap: 7 }}>
                    Get My Results <ArrowRight size={15} strokeWidth={2.5} />
                  </motion.span>
                  : <motion.div key="loader" initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.8 }}
                    style={{ display: "flex", alignItems: "center", gap: 12 }}>
                    <BeatingHeart color="#fff" />
                    <motion.span animate={{ opacity: [0.6, 1, 0.6] }} transition={{ repeat: Infinity, duration: 1.4 }}
                      style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.12em", textTransform: "uppercase" }}>
                      Analysing…
                    </motion.span>
                  </motion.div>
                }
              </AnimatePresence>
            </motion.button>
          ) : (
            <motion.button whileHover={{ scale: 1.02, y: -1 }} whileTap={{ scale: 0.98 }}
              onClick={() => navigate(step + 1)}
              style={{
                display: "flex", alignItems: "center", justifyContent: "center", gap: 7,
                padding: "13px 28px", border: "none", borderRadius: 12, cursor: "pointer",
                background: t.accent, color: "#fff", fontSize: 14, fontWeight: 700, fontFamily: "inherit"
              }}>
              {step === 0 ? "Start Assessment" : "Continue"}
              <motion.span animate={{ x: [0, 3, 0] }} transition={{ repeat: Infinity, duration: 1.4, ease: "easeInOut" }}>
                <ChevronRight size={16} strokeWidth={2.5} />
              </motion.span>
            </motion.button>
          )}
        </motion.div>
      </motion.div>

      {/* Result overlay */}
      <AnimatePresence>
        {showResult && result !== null && (
          <ResultCard result={result} factors={factors} form={form} t={t}
            onClose={() => setShowResult(false)} onRetake={handleRetake} />
        )}
      </AnimatePresence>

      {/* Error banner */}
      <AnimatePresence>
        {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}
      </AnimatePresence>
    </>
  );
}