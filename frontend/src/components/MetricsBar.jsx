const S = {
  bar: {
    flexShrink: 0, padding: "7px 14px",
    background: "#020e1fee",
    borderTop: "1px solid #0a3a5a",
    display: "flex", gap: 8, alignItems: "center",
    zIndex: 20, backdropFilter: "blur(8px)",
  },
  btn: (bg, col, brd) => ({
    fontFamily: "'Courier New',monospace",
    fontSize: 10, letterSpacing: 1,
    fontWeight: 700, padding: "5px 12px",
    borderRadius: 4, cursor: "pointer",
    background: bg, color: col,
    border: `1px solid ${brd}`,
    transition: "all 0.15s",
  }),
};

export default function MetricsBar({
  paused, setPaused, showCones, setShowCones,
  fireManeuver, totalTime,
}) {
  const fmtTime = (s) => {
    const ss=Math.floor(s), h=Math.floor(ss/3600),
          m=Math.floor((ss%3600)/60), sec=ss%60;
    return `T+${String(h).padStart(2,"0")}:${String(m).padStart(2,"0")}:${String(sec).padStart(2,"0")}`;
  };

  return (
    <div style={S.bar}>
      <button
        style={S.btn(
          paused?"#042a10":"#041e3a",
          paused?"#00ff88":"#00d4ff",
          paused?"#00ff8844":"#00d4ff44"
        )}
        onClick={()=>setPaused(p=>!p)}>
        {paused ? "▶ RESUME" : "⏸ PAUSE"}
      </button>

      <button
        style={S.btn("#042a10","#00ff88","#00ff8844")}
        onClick={fireManeuver}>
        ↗ FIRE PPO MANEUVER
      </button>

      <button
        style={S.btn(
          showCones?"#051e38":"#0a0a1a",
          showCones?"#00d4ff":"#4a6a80",
          showCones?"#00d4ff44":"#2a3a4a"
        )}
        onClick={()=>setShowCones(c=>!c)}>
        △ PINN CONES {showCones?"ON":"OFF"}
      </button>

      <span style={{marginLeft:"auto",fontSize:9,color:"#4a7090"}}>
        MSc Data Science & AI · ORBITGUARD v1.0 ·
        FastAPI + React + Three.js
      </span>

      <span style={{fontSize:12,color:"#00d4ff",fontWeight:700}}>
        {fmtTime(totalTime)}
      </span>
    </div>
  );
}