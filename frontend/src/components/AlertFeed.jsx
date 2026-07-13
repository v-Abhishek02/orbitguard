const S = {
  rightbar: {
    width: 185, flexShrink: 0,
    background: "#020e1fdd",
    borderLeft: "1px solid #0a3a5a22",
    padding: 8, display: "flex",
    flexDirection: "column", overflowY: "auto",
    position: "absolute", right: 0, top: 0, bottom: 0,
    zIndex: 10, backdropFilter: "blur(12px)",
  },
  panel: {
    border: "1px solid #0a3a5a66", borderRadius: 5,
    background: "#030f20aa", padding: "7px 9px",
    marginBottom: 6,
  },
  pt: { fontSize:8, letterSpacing:2, color:"#00d4ff55", marginBottom:5, fontWeight:700 },
  sr: { display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:3 },
  sk: { color:"#7ba8c0", fontSize:10 },
  sv: (c) => ({ fontSize:11, fontWeight:700, color:c||"#d0e8ff" }),
  al: (type) => ({
    fontSize: 9, padding: "3px 6px", borderRadius: 3,
    lineHeight: 1.4, marginBottom: 3,
    borderLeft: `2px solid ${type==="crit"?"#ef4444":type==="warn"?"#f97316":"#00d4ff"}`,
    background: "#030f2066",
    color: type==="crit"?"#fca5a5":type==="warn"?"#fdba74":"#93c5fd",
  }),
};

export default function AlertFeed({ alerts, metrics }) {
  return (
    <div style={S.rightbar}>

      <div style={S.panel}>
        <div style={S.pt}>ALERT FEED</div>
        <div style={{maxHeight:210,overflowY:"auto"}}>
          {alerts.map((a, i) => (
            <div key={i} style={S.al(a.type)}>
              {a.msg}
              <div style={{color:"#4a7090",fontSize:8,marginTop:1}}>{a.t}</div>
            </div>
          ))}
        </div>
      </div>

      <div style={S.panel}>
        <div style={S.pt}>SYSTEM METRICS</div>
        {[
          ["Det F1",       "0.9677",   "#00ff88"],
          ["PINN RMSE",    "48.27 km", "#00ff88"],
          ["DRL Success",  "100.0%",   "#00ff88"],
          ["Latency",      "131.7ms",  "#00ff88"],
          ["Obj tracked",  "25,201",   "#00d4ff"],
          ["Conj events",  "209,285",  "#00d4ff"],
          ["Traj rows",    "18.1M",    "#00d4ff"],
        ].map(([k,v,c])=>(
          <div key={k} style={S.sr}>
            <span style={S.sk}>{k}</span>
            <span style={S.sv(c)}>{v}</span>
          </div>
        ))}
      </div>

      <div style={S.panel}>
        <div style={S.pt}>CONTROLS</div>
        {[
          ["Rotate",  "Mouse drag"],
          ["Zoom",    "Scroll wheel"],
          ["Reset",   "R key"],
        ].map(([k,v])=>(
          <div key={k} style={S.sr}>
            <span style={S.sk}>{k}</span>
            <span style={S.sv("#00d4ff")}>{v}</span>
          </div>
        ))}
      </div>

    </div>
  );
}