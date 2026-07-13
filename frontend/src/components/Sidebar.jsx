const S = {
  sidebar: {
    width: 210, flexShrink: 0,
    background: "#020e1fdd",
    borderRight: "1px solid #0a3a5a22",
    padding: 8, display: "flex",
    flexDirection: "column", overflowY: "auto",
    position: "absolute", left: 0, top: 0, bottom: 0,
    zIndex: 10, backdropFilter: "blur(12px)",
  },
  panel: {
    border: "1px solid #0a3a5a66", borderRadius: 5,
    background: "#030f20aa", padding: "7px 9px",
    marginBottom: 6,
  },
  pt:  { fontSize: 8, letterSpacing: 2, color: "#00d4ff55", marginBottom: 5, fontWeight: 700 },
  sr:  { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 3 },
  sk:  { color: "#7ba8c0", fontSize: 10 },
  sv:  (c) => ({ fontSize: 11, fontWeight: 700, color: c || "#d0e8ff" }),
  dot: (ok) => ({
    display: "inline-block", width: 6, height: 6,
    borderRadius: "50%", marginRight: 5,
    background: ok ? "#00ff88" : "#f97316",
    boxShadow: ok ? "0 0 4px #00ff88" : "0 0 4px #f97316",
  }),
  pill: (r) => ({
    fontSize: 8, fontWeight: 700, padding: "1px 5px",
    borderRadius: 2,
    background: r==="HIGH"?"#3a0a0a":r==="MED"?"#2a1a00":"#0a2010",
    color: r==="HIGH"?"#ef4444":r==="MED"?"#f97316":"#3b82f6",
    border: `1px solid ${r==="HIGH"?"#ef444444":r==="MED"?"#f9731644":"#3b82f644"}`,
  }),
  rbar: { width:"100%",height:4,background:"#0a1a2a",borderRadius:3,overflow:"hidden",marginTop:3 },
};

export default function Sidebar({
  apiStatus, fuel, manCount, lastMan,
  objects, meta, totalTime,
}) {
  const fmtTime = (s) => {
    const ss=Math.floor(s), h=Math.floor(ss/3600),
          m=Math.floor((ss%3600)/60), sec=ss%60;
    return `T+${String(h).padStart(2,"0")}:${String(m).padStart(2,"0")}:${String(sec).padStart(2,"0")}`;
  };

  const highN = meta?.high || objects.filter(o=>o.risk==="HIGH").length;
  const medN  = meta?.med  || objects.filter(o=>o.risk==="MED").length;
  const risk  = Math.min(100, highN * 1.2 + medN * 0.3);

  return (
    <div style={S.sidebar}>

      <div style={S.panel}>
        <div style={S.pt}>AI MODULE STATUS</div>
        {[
          ["CNN+LSTM", "F1=0.9677",  true],
          ["PINN",     "48.3km",     true],
          ["PPO AGENT","100.0%",     true],
          ["FastAPI",  apiStatus.toUpperCase(), apiStatus==="online"],
        ].map(([n,v,ok])=>(
          <div key={n} style={S.sr}>
            <span style={S.sk}>
              <span style={S.dot(ok)} />{n}
            </span>
            <span style={S.sv(ok?"#00ff88":"#f97316")}>{v}</span>
          </div>
        ))}
      </div>

      <div style={S.panel}>
        <div style={S.pt}>SPACECRAFT STATE</div>
        {[
          ["ID",        "ISS-25544",            "#00d4ff"],
          ["Fuel (ΔV)", `${fuel.toFixed(0)} m/s`, fuel>200?"#00ff88":"#f97316"],
          ["Maneuvers", String(manCount),        "#d0e8ff"],
          ["Risk",      risk>60?"HIGH":risk>25?"MED":"LOW",
                        risk>60?"#ef4444":risk>25?"#f97316":"#00ff88"],
        ].map(([k,v,c])=>(
          <div key={k} style={S.sr}>
            <span style={S.sk}>{k}</span>
            <span style={S.sv(c)}>{v}</span>
          </div>
        ))}
        <div style={S.rbar}>
          <div style={{
            height:"100%", borderRadius:3, transition:"width .6s,background .6s",
            width:`${risk}%`,
            background:risk>60?"#ef4444":risk>25?"#f97316":"#00ff88",
          }}/>
        </div>
      </div>

      <div style={S.panel}>
        <div style={S.pt}>ORBITAL CENSUS</div>
        {[
          ["HIGH risk",    highN,               "#ef4444"],
          ["MED risk",     medN,                "#f97316"],
          ["Total objects",objects.length,      "#00d4ff"],
          ["Conjunctions", "209,285",           "#d0e8ff"],
          ["Trajectories", "18.1M rows",        "#d0e8ff"],
        ].map(([k,v,c])=>(
          <div key={k} style={S.sr}>
            <span style={S.sk}>{k}</span>
            <span style={S.sv(c)}>
              {typeof v==="number"?v.toLocaleString():v}
            </span>
          </div>
        ))}
      </div>

      {lastMan && (
        <div style={S.panel}>
          <div style={S.pt}>LAST MANEUVER</div>
          {[
            ["ID",  lastMan.id,                          "#00d4ff"],
            ["ΔVx", `${Number(lastMan.dvx)>0?"+":""}${Number(lastMan.dvx).toFixed(4)}m/s`, "#d0e8ff"],
            ["ΔVy", `${Number(lastMan.dvy)>0?"+":""}${Number(lastMan.dvy).toFixed(4)}m/s`, "#d0e8ff"],
            ["|ΔV|",`${Number(lastMan.mag).toFixed(4)} m/s`,                               "#00ff88"],
            ["Time",lastMan.t,                           "#7ba8c0"],
          ].map(([k,v,c])=>(
            <div key={k} style={S.sr}>
              <span style={S.sk}>{k}</span>
              <span style={S.sv(c)}>{v}</span>
            </div>
          ))}
        </div>
      )}

      <div style={S.panel}>
        <div style={S.pt}>HIGH RISK OBJECTS</div>
        <div style={{maxHeight:165,overflowY:"auto"}}>
          {objects.filter(o=>o.risk==="HIGH").slice(0,10).map(o=>(
            <div key={o.id} style={{
              display:"flex",justifyContent:"space-between",
              alignItems:"center",padding:"3px 5px",
              borderRadius:4,background:"#041428",
              marginBottom:2,fontSize:10,
            }}>
              <div>
                <div style={{color:"#7ba8c0",fontSize:8}}>{o.id}</div>
                <div>{o.alt}km · {o.type}</div>
              </div>
              <span style={S.pill("HIGH")}>HIGH</span>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}