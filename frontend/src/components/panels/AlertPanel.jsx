import { RISK_COLOR } from "../../utils/orbital";

const S = {
  panel: {
    width:250,flexShrink:0,
    background:"#020e1fee",
    borderLeft:"1px solid #0a3a5a",
    padding:10,display:"flex",
    flexDirection:"column",gap:6,
    overflowY:"auto",
    backdropFilter:"blur(12px)",
    zIndex:20,
  },
  pt:{fontSize:8,letterSpacing:2,color:"#00d4ff55",
      marginBottom:4,fontWeight:700},
  panel2:{border:"1px solid #0a3a5a66",borderRadius:5,
          background:"#030f20aa",padding:"7px 9px",marginBottom:6},
  sr:{display:"flex",justifyContent:"space-between",
      alignItems:"center",marginBottom:3},
  sk:{color:"#7ba8c0",fontSize:10},
  sv:(c)=>({fontSize:11,fontWeight:700,color:c||"#d0e8ff"}),
  al:(type)=>({
    fontSize:9,padding:"4px 7px",borderRadius:3,
    lineHeight:1.4,marginBottom:3,
    borderLeft:`2px solid ${type==="crit"?"#ef4444":type==="warn"?"#f97316":"#00d4ff"}`,
    background:"#030f2066",
    color:type==="crit"?"#fca5a5":type==="warn"?"#fdba74":"#93c5fd",
  }),
  dot:(ok)=>({
    display:"inline-block",width:6,height:6,
    borderRadius:"50%",marginRight:5,
    background:ok?"#00ff88":"#f97316",
    boxShadow:ok?"0 0 4px #00ff88":"0 0 4px #f97316",
  }),
};

export default function AlertPanel({ alerts, meta, apiStatus }) {
  return (
    <div style={S.panel}>

      <div style={S.panel2}>
        <div style={S.pt}>SYSTEM STATUS</div>
        {[
          ["CNN+LSTM", "F1=0.9677",  true],
          ["PINN",     "48.3km",     true],
          ["PPO Agent","100.0%",     true],
          ["FastAPI",  apiStatus.toUpperCase(), apiStatus==="online"],
        ].map(([n,v,ok])=>(
          <div key={n} style={S.sr}>
            <span style={S.sk}>
              <span style={S.dot(ok)}/>{n}
            </span>
            <span style={S.sv(ok?"#00ff88":"#f97316")}>{v}</span>
          </div>
        ))}
      </div>

      <div style={S.panel2}>
        <div style={S.pt}>LIVE ALERTS</div>
        <div style={{maxHeight:220,overflowY:"auto"}}>
          {alerts.length===0 && (
            <div style={{color:"#4a7090",fontSize:9,textAlign:"center",padding:10}}>
              No alerts
            </div>
          )}
          {alerts.map((a,i)=>(
            <div key={i} style={S.al(a.type)}>
              {a.msg}
              <div style={{color:"#4a7090",fontSize:8,marginTop:1}}>{a.t}</div>
            </div>
          ))}
        </div>
      </div>

      <div style={S.panel2}>
        <div style={S.pt}>METRICS</div>
        {[
          ["Det F1",     "0.9677",  "#00ff88"],
          ["PINN RMSE",  "48.27km", "#00ff88"],
          ["Avoidance",  "100.0%",  "#00ff88"],
          ["Latency",    "131.7ms", "#00ff88"],
          ["Objects",    (meta?.total||0).toLocaleString(), "#00d4ff"],
          ["Conj",       "209,285", "#00d4ff"],
          ["Traj rows",  "18.1M",   "#00d4ff"],
        ].map(([k,v,c])=>(
          <div key={k} style={S.sr}>
            <span style={S.sk}>{k}</span>
            <span style={S.sv(c)}>{v}</span>
          </div>
        ))}
      </div>

      <div style={S.panel2}>
        <div style={S.pt}>CONTROLS</div>
        {[["Rotate","Mouse drag"],["Zoom","Scroll"],
          ["Reset","R key"],["Select","Click object"],
          ["Details","Click → panel"]].map(([k,v])=>(
          <div key={k} style={S.sr}>
            <span style={S.sk}>{k}</span>
            <span style={S.sv("#00d4ff")}>{v}</span>
          </div>
        ))}
      </div>

    </div>
  );
}