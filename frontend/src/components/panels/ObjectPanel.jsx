import { stateToElements, eciToGeo, RISK_COLOR, RISK_BG } from "../../utils/orbital";

const S = {
  panel: {
    width:300, flexShrink:0,
    background:"#020e1fee",
    borderLeft:"1px solid #0a3a5a",
    display:"flex", flexDirection:"column",
    padding:12, overflowY:"auto",
    backdropFilter:"blur(12px)",
    zIndex:20,
  },
  close: {
    position:"absolute",top:8,right:10,
    cursor:"pointer",color:"#4a7090",
    fontSize:14,fontWeight:700,
  },
  header: {
    display:"flex",alignItems:"flex-start",
    justifyContent:"space-between",
    marginBottom:14, paddingBottom:10,
    borderBottom:"1px solid #0a3a5a",
  },
  title: {fontSize:13,fontWeight:700,color:"#00d4ff",letterSpacing:1},
  sub:   {fontSize:9,color:"#4a7090",marginTop:2},
  riskBadge: (r) => ({
    fontSize:10,fontWeight:700,padding:"3px 10px",
    borderRadius:3,marginTop:4,display:"inline-block",
    background:RISK_BG[r], color:RISK_COLOR[r],
    border:`1px solid ${RISK_COLOR[r]}44`,
  }),
  section: {marginBottom:12},
  secTitle:{
    fontSize:8,letterSpacing:2,color:"#00d4ff55",
    fontWeight:700,marginBottom:6,
    paddingBottom:3,borderBottom:"1px solid #0a3a5a44"
  },
  row: {
    display:"flex",justifyContent:"space-between",
    alignItems:"center",marginBottom:4,
  },
  key: {color:"#7ba8c0",fontSize:10},
  val: (c) => ({fontSize:10,fontWeight:700,color:c||"#d0e8ff"}),
  divider:{height:1,background:"#0a3a5a44",margin:"8px 0"},
  btn: (bg,col,brd) => ({
    fontFamily:"'Courier New',monospace",fontSize:10,
    fontWeight:700,padding:"6px 12px",borderRadius:4,
    cursor:"pointer",background:bg,color:col,
    border:`1px solid ${brd}`,letterSpacing:1,
    width:"100%",marginBottom:6,
  }),
};

export default function ObjectPanel({ object, onClose, onFireManeuver }) {
  if (!object) return null;

  const elems = stateToElements([
    object.x, object.y, object.z,
    object.vx||0, object.vy||0, object.vz||0,
  ]);
  const geo = eciToGeo(object.x, object.y, object.z);

  return (
    <div style={{...S.panel, position:"relative"}}>
      <span style={S.close} onClick={onClose}>✕</span>

      {/* Header */}
      <div style={S.header}>
        <div>
          <div style={S.title}>NORAD {object.id}</div>
          <div style={S.sub}>{object.type}</div>
          <span style={S.riskBadge(object.risk)}>
            {object.risk} RISK
          </span>
        </div>
      </div>

      {/* Identification */}
      <div style={S.section}>
        <div style={S.secTitle}>IDENTIFICATION</div>
        {[
          ["NORAD ID",   object.id,     "#00d4ff"],
          ["Object Type",object.type,   "#d0e8ff"],
          ["Risk Level", object.risk,   RISK_COLOR[object.risk]],
          ["Altitude",   `${object.alt} km`, "#d0e8ff"],
        ].map(([k,v,c])=>(
          <div key={k} style={S.row}>
            <span style={S.key}>{k}</span>
            <span style={S.val(c)}>{v}</span>
          </div>
        ))}
      </div>

      <div style={S.divider}/>

      {/* Orbital elements */}
      <div style={S.section}>
        <div style={S.secTitle}>ORBITAL ELEMENTS</div>
        {[
          ["Inclination",   `${elems.inclination}°`,    "#d0e8ff"],
          ["Eccentricity",  elems.eccentricity,         "#d0e8ff"],
          ["Mean Motion",   `${elems.mean_motion} rev/day`, "#d0e8ff"],
          ["Period",        `${elems.period_min} min`,  "#d0e8ff"],
          ["Semi-Major Axis",`${elems.sma} km`,         "#d0e8ff"],
        ].map(([k,v,c])=>(
          <div key={k} style={S.row}>
            <span style={S.key}>{k}</span>
            <span style={S.val(c)}>{v}</span>
          </div>
        ))}
      </div>

      <div style={S.divider}/>

      {/* Altitude */}
      <div style={S.section}>
        <div style={S.secTitle}>ALTITUDE</div>
        {[
          ["Current",  `${object.alt} km`,        "#d0e8ff"],
          ["Apogee",   `${elems.apogee_km} km`,   "#d0e8ff"],
          ["Perigee",  `${elems.perigee_km} km`,  "#d0e8ff"],
        ].map(([k,v,c])=>(
          <div key={k} style={S.row}>
            <span style={S.key}>{k}</span>
            <span style={S.val(c)}>{v}</span>
          </div>
        ))}
      </div>

      <div style={S.divider}/>

      {/* Current position */}
      <div style={S.section}>
        <div style={S.secTitle}>CURRENT POSITION</div>
        {[
          ["LAT",  `${geo.lat}°`,            "#d0e8ff"],
          ["LON",  `${geo.lon}°`,            "#d0e8ff"],
          ["ALT",  `${object.alt} km`,       "#d0e8ff"],
          ["VEL",  `${elems.velocity_kms} km/s`, "#d0e8ff"],
          ["X",    `${object.x.toFixed(1)} km`,  "#4a7090"],
          ["Y",    `${object.y.toFixed(1)} km`,  "#4a7090"],
          ["Z",    `${object.z.toFixed(1)} km`,  "#4a7090"],
        ].map(([k,v,c])=>(
          <div key={k} style={S.row}>
            <span style={S.key}>{k}</span>
            <span style={S.val(c)}>{v}</span>
          </div>
        ))}
      </div>

      <div style={S.divider}/>

      {/* AI Analysis */}
      <div style={S.section}>
        <div style={S.secTitle}>AI ANALYSIS</div>
        {[
          ["CNN+LSTM",  "DETECTED ✓",       "#00ff88"],
          ["Confidence","HIGH",             "#00ff88"],
          ["PINN RMSE", "48.27 km",         "#00ff88"],
          ["PPO Status","MANEUVER READY",   "#00ff88"],
          ["F1 Score",  "0.9677",           "#00ff88"],
        ].map(([k,v,c])=>(
          <div key={k} style={S.row}>
            <span style={S.key}>{k}</span>
            <span style={S.val(c)}>{v}</span>
          </div>
        ))}
      </div>

      <div style={S.divider}/>

      {/* Action buttons */}
      <button
        style={S.btn("#042a10","#00ff88","#00ff8844")}
        onClick={()=>onFireManeuver(object)}>
        ↗ FIRE PPO AVOIDANCE MANEUVER
      </button>
      <button
        style={S.btn("#041e3a","#00d4ff","#00d4ff44")}>
        △ SHOW PINN TRAJECTORY CONE
      </button>
      <button
        style={S.btn("#0a0a20","#7ba8c0","#2a3a4a")}>
        — ADD TO WATCHLIST
      </button>
    </div>
  );
}