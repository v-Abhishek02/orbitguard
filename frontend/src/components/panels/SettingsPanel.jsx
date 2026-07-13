export default function SettingsPanel({
  showCones, setShowCones,
  showTrails, setShowTrails,
  showLabels, setShowLabels,
  filterRisk, setFilterRisk,
  filterType, setFilterType,
}) {
  const S = {
    panel:{padding:12,color:"#d0e8ff",
           fontFamily:"'Courier New',monospace"},
    title:{fontSize:13,fontWeight:700,color:"#00d4ff",
           letterSpacing:1,marginBottom:14},
    group:{marginBottom:14},
    gl:{fontSize:8,letterSpacing:2,color:"#00d4ff55",
        fontWeight:700,marginBottom:8,
        paddingBottom:4,borderBottom:"1px solid #0a3a5a44"},
    row:{display:"flex",justifyContent:"space-between",
         alignItems:"center",marginBottom:8},
    label:{color:"#7ba8c0",fontSize:10},
    toggle:(on)=>({
      width:36,height:18,borderRadius:9,
      background:on?"#00d4ff":"#1a2a3a",
      cursor:"pointer",position:"relative",
      border:"1px solid "+( on?"#00d4ff88":"#0a3a5a"),
      transition:"all .2s",flexShrink:0,
    }),
    knob:(on)=>({
      position:"absolute",top:2,
      left:on?18:2,width:12,height:12,
      borderRadius:"50%",
      background:on?"#fff":"#4a7090",
      transition:"left .2s",
    }),
    select:{
      background:"#041428",color:"#d0e8ff",
      border:"1px solid #0a3a5a",borderRadius:4,
      fontFamily:"'Courier New',monospace",
      fontSize:10,padding:"3px 8px",outline:"none",
    },
  };

  const Toggle = ({val,setVal}) => (
    <div style={S.toggle(val)} onClick={()=>setVal(!val)}>
      <div style={S.knob(val)}/>
    </div>
  );

  return (
    <div style={S.panel}>
      <div style={S.title}>⚙ SETTINGS</div>

      <div style={S.group}>
        <div style={S.gl}>VISUALISATION</div>
        {[
          ["PINN Prediction Cones", showCones,  setShowCones],
          ["Orbital Trails",        showTrails, setShowTrails],
          ["Object Labels",         showLabels, setShowLabels],
        ].map(([label,val,setter])=>(
          <div key={label} style={S.row}>
            <span style={S.label}>{label}</span>
            <Toggle val={val} setVal={setter}/>
          </div>
        ))}
      </div>

      <div style={S.group}>
        <div style={S.gl}>FILTER BY RISK</div>
        <select style={S.select}
                value={filterRisk}
                onChange={e=>setFilterRisk(e.target.value)}>
          {["ALL","HIGH","MED","LOW"].map(r=>(
            <option key={r} value={r}>{r}</option>
          ))}
        </select>
      </div>

      <div style={S.group}>
        <div style={S.gl}>FILTER BY TYPE</div>
        <select style={S.select}
                value={filterType}
                onChange={e=>setFilterType(e.target.value)}>
          {["ALL","DEBRIS","PAYLOAD","ROCKET BODY"].map(t=>(
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
      </div>

      <div style={S.group}>
        <div style={S.gl}>FUTURE ENHANCEMENTS (3 MONTHS)</div>
        {[
          "1. Real-time TLE WebSocket streaming",
          "2. CesiumJS photorealistic terrain",
          "3. Conjunction countdown timers",
          "4. Maneuver planning interface",
          "5. Starlink constellation view",
          "6. Mobile companion app",
          "7. 24h historical replay",
          "8. Multi-spacecraft tracking",
          "9. ESA DISCOS integration",
          "10. CNN+LSTM attention explainability",
        ].map(item=>(
          <div key={item} style={{
            fontSize:9,color:"#4a7090",
            marginBottom:4,lineHeight:1.4
          }}>{item}</div>
        ))}
      </div>
    </div>
  );
}