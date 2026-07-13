
export default function AlertsFullPanel({ alerts }) {
  const S = {
    wrap:{padding:12,fontFamily:"'Courier New',monospace",color:'#d0e8ff'},
    title:{fontSize:13,fontWeight:700,color:'#00d4ff',letterSpacing:1,marginBottom:12},
    al:(type)=>({
      fontSize:10,padding:'6px 8px',borderRadius:4,marginBottom:6,lineHeight:1.5,
      borderLeft:'2px solid '+(type==='crit'?'#ef4444':type==='warn'?'#f97316':'#00d4ff'),
      background:'#030f20',
      color:type==='crit'?'#fca5a5':type==='warn'?'#fdba74':'#93c5fd',
    }),
    time:{color:'#4a7090',fontSize:8,marginTop:2},
  };
  return (
    <div style={S.wrap}>
      <div style={S.title}>🔔 ALL ALERTS ({alerts.length})</div>
      {alerts.length===0 && (
        <div style={{color:'#4a7090',fontSize:10,textAlign:'center',padding:20}}>
          No active alerts
        </div>
      )}
      {alerts.map((a,i)=>(
        <div key={i} style={S.al(a.type)}>
          {a.msg}
          <div style={S.time}>{a.t}</div>
        </div>
      ))}
    </div>
  );
}
