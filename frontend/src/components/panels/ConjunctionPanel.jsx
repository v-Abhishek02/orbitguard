import { RISK_COLOR } from "../../utils/orbital";

export default function ConjunctionPanel({ conjunctions, objects, onSelect }) {
  const S = {
    panel:{padding:12,color:"#d0e8ff",fontFamily:"'Courier New',monospace"},
    title:{fontSize:13,fontWeight:700,color:"#00d4ff",letterSpacing:1,marginBottom:12},
    card:{
      border:"1px solid #0a3a5a",borderRadius:5,
      background:"#030f20",padding:"8px 10px",marginBottom:6,
      cursor:"pointer",transition:"border-color .15s",
    },
    row:{display:"flex",justifyContent:"space-between",marginBottom:2},
    k:{color:"#7ba8c0",fontSize:9},
    v:(c)=>({fontSize:10,fontWeight:700,color:c||"#d0e8ff"}),
  };

  return (
    <div style={S.panel}>
      <div style={S.title}>⚠ ACTIVE CONJUNCTIONS ({conjunctions.length})</div>
      {conjunctions.length===0 && (
        <div style={{color:"#4a7090",fontSize:10,textAlign:"center",padding:20}}>
          No active conjunction events
        </div>
      )}
      {conjunctions.map((c,i)=>(
        <div key={i} style={S.card}
          onMouseEnter={e=>e.currentTarget.style.borderColor="#ef444466"}
          onMouseLeave={e=>e.currentTarget.style.borderColor="#0a3a5a"}
          onClick={()=>{
            const o=objects.find(ob=>ob.id===c.obj1);
            if(o) onSelect(o);
          }}>
          <div style={S.row}>
            <span style={S.k}>PAIR</span>
            <span style={S.v("#ef4444")}>
              {c.obj1} ↔ {c.obj2}
            </span>
          </div>
          <div style={S.row}>
            <span style={S.k}>MISS DIST</span>
            <span style={S.v(c.miss<1?"#ef4444":"#f97316")}>
              {c.miss} km
            </span>
          </div>
          <div style={S.row}>
            <span style={S.k}>STATUS</span>
            <span style={S.v("#ef4444")}>HIGH RISK</span>
          </div>
        </div>
      ))}
    </div>
  );
}