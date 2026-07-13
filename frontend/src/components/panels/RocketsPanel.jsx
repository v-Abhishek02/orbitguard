
import { RISK_COLOR } from '../../utils/orbital';
export default function RocketsPanel({ objects, onSelect }) {
  const rockets = objects.filter(o=>o.type==='ROCKET BODY');
  const S = {
    wrap:{padding:12,fontFamily:"'Courier New',monospace",color:'#d0e8ff'},
    title:{fontSize:13,fontWeight:700,color:'#00d4ff',letterSpacing:1,marginBottom:4},
    sub:{fontSize:9,color:'#4a7090',marginBottom:12},
    card:{border:'1px solid #0a3a5a',borderRadius:5,background:'#030f20',
      padding:'7px 10px',marginBottom:5,cursor:'pointer',transition:'border-color .15s'},
    row:{display:'flex',justifyContent:'space-between',marginBottom:2},
    k:{color:'#7ba8c0',fontSize:9},
    v:(c)=>({fontSize:10,fontWeight:700,color:c||'#d0e8ff'}),
    pill:(r)=>({fontSize:8,fontWeight:700,padding:'1px 5px',borderRadius:2,
      background:r==='HIGH'?'#3a0a0a':r==='MED'?'#2a1a00':'#0a2010',color:RISK_COLOR[r]}),
  };
  return (
    <div style={S.wrap}>
      <div style={S.title}>↑ ROCKET BODIES</div>
      <div style={S.sub}>{rockets.length.toLocaleString()} spent upper stages</div>
      <div style={{maxHeight:'calc(100vh - 220px)',overflowY:'auto'}}>
        {rockets.slice(0,60).map(o=>(
          <div key={o.id} style={S.card}
            onMouseEnter={e=>e.currentTarget.style.borderColor='#f9731644'}
            onMouseLeave={e=>e.currentTarget.style.borderColor='#0a3a5a'}
            onClick={()=>onSelect(o)}>
            <div style={S.row}>
              <span style={S.k}>NORAD {o.id}</span>
              <span style={S.pill(o.risk)}>{o.risk}</span>
            </div>
            <div style={S.row}>
              <span style={S.k}>Altitude</span>
              <span style={S.v('#f97316')}>{o.alt} km</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
