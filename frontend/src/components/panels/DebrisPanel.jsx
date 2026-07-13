
import { useState } from 'react';
import { RISK_COLOR } from '../../utils/orbital';
export default function DebrisPanel({ objects, onSelect }) {
  const [filter, setFilter] = useState('ALL');
  const debris = objects.filter(o=>o.type==='DEBRIS');
  const shown  = debris.filter(o=>filter==='ALL'||o.risk===filter).slice(0,60);
  const S = {
    wrap:{padding:12,fontFamily:"'Courier New',monospace",color:'#d0e8ff'},
    title:{fontSize:13,fontWeight:700,color:'#00d4ff',letterSpacing:1,marginBottom:4},
    sub:{fontSize:9,color:'#4a7090',marginBottom:10},
    filter:{display:'flex',gap:6,marginBottom:10,flexWrap:'wrap'},
    fbtn:(a)=>({fontSize:9,padding:'3px 8px',borderRadius:3,cursor:'pointer',
      fontFamily:"'Courier New',monospace",fontWeight:700,letterSpacing:1,
      background:a?'#051e38':'#041428',color:a?'#00d4ff':'#7ba8c0',
      border:'1px solid '+(a?'#00d4ff44':'#0a3a5a')}),
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
      <div style={S.title}>✦ DEBRIS OBJECTS</div>
      <div style={S.sub}>{debris.length.toLocaleString()} debris objects · {debris.filter(o=>o.risk==='HIGH').length} HIGH risk</div>
      <div style={S.filter}>
        {['ALL','HIGH','MED','LOW'].map(r=>(
          <button key={r} style={S.fbtn(filter===r)} onClick={()=>setFilter(r)}>{r}</button>
        ))}
      </div>
      <div style={{maxHeight:'calc(100vh - 240px)',overflowY:'auto'}}>
        {shown.map(o=>(
          <div key={o.id} style={S.card}
            onMouseEnter={e=>e.currentTarget.style.borderColor='#ef444444'}
            onMouseLeave={e=>e.currentTarget.style.borderColor='#0a3a5a'}
            onClick={()=>onSelect(o)}>
            <div style={S.row}>
              <span style={S.k}>NORAD {o.id}</span>
              <span style={S.pill(o.risk)}>{o.risk}</span>
            </div>
            <div style={S.row}>
              <span style={S.k}>Altitude</span>
              <span style={S.v()}>{o.alt} km</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
