
import { useState } from 'react';
import { RISK_COLOR } from '../../utils/orbital';
export default function SearchFullPanel({ objects, onSelect }) {
  const [q, setQ] = useState('');
  const results = q.length>1
    ? objects.filter(o=>
        String(o.id).includes(q) ||
        o.type?.toLowerCase().includes(q.toLowerCase()) ||
        o.risk?.toLowerCase().includes(q.toLowerCase()) ||
        String(o.alt).includes(q)
      ).slice(0,30)
    : [];
  const S = {
    wrap:{padding:12,fontFamily:"'Courier New',monospace",color:'#d0e8ff'},
    title:{fontSize:13,fontWeight:700,color:'#00d4ff',letterSpacing:1,marginBottom:10},
    input:{width:'100%',background:'#041428',border:'1px solid #0a3a5a',
      borderRadius:5,color:'#d0e8ff',fontFamily:"'Courier New',monospace",
      fontSize:11,padding:'8px 10px',outline:'none',marginBottom:10,boxSizing:'border-box'},
    card:{border:'1px solid #0a3a5a',borderRadius:5,background:'#030f20',
      padding:'7px 10px',marginBottom:5,cursor:'pointer',transition:'border-color .15s'},
    row:{display:'flex',justifyContent:'space-between',marginBottom:2},
    k:{color:'#7ba8c0',fontSize:9},
    v:(c)=>({fontSize:10,fontWeight:700,color:c||'#d0e8ff'}),
    pill:(r)=>({fontSize:8,fontWeight:700,padding:'1px 5px',borderRadius:2,
      background:r==='HIGH'?'#3a0a0a':r==='MED'?'#2a1a00':'#0a2010',color:RISK_COLOR[r]}),
    hint:{fontSize:9,color:'#4a7090',textAlign:'center',padding:12},
  };
  return (
    <div style={S.wrap}>
      <div style={S.title}>⌕ SEARCH OBJECTS</div>
      <input style={S.input} placeholder='NORAD ID, type, risk, altitude...'
             value={q} onChange={e=>setQ(e.target.value)} autoFocus/>
      {q.length<2 && (
        <div style={S.hint}>Type at least 2 characters<br/>to search {objects.length.toLocaleString()} objects</div>
      )}
      {results.map(o=>(
        <div key={o.id} style={S.card}
          onMouseEnter={e=>e.currentTarget.style.borderColor='#00d4ff44'}
          onMouseLeave={e=>e.currentTarget.style.borderColor='#0a3a5a'}
          onClick={()=>onSelect(o)}>
          <div style={S.row}>
            <span style={{color:'#00d4ff',fontSize:10,fontWeight:700}}>NORAD {o.id}</span>
            <span style={S.pill(o.risk)}>{o.risk}</span>
          </div>
          <div style={S.row}>
            <span style={S.k}>{o.type}</span>
            <span style={S.v()}>{o.alt} km</span>
          </div>
        </div>
      ))}
      {q.length>=2 && results.length===0 && (
        <div style={S.hint}>No objects found for '{q}'</div>
      )}
    </div>
  );
}
