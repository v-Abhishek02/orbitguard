
import { RISK_COLOR } from '../../utils/orbital';
export default function WatchlistPanel({ trailObjects, objects, onSelect, onRemove }) {
  const watched = objects.filter(o=>trailObjects.has(o.id));
  const S = {
    wrap:{padding:12,fontFamily:"'Courier New',monospace",color:'#d0e8ff'},
    title:{fontSize:13,fontWeight:700,color:'#00d4ff',letterSpacing:1,marginBottom:4},
    sub:{fontSize:9,color:'#4a7090',marginBottom:12},
    card:{border:'1px solid #0a3a5a',borderRadius:5,background:'#030f20',
      padding:'7px 10px',marginBottom:5},
    row:{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:2},
    k:{color:'#7ba8c0',fontSize:9},
    v:(c)=>({fontSize:10,fontWeight:700,color:c||'#d0e8ff'}),
    empty:{color:'#4a7090',fontSize:10,textAlign:'center',padding:20,lineHeight:1.6},
  };
  return (
    <div style={S.wrap}>
      <div style={S.title}>👁 WATCHLIST</div>
      <div style={S.sub}>{watched.length} objects monitored</div>
      {watched.length===0 ? (
        <div style={S.empty}>
          No objects on watchlist.<br/>
          Click an object then select<br/>
          'Add to Watchlist' to track it.
        </div>
      ) : (
        watched.map(o=>(
          <div key={o.id} style={S.card}>
            <div style={S.row}>
              <span style={{...S.k,cursor:'pointer',color:'#00d4ff'}}
                    onClick={()=>onSelect(o)}>NORAD {o.id}</span>
              <span style={{color:'#ef4444',cursor:'pointer',fontSize:10}}
                    onClick={()=>onRemove(o.id)}>✕</span>
            </div>
            <div style={S.row}>
              <span style={S.k}>{o.type} · {o.alt}km</span>
              <span style={{...S.v(RISK_COLOR[o.risk])}}>{o.risk}</span>
            </div>
          </div>
        ))
      )}
    </div>
  );
}
