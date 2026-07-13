
export default function StarlinkPanel({ objects, onSelect }) {
  const starlink = objects.filter(o=>o.type==='PAYLOAD' && o.alt < 600);
  const S = {
    wrap:{padding:12,fontFamily:"'Courier New',monospace",color:'#d0e8ff'},
    title:{fontSize:13,fontWeight:700,color:'#00d4ff',letterSpacing:1,marginBottom:4},
    sub:{fontSize:9,color:'#4a7090',marginBottom:12},
    info:{border:'1px solid #0a3a5a66',borderRadius:5,background:'#030f20aa',
      padding:'10px 12px',marginBottom:10},
    card:{border:'1px solid #0a3a5a',borderRadius:5,background:'#030f20',
      padding:'7px 10px',marginBottom:5,cursor:'pointer',transition:'border-color .15s'},
    row:{display:'flex',justifyContent:'space-between',marginBottom:2},
    k:{color:'#7ba8c0',fontSize:9},
    v:(c)=>({fontSize:10,fontWeight:700,color:c||'#d0e8ff'}),
  };
  return (
    <div style={S.wrap}>
      <div style={S.title}>★ STARLINK CONSTELLATION</div>
      <div style={S.sub}>Low-altitude payload objects</div>
      <div style={S.info}>
        <div style={S.row}><span style={S.k}>Total tracked</span><span style={S.v('#00d4ff')}>{starlink.length}</span></div>
        <div style={S.row}><span style={S.k}>Altitude range</span><span style={S.v()}>340-600 km</span></div>
        <div style={S.row}><span style={S.k}>Orbital planes</span><span style={S.v()}>Est. 72+</span></div>
        <div style={S.row}><span style={S.k}>Inclinations</span><span style={S.v()}>53°, 70°, 97.6°</span></div>
        <div style={{fontSize:9,color:'#f97316',marginTop:6}}>
          ⚠ Future enhancement: Real Starlink TLE streaming
        </div>
      </div>
      <div style={{maxHeight:'calc(100vh - 300px)',overflowY:'auto'}}>
        {starlink.slice(0,40).map(o=>(
          <div key={o.id} style={S.card}
            onMouseEnter={e=>e.currentTarget.style.borderColor='#00d4ff44'}
            onMouseLeave={e=>e.currentTarget.style.borderColor='#0a3a5a'}
            onClick={()=>onSelect(o)}>
            <div style={S.row}>
              <span style={S.k}>NORAD {o.id}</span>
              <span style={S.v('#3b82f6')}>LEO</span>
            </div>
            <div style={S.row}>
              <span style={S.k}>Alt</span>
              <span style={S.v()}>{o.alt} km</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
