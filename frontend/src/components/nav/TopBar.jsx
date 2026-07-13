import { RISK_COLOR } from "../../utils/orbital";

const S = {
  bar: {
    display:"flex", alignItems:"center",
    justifyContent:"space-between",
    padding:"0 16px", height:52,
    background:"#020e1fee",
    borderBottom:"1px solid #0a3a5a",
    flexShrink:0, zIndex:30,
    backdropFilter:"blur(12px)",
  },
  logo: {
    display:"flex",alignItems:"center",gap:10
  },
  logoIcon: {fontSize:18,color:"#00d4ff"},
  logoText: {
    fontSize:13,fontWeight:700,
    color:"#00d4ff",letterSpacing:3
  },
  logoSub: {fontSize:8,color:"#4a7090",letterSpacing:2,marginTop:1},
  search: {
    display:"flex",alignItems:"center",
    background:"#041428",
    border:"1px solid #0a3a5a",
    borderRadius:5, padding:"0 10px",
    gap:8, width:240,
  },
  searchInput: {
    background:"none",border:"none",outline:"none",
    color:"#d0e8ff",fontFamily:"'Courier New',monospace",
    fontSize:11, width:"100%", padding:"7px 0",
  },
  badges: {display:"flex",gap:6,alignItems:"center"},
  badge: (bg,col,brd) => ({
    fontSize:9, padding:"3px 9px",
    borderRadius:3, fontWeight:700,
    letterSpacing:1, background:bg,
    color:col, border:`1px solid ${brd}`,
  }),
  clock: {
    textAlign:"right"
  },
  utc: {
    fontSize:13,fontWeight:700,
    color:"#00d4ff",letterSpacing:1
  },
  met: {fontSize:9,color:"#4a7090",marginTop:1},
};

export default function TopBar({
  apiStatus, objects, meta,
  searchQuery, setSearchQuery,
  onSearchSelect, totalTime, utcTime,
}) {
  const highN = meta?.high || 0;

  const handleSearch = (e) => {
    setSearchQuery(e.target.value);
  };

  const filtered = searchQuery.length > 1
    ? objects.filter(o =>
        String(o.id).includes(searchQuery) ||
        o.type?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        o.risk?.toLowerCase().includes(searchQuery.toLowerCase())
      ).slice(0,6)
    : [];

  const fmtMET = (s) => {
    const ss=Math.floor(s), h=Math.floor(ss/3600),
          m=Math.floor((ss%3600)/60), sec=ss%60;
    return `T+${String(h).padStart(2,"0")}:`
         + `${String(m).padStart(2,"0")}:`
         + `${String(sec).padStart(2,"0")}`;
  };

  return (
    <div style={S.bar}>
      {/* LOGO */}
      <div style={S.logo}>
        <span style={S.logoIcon}>⬟</span>
        <div>
          <div style={S.logoText}>ORBITGUARD</div>
          <div style={S.logoSub}>MISSION CONTROL v1.0</div>
        </div>
      </div>

      {/* SEARCH */}
      <div style={{position:"relative"}}>
        <div style={S.search}>
          <span style={{color:"#4a7090",fontSize:12}}>⌕</span>
          <input
            style={S.searchInput}
            placeholder="Search NORAD ID, type, risk..."
            value={searchQuery}
            onChange={handleSearch}
          />
          {searchQuery && (
            <span
              style={{color:"#4a7090",cursor:"pointer",fontSize:12}}
              onClick={()=>setSearchQuery("")}>✕</span>
          )}
        </div>
        {/* Search dropdown */}
        {filtered.length > 0 && (
          <div style={{
            position:"absolute",top:"100%",left:0,right:0,
            background:"#020e1f",border:"1px solid #0a3a5a",
            borderRadius:"0 0 5px 5px",zIndex:100,marginTop:1,
          }}>
            {filtered.map(o=>(
              <div key={o.id}
                onClick={()=>{onSearchSelect(o);setSearchQuery("");}}
                style={{
                  display:"flex",justifyContent:"space-between",
                  padding:"6px 10px",cursor:"pointer",
                  borderBottom:"1px solid #0a3a5a22",
                  fontFamily:"'Courier New',monospace",fontSize:10,
                }}
                onMouseEnter={e=>e.currentTarget.style.background="#041428"}
                onMouseLeave={e=>e.currentTarget.style.background="none"}
              >
                <div>
                  <span style={{color:"#00d4ff"}}>#{o.id}</span>
                  <span style={{color:"#7ba8c0",marginLeft:8}}>{o.type}</span>
                  <span style={{color:"#7ba8c0",marginLeft:8}}>{o.alt}km</span>
                </div>
                <span style={{
                  fontSize:8,fontWeight:700,padding:"1px 5px",
                  borderRadius:2,
                  background:o.risk==="HIGH"?"#3a0a0a":o.risk==="MED"?"#2a1a00":"#0a2010",
                  color:RISK_COLOR[o.risk],
                }}>{o.risk}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* STATUS BADGES */}
      <div style={S.badges}>
        <span style={S.badge("#0d3320","#00ff88","#00ff8844")}>
          ● LIVE
        </span>
        <span style={S.badge("#0a1a3a","#00d4ff","#00d4ff44")}>
          {(meta?.total||objects.length).toLocaleString()} OBJECTS
        </span>
        <span style={S.badge("#3a0a0a","#ef4444","#ef444444")}>
          ⚠ {highN} HIGH RISK
        </span>
        <span style={S.badge(
          apiStatus==="online"?"#0d3320":"#3a0a0a",
          apiStatus==="online"?"#00ff88":"#ef4444",
          apiStatus==="online"?"#00ff8844":"#ef444444",
        )}>
          API {apiStatus.toUpperCase()}
        </span>
      </div>

      {/* CLOCKS */}
      <div style={S.clock}>
        <div style={S.utc}>{utcTime}</div>
        <div style={S.met}>MET {fmtMET(totalTime)}</div>
      </div>
    </div>
  );
}