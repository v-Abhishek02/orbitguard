const S = {
  nav: {
    width:58, flexShrink:0,
    background:"#020e1fee",
    borderRight:"1px solid #0a3a5a",
    display:"flex", flexDirection:"column",
    alignItems:"center", padding:"10px 0",
    gap:4, zIndex:20,
    backdropFilter:"blur(12px)",
  },
  item: (active) => ({
    display:"flex", flexDirection:"column",
    alignItems:"center", justifyContent:"center",
    width:46, height:46, borderRadius:6,
    cursor:"pointer", transition:"all 0.15s",
    background:active?"#051e38":"transparent",
    border:active?"1px solid #00d4ff44":"1px solid transparent",
    color:active?"#00d4ff":"#4a7090",
    fontSize:18, gap:2,
  }),
  label: {fontSize:7, letterSpacing:0.5, fontWeight:700},
  sep: {
    width:32, height:1,
    background:"#0a3a5a", margin:"4px 0"
  },
};

const MENU = [
  { id:"dashboard",    icon:"⬟", label:"HOME"   },
  { id:"conjunctions", icon:"⚠",  label:"CONJ"   },
  { id:"satellites",   icon:"◉",  label:"SATS"   },
  { id:"debris",       icon:"✦",  label:"DEBRIS" },
  { id:"starlink",     icon:"★",  label:"STRLNK" },
  { id:"rockets",      icon:"↑",  label:"ROCKET" },
  null, // separator
  { id:"alerts",       icon:"🔔", label:"ALERTS" },
  { id:"watchlist",    icon:"👁",  label:"WATCH"  },
  { id:"search",       icon:"⌕",  label:"SEARCH" },
  null,
  { id:"settings",     icon:"⚙",  label:"SET"    },
];

export default function LeftNav({ activePanel, setActivePanel }) {
  return (
    <div style={S.nav}>
      {MENU.map((item, i) =>
        item === null
          ? <div key={i} style={S.sep}/>
          : (
            <div
              key={item.id}
              style={S.item(activePanel===item.id)}
              onClick={()=>setActivePanel(item.id)}
              title={item.label}
            >
              <span>{item.icon}</span>
              <span style={S.label}>{item.label}</span>
            </div>
          )
      )}
    </div>
  );
}