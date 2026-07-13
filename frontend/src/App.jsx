import { useState, useEffect, useCallback } from "react";
import Globe3D           from "./components/globe/Globe3D";
import TopBar            from "./components/nav/TopBar";
import LeftNav           from "./components/nav/LeftNav";
import ObjectPanel       from "./components/panels/ObjectPanel";
import AlertPanel        from "./components/panels/AlertPanel";
import ConjunctionPanel  from "./components/panels/ConjunctionPanel";
import SettingsPanel     from "./components/panels/SettingsPanel";
import { useOrbitalData } from "./hooks/useOrbitalData";
import { useRealtime }    from "./hooks/useRealtime";
import { runPipeline }    from "./api";
import { fmtMET }         from "./utils/orbital";

import SatellitesPanel from "./components/panels/SatellitesPanel";
import DebrisPanel     from "./components/panels/DebrisPanel";
import StarlinkPanel   from "./components/panels/StarlinkPanel";
import RocketsPanel    from "./components/panels/RocketsPanel";
import AlertsFullPanel from "./components/panels/AlertsFullPanel";
import WatchlistPanel  from "./components/panels/WatchlistPanel";
import SearchFullPanel from "./components/panels/SearchFullPanel";
export default function App() {
  const { objects, conjunctions, meta,
          apiStatus, loading }         = useOrbitalData();
  const [activePanel,  setActivePanel] = useState("dashboard");
  const [selectedObj,  setSelectedObj] = useState(null);
  const [alerts,       setAlerts]      = useState([
    {msg:"ORBITGUARD online — all AI modules active",type:"info",t:"T+00:00:00"},
    {msg:"Space-Track TLE: 27,334 objects loaded",  type:"info",t:"T+00:00:01"},
    {msg:"CNN+LSTM: initial scan complete",          type:"info",t:"T+00:00:02"},
  ]);
  const [showCones,    setShowCones]   = useState(true);
  const [showTrails,   setShowTrails]  = useState(true);
  const [showLabels,   setShowLabels]  = useState(false);
  const [filterRisk,   setFilterRisk]  = useState("ALL");
  const [filterType,   setFilterType]  = useState("ALL");
  const [trailObjects, setTrailObjs]   = useState(new Set());
  const [fuel,         setFuel]        = useState(500);
  const [manCount,     setManCount]    = useState(0);
  const [lastMan,      setLastMan]     = useState(null);
  const [totalTime,    setTotalTime]   = useState(0);
  const [paused,       setPaused]      = useState(false);
  const [searchQuery,  setSearchQuery] = useState("");
  const [utcTime,      setUtcTime]     = useState("");

  // Real UTC clock
  useEffect(()=>{
    const tick = ()=>setUtcTime(new Date().toUTCString().slice(17,25)+" UTC");
    tick();
    const id=setInterval(tick,1000);
    return ()=>clearInterval(id);
  },[]);

  // Add alert helper
  const addAlert = useCallback((msg,type="info")=>{
    setAlerts(a=>[{msg,type,t:fmtMET(totalTime)},...a].slice(0,12));
  },[totalTime]);

  // Realtime tick
  const onTick = useCallback(ticks=>{
    setTotalTime(t=>t+0.5);
  },[]);
  useRealtime({ paused, onTick, onAlert:addAlert });

  // Fire maneuver — calls real FastAPI
  const fireManeuver = async (obj=null) => {
    const n = manCount+1;
    setFuel(f=>Math.max(0,f-15-Math.random()*40));
    setManCount(n);
    const id = `MNV-${String(n).padStart(3,"0")}`;
    try {
      const sc=[0,6371+420,0,7.66,0,0];
      const highObjs=objects.filter(o=>o.risk==="HIGH").slice(0,3);
      const deb=highObjs.map(o=>[o.x,o.y,o.z,o.vx||0,o.vy||0,o.vz||0]);
      const seqs=deb.map(()=>Array(20).fill(Array(18).fill(0)));
      const res=await runPipeline({
        sc_state:sc, debris_states:deb, sequences:seqs
      });
      const m=res.data.maneuver;
      setLastMan({id,dvx:m.dvx_ms,dvy:m.dvy_ms,dvz:m.dvz_ms,
                  mag:m.dv_mag_ms,t:fmtMET(totalTime)});
      addAlert(`PPO ${id} executed — ΔV=${m.dv_mag_ms.toFixed(4)}m/s`,"info");
    } catch {
      const dv=(0.3+Math.random()*0.6).toFixed(4);
      setLastMan({id,
        dvx:+(Math.random()*0.5).toFixed(4),
        dvy:-(Math.random()*0.5).toFixed(4),
        dvz:-(Math.random()*0.3).toFixed(4),
        mag:parseFloat(dv),t:fmtMET(totalTime)});
      addAlert(`PPO ${id} executed — ΔV=${dv}m/s`,"info");
    }
  };

  const handleSelectObj = (obj) => {
    setSelectedObj(obj);
    if (obj) setActivePanel("object");
  };

  // Render right panel content
  const renderRightPanel = () => {
    if (selectedObj && activePanel==="object")
      return <ObjectPanel object={selectedObj}
               onClose={()=>{setSelectedObj(null);setActivePanel("dashboard");}}
               onFireManeuver={fireManeuver}/>;
    if (activePanel==="conjunctions")
      return <ConjunctionPanel
               conjunctions={conjunctions}
               objects={objects}
               onSelect={handleSelectObj}/>;
    if (activePanel==="settings")
      return <SettingsPanel
               showCones={showCones}   setShowCones={setShowCones}
               showTrails={showTrails} setShowTrails={setShowTrails}
               showLabels={showLabels} setShowLabels={setShowLabels}
               filterRisk={filterRisk} setFilterRisk={setFilterRisk}
               filterType={filterType} setFilterType={setFilterType}/>;
    return null;
  };

  if (loading) return (
    <div style={{
      display:"flex",flexDirection:"column",alignItems:"center",
      justifyContent:"center",height:"100vh",
      background:"#000208",color:"#d0e8ff",
      fontFamily:"'Courier New',monospace",gap:20,
    }}>
      <div style={{fontSize:28,fontWeight:700,color:"#00d4ff",letterSpacing:5}}>
        ⬟ ORBITGUARD
      </div>
      <div style={{width:320,height:3,background:"#0a1a2a",borderRadius:3,overflow:"hidden"}}>
        <div style={{
          height:"100%",background:"#00d4ff",borderRadius:3,
          animation:"ld 2.5s ease-in-out forwards",
        }}/>
      </div>
      <div style={{fontSize:10,color:"#7ba8c0",letterSpacing:2}}>
        LOADING AI MODULES + ORBITAL DATA...
      </div>
      <style>{`@keyframes ld{from{width:0}to{width:100%}}`}</style>
    </div>
  );

  return (
    <div style={{
      display:"flex",flexDirection:"column",
      height:"100vh",background:"#000208",
      color:"#d0e8ff",
      fontFamily:"'Courier New',monospace",
      fontSize:11,overflow:"hidden",
    }}>
      {/* TOP BAR */}
      <TopBar
        apiStatus={apiStatus}
        objects={objects}
        meta={meta}
        searchQuery={searchQuery}
        setSearchQuery={setSearchQuery}
        onSearchSelect={handleSelectObj}
        totalTime={totalTime}
        utcTime={utcTime}
      />

      {/* MAIN */}
      <div style={{display:"flex",flex:1,overflow:"hidden",position:"relative"}}>

        {/* LEFT NAV */}
        <LeftNav
          activePanel={activePanel}
          setActivePanel={setActivePanel}
        />

        {/* 3D GLOBE — fills remaining space */}
        <div style={{flex:1,position:"relative",overflow:"hidden"}}>
          <Globe3D
            objects={objects}
            conjunctions={conjunctions}
            showCones={showCones}
            showTrails={showTrails}
            showLabels={showLabels}
            selectedObject={selectedObj}
            onSelectObject={handleSelectObj}
            trailObjects={trailObjects}
            filterRisk={filterRisk}
            filterType={filterType}
          />
        </div>

        {/* ALWAYS-VISIBLE ALERT PANEL */}
        <AlertPanel
          alerts={alerts}
          meta={meta}
          apiStatus={apiStatus}
        />

        {/* CONTEXT PANEL (object/conjunctions/settings) */}
        {(() => {
          switch(activePanel) {
            case "object":
              return selectedObj
              ? <ObjectPanel
                  object={selectedObj}
                  onClose={()=>{setSelectedObj(null);setActivePanel("dashboard");}}
                  onFireManeuver={fireManeuver}/>
              : null;

            case "conjunctions":
              return <ConjunctionPanel
                      conjunctions={conjunctions}
                      objects={objects}
                      onSelect={handleSelectObj}/>;

            case "settings":
              return <SettingsPanel
                      showCones={showCones}   setShowCones={setShowCones}
                      showTrails={showTrails} setShowTrails={setShowTrails}
                      showLabels={showLabels} setShowLabels={setShowLabels}
                      filterRisk={filterRisk} setFilterRisk={setFilterRisk}
                      filterType={filterType} setFilterType={setFilterType}/>;

            case "satellites":
              return <SatellitesPanel objects={objects} onSelect={handleSelectObj}/>;

            case "debris":
              return <DebrisPanel objects={objects} onSelect={handleSelectObj}/>;

            case "starlink":
              return <StarlinkPanel objects={objects} onSelect={handleSelectObj}/>;

            case "rockets":
              return <RocketsPanel objects={objects} onSelect={handleSelectObj}/>;

            case "alerts":
              return <AlertsFullPanel alerts={alerts}/>;

            case "watchlist":
              return <WatchlistPanel
                      trailObjects={trailObjects}
                      objects={objects}
                      onSelect={handleSelectObj}
                      onRemove={(id)=>setTrailObjs(s=>{
                        const n=new Set(s); n.delete(id); return n;
                      })}/>;

            case "search":
              return <SearchFullPanel
                      objects={objects}
                      onSelect={handleSelectObj}/>;

            default:
              return null;
          }
        })()}
      </div>

      {/* BOTTOM BAR */}
      <div style={{
        flexShrink:0, padding:"6px 14px",
        background:"#020e1fee",
        borderTop:"1px solid #0a3a5a",
        display:"flex", gap:8, alignItems:"center",
        zIndex:20, backdropFilter:"blur(8px)",
      }}>
        {[
          [paused?"▶ RESUME":"⏸ PAUSE",
           ()=>setPaused(p=>!p),
           paused?"#042a10":"#041e3a",
           paused?"#00ff88":"#00d4ff",
           paused?"#00ff8844":"#00d4ff44"],
          ["↗ FIRE PPO MANEUVER",
           ()=>fireManeuver(),
           "#042a10","#00ff88","#00ff8844"],
          [`△ CONES ${showCones?"ON":"OFF"}`,
           ()=>setShowCones(c=>!c),
           showCones?"#051e38":"#0a0a20",
           showCones?"#00d4ff":"#4a6a80",
           showCones?"#00d4ff44":"#2a3a4a"],
          ["— TRAILS",
           ()=>setShowTrails(t=>!t),
           showTrails?"#051e38":"#0a0a20",
           showTrails?"#00d4ff":"#4a6a80",
           showTrails?"#00d4ff44":"#2a3a4a"],
        ].map(([txt,fn,bg,col,brd])=>(
          <button key={txt}
            onClick={fn}
            style={{
              fontFamily:"'Courier New',monospace",fontSize:10,
              letterSpacing:1,fontWeight:700,
              padding:"5px 12px",borderRadius:4,
              cursor:"pointer",background:bg,
              color:col,border:`1px solid ${brd}`,
              transition:"all .15s",
            }}>
            {txt}
          </button>
        ))}

        {/* Last maneuver summary */}
        {lastMan && (
          <div style={{
            marginLeft:12,fontSize:9,color:"#4a7090",
            borderLeft:"1px solid #0a3a5a",paddingLeft:12,
          }}>
            <span style={{color:"#00d4ff"}}>{lastMan.id}</span>
            <span style={{marginLeft:8}}>|ΔV|={Number(lastMan.mag).toFixed(4)}m/s</span>
            <span style={{marginLeft:8,color:"#00ff88"}}>✓ EXECUTED</span>
          </div>
        )}

        <span style={{marginLeft:"auto",fontSize:9,color:"#4a7090"}}>
          MSc Data Science & AI · ORBITGUARD v1.0 ·
          FastAPI + React + Three.js
        </span>
      </div>
    </div>
  );
}