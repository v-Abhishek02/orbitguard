import { useState, useEffect } from "react";
import { getObjects, getHealth } from "../api";

export function useOrbitalData() {
  const [objects,      setObjects]      = useState([]);
  const [conjunctions, setConjunctions] = useState([]);
  const [meta,         setMeta]         = useState({});
  const [apiStatus,    setApiStatus]    = useState("connecting");
  const [loading,      setLoading]      = useState(true);
  const [error,        setError]        = useState(null);

  useEffect(() => {
    const load = async () => {
      try {
        const [, oRes] = await Promise.all([
          getHealth(), getObjects()
        ]);
        setObjects(oRes.data.objects);
        setConjunctions(oRes.data.conjunctions);
        setMeta(oRes.data.meta);
        setApiStatus("online");
      } catch (e) {
        setError(e.message);
        setApiStatus("offline");
        // Synthetic fallback — 5000 objects for performance
        const syn = generateSynthetic(5000);
        setObjects(syn.objects);
        setConjunctions(syn.conj);
        setMeta({
          high: syn.objects.filter(o=>o.risk==="HIGH").length,
          med:  syn.objects.filter(o=>o.risk==="MED").length,
          low:  syn.objects.filter(o=>o.risk==="LOW").length,
          total: syn.objects.length,
          conjunction_events: 209285,
          pinn_rmse_km: 48.27,
          det_f1: 0.9677,
          drl_success: 1.0,
          latency_ms: 131.7,
        });
      }
      setLoading(false);
    };
    load();
  }, []);

  return { objects, conjunctions, meta,
           apiStatus, loading, error };
}

function generateSynthetic(n) {
  const objects = [];
  const risks   = ["HIGH","HIGH","MED","MED","MED",
                    "LOW","LOW","LOW","LOW","LOW"];
  const types   = ["DEBRIS","DEBRIS","DEBRIS",
                    "PAYLOAD","ROCKET BODY"];
  let   rng     = 42;
  const rand    = () => {
    rng = (rng * 16807) % 2147483647;
    return (rng - 1) / 2147483646;
  };
  for (let i = 0; i < n; i++) {
    const theta = rand() * Math.PI * 2;
    const phi   = Math.acos(2*rand() - 1);
    const r     = 6371 + 200 + rand() * 1800;
    objects.push({
      id:   20000 + i,
      x:    r * Math.sin(phi) * Math.cos(theta),
      y:    r * Math.cos(phi),
      z:    r * Math.sin(phi) * Math.sin(theta),
      vx:   (rand()-0.5)*16, vy:(rand()-0.5)*16,
      vz:   (rand()-0.5)*16,
      alt:  Math.round(r - 6371),
      risk: risks[Math.floor(rand()*10)],
      type: types[Math.floor(rand()*5)],
    });
  }
  const conj = [];
  for (let i = 0; i < 40; i++) {
    const a = objects[i], b = objects[i+100];
    conj.push({
      obj1:i+20000, obj2:i+20100,
      miss:(rand()*4).toFixed(3),
      x1:a.x,y1:a.y,z1:a.z,
      x2:b.x,y2:b.y,z2:b.z,
    });
  }
  return { objects, conj };
}