import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { R_EARTH, RISK_COLOR } from "../../utils/orbital";

export default function Globe3D({
  objects, conjunctions,
  showCones, showTrails, showLabels,
  selectedObject, onSelectObject,
  trailObjects, filterRisk, filterType,
}) {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const stateRef = useRef({
    objects, conjunctions, showCones,
    showTrails, showLabels, selectedObject,
    trailObjects, filterRisk, filterType,
  });

  useEffect(() => {
    stateRef.current = {
      objects, conjunctions, showCones,
      showTrails, showLabels, selectedObject,
      trailObjects, filterRisk, filterType,
    };
  });

  useEffect(() => {
    const mount = mountRef.current;
    const W = mount.clientWidth, H = mount.clientHeight;

    // ── Renderer ─────────────────────────────────────
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: "high-performance",
    });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping      = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.95;
    mount.appendChild(renderer.domElement);

    // ── Scene ─────────────────────────────────────────
    const scene  = new THREE.Scene();
    scene.background = new THREE.Color("#000208");
    const camera = new THREE.PerspectiveCamera(45, W/H, 50, 300000);
    camera.position.set(0, 5000, 22000);
    camera.lookAt(0, 0, 0);

    // ── Orbit controls ────────────────────────────────
    let isDragging = false, lastX = 0, lastY = 0;
    let sph = { theta: 0.4, phi: 1.1, r: 22000 };
    const updateCam = () => {
      camera.position.set(
        sph.r * Math.sin(sph.phi) * Math.sin(sph.theta),
        sph.r * Math.cos(sph.phi),
        sph.r * Math.sin(sph.phi) * Math.cos(sph.theta),
      );
      camera.lookAt(0, 0, 0);
    };
    updateCam();
    const el = renderer.domElement;
    el.addEventListener("mousedown", e => {
      isDragging = true;
      lastX = e.clientX; lastY = e.clientY;
    });
    window.addEventListener("mouseup",   () => isDragging = false);
    window.addEventListener("mousemove", e => {
      if (!isDragging) return;
      sph.theta -= (e.clientX - lastX) * 0.004;
      sph.phi    = Math.max(0.1, Math.min(
        Math.PI - 0.1, sph.phi - (e.clientY - lastY) * 0.004
      ));
      lastX = e.clientX; lastY = e.clientY;
      updateCam();
    });
    el.addEventListener("wheel", e => {
      sph.r = Math.max(7200, Math.min(60000, sph.r + e.deltaY * 8));
      updateCam();
    });
    window.addEventListener("keydown", e => {
      if (e.key === "r" || e.key === "R") {
        sph = { theta: 0.4, phi: 1.1, r: 22000 };
        updateCam();
      }
    });

    // ── Stars ─────────────────────────────────────────
    const starGeo = new THREE.BufferGeometry();
    const starPos = new Float32Array(6000);
    for (let i = 0; i < 6000; i += 3) {
      const t = Math.random()*Math.PI*2;
      const p = Math.acos(2*Math.random()-1);
      const r = 120000 + Math.random()*30000;
      starPos[i]   = r*Math.sin(p)*Math.cos(t);
      starPos[i+1] = r*Math.sin(p)*Math.sin(t);
      starPos[i+2] = r*Math.cos(p);
    }
    starGeo.setAttribute("position", new THREE.BufferAttribute(starPos, 3));
    scene.add(new THREE.Points(starGeo, new THREE.PointsMaterial({
      color: "#ffffff", size: 70, sizeAttenuation: true, transparent: true, opacity: 0.85,
    })));

    // ── Earth with NASA textures ───────────────────────
    const loader = new THREE.TextureLoader();
    let earthMesh, cloudMesh;

    const buildEarth = (dayTex, nightTex, cloudTex, normalTex, specTex) => {
      // Earth body
      const earthGeo = new THREE.SphereGeometry(R_EARTH, 80, 80);
      const earthMat = new THREE.MeshPhongMaterial({
        map:               dayTex,
        normalMap:         normalTex,
        normalScale:       new THREE.Vector2(0.85, 0.85),
        specularMap:       specTex,
        specular:          new THREE.Color("#336699"),
        shininess:         28,
        emissiveMap:       nightTex,
        emissive:          new THREE.Color("#ffffff"),
        emissiveIntensity: 0.55,
      });
      earthMesh = new THREE.Mesh(earthGeo, earthMat);
      scene.add(earthMesh);

      // Cloud layer
      const cloudGeo = new THREE.SphereGeometry(R_EARTH * 1.005, 64, 64);
      const cloudMat = new THREE.MeshPhongMaterial({
        map:         cloudTex,
        transparent: true,
        opacity:     0.38,
        depthWrite:  false,
        blending:    THREE.NormalBlending,
      });
      cloudMesh = new THREE.Mesh(cloudGeo, cloudMat);
      scene.add(cloudMesh);
    };

    // Try NASA textures, fallback to procedural
    Promise.all([
      loadTexture(loader, "/textures/earth_day.jpg"),
      loadTexture(loader, "/textures/earth_night.jpg"),
      loadTexture(loader, "/textures/earth_clouds.jpg"),
      loadTexture(loader, "/textures/earth_normal.jpg"),
      loadTexture(loader, "/textures/earth_specular.jpg"),
    ]).then(([day, night, cloud, normal, spec]) => {
      buildEarth(day, night, cloud, normal, spec);
    }).catch(() => {
      // Procedural fallback
      const { mesh, clouds } = buildProceduralEarth();
      earthMesh  = mesh;
      cloudMesh  = clouds;
      scene.add(earthMesh);
      scene.add(cloudMesh);
    });

    // Atmosphere glow
    const atmGeo = new THREE.SphereGeometry(R_EARTH * 1.025, 48, 48);
    const atmMat = new THREE.ShaderMaterial({
      vertexShader: `
        varying vec3 vNormal;
        void main() {
          vNormal = normalize(normalMatrix * normal);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0);
        }
      `,
      fragmentShader: `
        varying vec3 vNormal;
        void main() {
          float intensity = pow(0.65 - dot(vNormal, vec3(0,0,1)), 3.0);
          gl_FragColor = vec4(0.15, 0.45, 1.0, 1.0) * intensity;
        }
      `,
      side: THREE.BackSide,
      blending: THREE.AdditiveBlending,
      transparent: true,
    });
    scene.add(new THREE.Mesh(atmGeo, atmMat));

    // ── Lighting ───────────────────────────────────────
    scene.add(new THREE.AmbientLight("#334466", 0.35));
    const sun = new THREE.DirectionalLight("#fffbe8", 2.0);
    sun.position.set(35000, 10000, 25000);
    scene.add(sun);
    const fill = new THREE.DirectionalLight("#223355", 0.20);
    fill.position.set(-25000, -8000, -15000);
    scene.add(fill);

    // ── Object points — GPU shader ─────────────────────
    let ptPoints = null, ptGeo = null;
    const buildPoints = (objs) => {
      if (ptPoints) { scene.remove(ptPoints); ptGeo?.dispose(); }
      const n    = objs.length;
      const pos  = new Float32Array(n * 3);
      const cols = new Float32Array(n * 3);
      const szs  = new Float32Array(n);
      objs.forEach((o, i) => {
        pos[i*3]=o.x; pos[i*3+1]=o.y; pos[i*3+2]=o.z;
        const c = new THREE.Color(RISK_COLOR[o.risk] || "#3b82f6");
        cols[i*3]=c.r; cols[i*3+1]=c.g; cols[i*3+2]=c.b;
        szs[i] = o.risk==="HIGH"?40:o.risk==="MED"?25:14;
      });
      ptGeo = new THREE.BufferGeometry();
      ptGeo.setAttribute("position", new THREE.BufferAttribute(pos,3));
      ptGeo.setAttribute("color",    new THREE.BufferAttribute(cols,3));
      ptGeo.setAttribute("size",     new THREE.BufferAttribute(szs,1));
      const mat = new THREE.ShaderMaterial({
        vertexShader:`
          attribute float size; attribute vec3 color;
          varying vec3 vColor;
          void main(){
            vColor=color;
            vec4 mv=modelViewMatrix*vec4(position,1.0);
            gl_PointSize=size*(3200.0/-mv.z);
            gl_Position=projectionMatrix*mv;
          }`,
        fragmentShader:`
          varying vec3 vColor;
          void main(){
            vec2 uv=gl_PointCoord-0.5;
            float d=length(uv);
            if(d>0.5)discard;
            float a=1.0-smoothstep(0.25,0.5,d);
            float glow=1.0-smoothstep(0.0,0.5,d);
            gl_FragColor=vec4(vColor+glow*0.4,a);
          }`,
        transparent:true, depthWrite:false,
        vertexColors:true,
        blending:THREE.AdditiveBlending,
      });
      ptPoints = new THREE.Points(ptGeo, mat);
      ptPoints._objects = objs;
      scene.add(ptPoints);
    };
    buildPoints(objects);

    // ── Conjunction lines ──────────────────────────────
    const conjGroup = new THREE.Group();
    conjunctions.forEach(c => {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(
        new Float32Array([c.x1,c.y1,c.z1,c.x2,c.y2,c.z2]),3
      ));
      conjGroup.add(new THREE.Line(g, new THREE.LineBasicMaterial({
        color:"#ef4444", transparent:true, opacity:0.4,
      })));
    });
    scene.add(conjGroup);

    // ── Orbital trails group ───────────────────────────
    const trailGroup = new THREE.Group();
    scene.add(trailGroup);

    // ── PINN cones ─────────────────────────────────────
    const coneGroup = new THREE.Group();
    scene.add(coneGroup);

    // ── Spacecraft ─────────────────────────────────────
    const scGroup = new THREE.Group();
    const scR     = R_EARTH + 420;
    // Body
    scGroup.add(new THREE.Mesh(
      new THREE.ConeGeometry(85, 240, 6),
      new THREE.MeshPhongMaterial({
        color:"#00ff88", emissive:"#004422", emissiveIntensity:0.5
      })
    ));
    // Solar panels
    const panMat = new THREE.MeshPhongMaterial({
      color:"#1d4ed8", emissive:"#0d2070", emissiveIntensity:0.3
    });
    [-330,330].forEach(px => {
      const p = new THREE.Mesh(new THREE.BoxGeometry(520,8,130), panMat);
      p.position.set(px,0,0);
      scGroup.add(p);
    });
    // Glow halo
    scGroup.add(new THREE.Mesh(
      new THREE.SphereGeometry(280,16,16),
      new THREE.MeshBasicMaterial({
        color:"#00ff88", transparent:true,
        opacity:0.06, side:THREE.BackSide
      })
    ));
    scene.add(scGroup);

    // ── Raycaster for click selection ──────────────────
    const raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 180;

    el.addEventListener("click", e => {
      if (!ptPoints) return;
      const rect  = el.getBoundingClientRect();
      const mouse = new THREE.Vector2(
        ((e.clientX-rect.left)/rect.width)*2-1,
        -((e.clientY-rect.top)/rect.height)*2+1,
      );
      raycaster.setFromCamera(mouse, camera);
      const hits = raycaster.intersectObject(ptPoints);
      if (hits.length > 0) {
        const idx = hits[0].index;
        const obj = ptPoints._objects?.[idx];
        if (obj) onSelectObject(obj);
      } else {
        onSelectObject(null);
      }
    });

    // ── Hover tooltip ──────────────────────────────────
    let hoverTimeout;
    el.addEventListener("mousemove", e => {
      clearTimeout(hoverTimeout);
      hoverTimeout = setTimeout(() => {
        if (!ptPoints) return;
        const rect  = el.getBoundingClientRect();
        const mouse = new THREE.Vector2(
          ((e.clientX-rect.left)/rect.width)*2-1,
          -((e.clientY-rect.top)/rect.height)*2+1,
        );
        raycaster.setFromCamera(mouse, camera);
        const hits = raycaster.intersectObject(ptPoints);
        const tip  = document.getElementById("og-tooltip");
        if (!tip) return;
        if (hits.length > 0) {
          const obj = ptPoints._objects?.[hits[0].index];
          if (obj) {
            tip.style.display  = "block";
            tip.style.left     = (e.clientX+14)+"px";
            tip.style.top      = (e.clientY-8)+"px";
            tip.innerHTML      =
              `<div style="color:#00d4ff;font-weight:700">NORAD ${obj.id}</div>
               <div style="color:#7ba8c0">${obj.type} · ${obj.alt}km</div>
               <div style="color:${RISK_COLOR[obj.risk]};font-weight:700">${obj.risk} RISK</div>`;
          }
        } else {
          tip.style.display = "none";
        }
      }, 80);
    });

    sceneRef.current = {
      scene, camera, renderer, earthMesh,
      cloudMesh, coneGroup, trailGroup,
      ptPoints, ptGeo, buildPoints, scGroup,
    };

    // ── Animate ────────────────────────────────────────
    let animId, t = 0;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      t += 0.016;
      if (earthMesh)  earthMesh.rotation.y  += 0.000075;
      if (cloudMesh)  cloudMesh.rotation.y  += 0.000110;

      // Orbit spacecraft (ISS: 92-min period)
      scGroup.position.set(
        scR * Math.cos(t * 0.0073),
        scR * 0.098 * Math.sin(t * 0.0073 * 2),
        scR * Math.sin(t * 0.0073),
      );
      scGroup.lookAt(0, 0, 0);

      // Pulse PINN cones
      coneGroup.children.forEach((c, i) => {
        if (c.material?.opacity !== undefined)
          c.material.opacity = 0.06+0.05*Math.abs(Math.sin(t*1.4+i));
      });

      renderer.render(scene, camera);
    };
    animate();

    // ── Resize ─────────────────────────────────────────
    const ro = new ResizeObserver(() => {
      const w = mount.clientWidth, h = mount.clientHeight;
      camera.aspect = w/h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    });
    ro.observe(mount);

    return () => {
      cancelAnimationFrame(animId);
      ro.disconnect();
      clearTimeout(hoverTimeout);
      renderer.dispose();
      if (mount.contains(renderer.domElement))
        mount.removeChild(renderer.domElement);
    };
  }, []);

  // Rebuild points when objects/filter change
  useEffect(() => {
    if (!sceneRef.current?.buildPoints) return;
    let filtered = objects;
    if (filterRisk !== "ALL")
      filtered = filtered.filter(o => o.risk === filterRisk);
    if (filterType !== "ALL")
      filtered = filtered.filter(o => o.type === filterType);
    sceneRef.current.buildPoints(filtered);
  }, [objects, filterRisk, filterType]);

  // PINN cones when showCones changes
  useEffect(() => {
    const { coneGroup, scene: sc } = sceneRef.current || {};
    if (!coneGroup) return;
    coneGroup.visible = showCones;
    if (showCones && coneGroup.children.length === 0) {
      objects.filter(o=>o.risk==="HIGH").slice(0,18).forEach(o => {
        const p3  = new THREE.Vector3(o.x,o.y,o.z);
        const vel = new THREE.Vector3(o.vx||0.5,o.vy||0.5,o.vz||0.5).normalize();
        const end = p3.clone().addScaledVector(vel,950);
        const cone = new THREE.Mesh(
          new THREE.ConeGeometry(140,950,8,1,true),
          new THREE.MeshBasicMaterial({
            color:"#00d4ff",transparent:true,opacity:0.09,side:THREE.DoubleSide
          })
        );
        cone.position.copy(p3).addScaledVector(vel,475);
        cone.lookAt(end); cone.rotateX(Math.PI/2);
        coneGroup.add(cone);
        const lg=new THREE.BufferGeometry().setFromPoints([p3,end]);
        const lm=new THREE.LineDashedMaterial({
          color:"#00d4ff",transparent:true,opacity:0.5,dashSize:160,gapSize:90
        });
        const ln=new THREE.Line(lg,lm); ln.computeLineDistances();
        coneGroup.add(ln);
      });
    }
  }, [showCones, objects]);

  // Orbital trails for selected / watchlist objects
  useEffect(() => {
    const { trailGroup } = sceneRef.current || {};
    if (!trailGroup) return;
    trailGroup.clear();
    if (!showTrails) return;
    const toShow = selectedObject
      ? [selectedObject, ...objects.filter(o=>trailObjects.has(o.id))]
      : objects.filter(o=>trailObjects.has(o.id));
    toShow.slice(0,20).forEach(o => {
      const pts = [];
      const r   = Math.sqrt(o.x**2+o.y**2+o.z**2);
      const n   = 120;
      for (let i=0; i<=n; i++) {
        const angle = (i/n)*Math.PI*2;
        pts.push(new THREE.Vector3(
          r*Math.cos(angle), o.y/r*r*0.1, r*Math.sin(angle)
        ));
      }
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      const col = o.id===selectedObject?.id?"#facc15":RISK_COLOR[o.risk];
      trailGroup.add(new THREE.Line(geo, new THREE.LineBasicMaterial({
        color:col, transparent:true, opacity:o.id===selectedObject?.id?0.8:0.35
      })));
    });
  }, [selectedObject, trailObjects, showTrails, objects]);

  return (
    <>
      <div ref={mountRef}
           style={{position:"absolute",inset:0,width:"100%",height:"100%"}}/>
      {/* Hover tooltip */}
      <div id="og-tooltip" style={{
        position:"fixed", display:"none",
        background:"#020e1fee", border:"1px solid #00d4ff66",
        borderRadius:4, padding:"6px 10px",
        fontFamily:"'Courier New',monospace",
        fontSize:9, pointerEvents:"none", zIndex:50,
        backdropFilter:"blur(8px)",
      }}/>
    </>
  );
}

// Helper: load texture with timeout fallback
function loadTexture(loader, url) {
  return new Promise((res, rej) => {
    const timer = setTimeout(() => rej("timeout"), 8000);
    loader.load(
      url,
      tex => { clearTimeout(timer); res(tex); },
      undefined,
      () => { clearTimeout(timer); rej("error"); }
    );
  });
}

// Procedural Earth fallback
function buildProceduralEarth() {
  const tc=document.createElement("canvas");
  tc.width=2048; tc.height=1024;
  const tctx=tc.getContext("2d");
  const og=tctx.createLinearGradient(0,0,0,1024);
  og.addColorStop(0,"#0d3a70");og.addColorStop(0.5,"#0e4f90");og.addColorStop(1,"#083060");
  tctx.fillStyle=og; tctx.fillRect(0,0,2048,1024);
  tctx.fillStyle="#1a6b35";
  [[900,250,280,350],[1150,280,380,420],[650,420,200,320],
   [1350,360,260,300],[820,680,160,130],[480,290,110,190],
   [1700,290,130,170]].forEach(([x,y,rw,rh])=>{
    tctx.beginPath();tctx.ellipse(x,y,rw,rh,0.1,0,Math.PI*2);tctx.fill();
  });
  tctx.fillStyle="#ddeeff";
  tctx.fillRect(0,0,2048,50); tctx.fillRect(0,970,2048,54);
  const mesh=new THREE.Mesh(
    new THREE.SphereGeometry(6371,64,64),
    new THREE.MeshPhongMaterial({
      map:new THREE.CanvasTexture(tc),
      specular:new THREE.Color("#224488"),shininess:18,
    })
  );
  const cc=document.createElement("canvas");
  cc.width=1024;cc.height=512;
  const cctx=cc.getContext("2d");
  cctx.fillStyle="rgba(0,0,0,0)";cctx.fillRect(0,0,1024,512);
  cctx.fillStyle="rgba(255,255,255,0.18)";
  for(let i=0;i<40;i++){
    cctx.beginPath();
    cctx.ellipse(Math.random()*1024,Math.random()*512,
      28+Math.random()*80,12+Math.random()*28,
      Math.random()*Math.PI,0,Math.PI*2);
    cctx.fill();
  }
  const clouds=new THREE.Mesh(
    new THREE.SphereGeometry(6371*1.006,48,48),
    new THREE.MeshPhongMaterial({
      map:new THREE.CanvasTexture(cc),transparent:true,opacity:0.22,depthWrite:false
    })
  );
  return { mesh, clouds };
}