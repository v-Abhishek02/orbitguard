import { useEffect, useRef } from "react";
import * as THREE from "three";

const R_EARTH = 6371;
const RISK_COLOR = {
  HIGH: new THREE.Color("#ef4444"),
  MED:  new THREE.Color("#f97316"),
  LOW:  new THREE.Color("#3b82f6"),
};

export default function Globe({ objects, conjunctions, showCones }) {
  const mountRef = useRef(null);

  useEffect(() => {
    if (!objects.length) return;
    const mount = mountRef.current;
    const W = mount.clientWidth, H = mount.clientHeight;

    // ── Renderer ───────────────────────────────────────
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: "high-performance",
    });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.9;
    mount.appendChild(renderer.domElement);

    // ── Scene + Camera ─────────────────────────────────
    const scene  = new THREE.Scene();
    scene.background = new THREE.Color("#010508");
    const camera = new THREE.PerspectiveCamera(45, W/H, 100, 200000);
    camera.position.set(0, 4000, 20000);
    camera.lookAt(0, 0, 0);

    // ── Manual orbit controls ──────────────────────────
    let drag = false, lastX = 0, lastY = 0;
    let sph  = { theta: 0.3, phi: 1.2, r: 20000 };
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
      drag = true; lastX = e.clientX; lastY = e.clientY;
    });
    window.addEventListener("mouseup",   () => drag = false);
    window.addEventListener("mousemove", e => {
      if (!drag) return;
      sph.theta -= (e.clientX - lastX) * 0.004;
      sph.phi    = Math.max(0.1, Math.min(
        Math.PI - 0.1, sph.phi - (e.clientY - lastY) * 0.004
      ));
      lastX = e.clientX; lastY = e.clientY;
      updateCam();
    });
    el.addEventListener("wheel", e => {
      sph.r = Math.max(7500, Math.min(55000, sph.r + e.deltaY * 7));
      updateCam();
    });

    // ── Stars ──────────────────────────────────────────
    const sGeo = new THREE.BufferGeometry();
    const sPos = new Float32Array(3000);
    for (let i = 0; i < 3000; i += 3) {
      const t = Math.random() * Math.PI * 2;
      const p = Math.acos(2 * Math.random() - 1);
      const r = 85000 + Math.random() * 10000;
      sPos[i]   = r * Math.sin(p) * Math.cos(t);
      sPos[i+1] = r * Math.sin(p) * Math.sin(t);
      sPos[i+2] = r * Math.cos(p);
    }
    sGeo.setAttribute("position", new THREE.BufferAttribute(sPos, 3));
    scene.add(new THREE.Points(sGeo, new THREE.PointsMaterial({
      color: "#ffffff", size: 55, sizeAttenuation: true,
    })));

    // ── Earth texture (procedural) ─────────────────────
    const tc   = document.createElement("canvas");
    tc.width   = 2048; tc.height = 1024;
    const tctx = tc.getContext("2d");

    // Ocean gradient
    const og = tctx.createLinearGradient(0, 0, 0, 1024);
    og.addColorStop(0,   "#0d3a70");
    og.addColorStop(0.5, "#0e4f90");
    og.addColorStop(1,   "#083060");
    tctx.fillStyle = og;
    tctx.fillRect(0, 0, 2048, 1024);

    // Continents
    tctx.fillStyle = "#1a6b35";
    [
      [900, 250, 280, 350, 0.1],
      [1150,280, 380, 420, 0.15],
      [650, 420, 200, 320, 0.18],
      [1350,360, 260, 300, 0.10],
      [820, 680, 160, 130, 0.08],
      [480, 290, 110, 190, 0.05],
      [1700,290, 130, 170, 0.12],
    ].forEach(([x, y, rw, rh, rot]) => {
      tctx.beginPath();
      tctx.ellipse(x, y, rw, rh, rot, 0, Math.PI * 2);
      tctx.fill();
      tctx.strokeStyle = "#0f4a22";
      tctx.lineWidth   = 2;
      tctx.stroke();
    });

    // Ice caps
    tctx.fillStyle = "#ddeeff";
    tctx.fillRect(0, 0,    2048, 55);
    tctx.fillRect(0, 965,  2048, 59);

    // Lat/lon grid
    tctx.strokeStyle = "rgba(0,200,255,0.07)";
    tctx.lineWidth   = 1;
    for (let lon = 0; lon < 2048; lon += 2048/12) {
      tctx.beginPath();
      tctx.moveTo(lon, 0); tctx.lineTo(lon, 1024);
      tctx.stroke();
    }
    for (let lat = 0; lat < 1024; lat += 1024/6) {
      tctx.beginPath();
      tctx.moveTo(0, lat); tctx.lineTo(2048, lat);
      tctx.stroke();
    }

    const earthTex = new THREE.CanvasTexture(tc);
    const earth = new THREE.Mesh(
      new THREE.SphereGeometry(R_EARTH, 64, 64),
      new THREE.MeshPhongMaterial({
        map:               earthTex,
        specular:          new THREE.Color("#224488"),
        shininess:         18,
        emissive:          new THREE.Color("#030818"),
        emissiveIntensity: 0.12,
      })
    );
    scene.add(earth);

    // Atmosphere shell
    scene.add(new THREE.Mesh(
      new THREE.SphereGeometry(R_EARTH * 1.02, 48, 48),
      new THREE.MeshPhongMaterial({
        color: "#1a6aff", transparent: true,
        opacity: 0.055, side: THREE.FrontSide,
      })
    ));

    // Cloud layer
    const cc2   = document.createElement("canvas");
    cc2.width   = 1024; cc2.height = 512;
    const cctx2 = cc2.getContext("2d");
    cctx2.fillStyle = "rgba(0,0,0,0)";
    cctx2.fillRect(0, 0, 1024, 512);
    cctx2.fillStyle = "rgba(255,255,255,0.16)";
    for (let i = 0; i < 38; i++) {
      const cx = Math.random()*1024, cy = Math.random()*512;
      cctx2.beginPath();
      cctx2.ellipse(
        cx, cy,
        28 + Math.random()*80,
        12 + Math.random()*28,
        Math.random()*Math.PI, 0, Math.PI*2
      );
      cctx2.fill();
    }
    const cloudMesh = new THREE.Mesh(
      new THREE.SphereGeometry(R_EARTH * 1.008, 48, 48),
      new THREE.MeshPhongMaterial({
        map: new THREE.CanvasTexture(cc2),
        transparent: true, opacity: 0.20,
        depthWrite: false,
      })
    );
    scene.add(cloudMesh);

    // ── Lighting ───────────────────────────────────────
    scene.add(new THREE.AmbientLight("#334466", 0.40));
    const sun = new THREE.DirectionalLight("#fffbe8", 1.8);
    sun.position.set(30000, 8000, 20000);
    scene.add(sun);
    const fill = new THREE.DirectionalLight("#223355", 0.25);
    fill.position.set(-20000, -5000, -10000);
    scene.add(fill);

    // ── All objects — GPU shader points ───────────────
    const n    = objects.length;
    const pos  = new Float32Array(n * 3);
    const cols = new Float32Array(n * 3);
    const szs  = new Float32Array(n);

    objects.forEach((o, i) => {
      pos[i*3]   = o.x;
      pos[i*3+1] = o.y;
      pos[i*3+2] = o.z;
      const c = RISK_COLOR[o.risk] || new THREE.Color("#888");
      cols[i*3]   = c.r;
      cols[i*3+1] = c.g;
      cols[i*3+2] = c.b;
      szs[i] = o.risk==="HIGH" ? 38 : o.risk==="MED" ? 24 : 13;
    });

    const ptGeo = new THREE.BufferGeometry();
    ptGeo.setAttribute("position", new THREE.BufferAttribute(pos,  3));
    ptGeo.setAttribute("color",    new THREE.BufferAttribute(cols, 3));
    ptGeo.setAttribute("size",     new THREE.BufferAttribute(szs,  1));

    const ptMat = new THREE.ShaderMaterial({
      vertexShader: `
        attribute float size;
        attribute vec3  color;
        varying vec3 vColor;
        void main() {
          vColor = color;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = size * (3000.0 / -mv.z);
          gl_Position  = projectionMatrix * mv;
        }
      `,
      fragmentShader: `
        varying vec3 vColor;
        void main() {
          vec2  uv = gl_PointCoord - 0.5;
          float d  = length(uv);
          if (d > 0.5) discard;
          float a = 1.0 - smoothstep(0.28, 0.5, d);
          gl_FragColor = vec4(vColor, a);
        }
      `,
      transparent: true, depthWrite: false,
      vertexColors: true,
      blending: THREE.AdditiveBlending,
    });

    scene.add(new THREE.Points(ptGeo, ptMat));

    // ── Conjunction lines ──────────────────────────────
    conjunctions.forEach(c => {
      const lg = new THREE.BufferGeometry();
      lg.setAttribute("position", new THREE.BufferAttribute(
        new Float32Array([c.x1,c.y1,c.z1, c.x2,c.y2,c.z2]), 3
      ));
      scene.add(new THREE.Line(lg, new THREE.LineBasicMaterial({
        color: "#ef4444", transparent: true, opacity: 0.35,
      })));
    });

    // ── PINN prediction cones ──────────────────────────
    const coneGroup = new THREE.Group();
    if (showCones) {
      objects
        .filter(o => o.risk === "HIGH")
        .slice(0, 15)
        .forEach(o => {
          const p3  = new THREE.Vector3(o.x, o.y, o.z);
          const vel = new THREE.Vector3(o.vx||0.1, o.vy||0.1, o.vz||0.1).normalize();
          const end = p3.clone().addScaledVector(vel, 900);

          // Cone body
          const cone = new THREE.Mesh(
            new THREE.ConeGeometry(130, 900, 8, 1, true),
            new THREE.MeshBasicMaterial({
              color: "#00d4ff", transparent: true,
              opacity: 0.10, side: THREE.DoubleSide,
            })
          );
          cone.position.copy(p3).addScaledVector(vel, 450);
          cone.lookAt(end);
          cone.rotateX(Math.PI / 2);
          coneGroup.add(cone);

          // Prediction line
          const lg  = new THREE.BufferGeometry().setFromPoints([p3, end]);
          const lm  = new THREE.LineDashedMaterial({
            color: "#00d4ff", transparent: true,
            opacity: 0.5, dashSize: 150, gapSize: 80,
          });
          const ln = new THREE.Line(lg, lm);
          ln.computeLineDistances();
          coneGroup.add(ln);

          // Uncertainty sphere
          const sp = new THREE.Mesh(
            new THREE.SphereGeometry(120, 8, 8),
            new THREE.MeshBasicMaterial({
              color: "#00d4ff", transparent: true,
              opacity: 0.22, wireframe: true,
            })
          );
          sp.position.copy(end);
          coneGroup.add(sp);
        });
    }
    scene.add(coneGroup);

    // ── Spacecraft marker ──────────────────────────────
    const scGroup = new THREE.Group();
    const scR     = R_EARTH + 420;
    scGroup.add(new THREE.Mesh(
      new THREE.ConeGeometry(80, 220, 6),
      new THREE.MeshPhongMaterial({
        color: "#00ff88", emissive: "#004422",
        emissiveIntensity: 0.4,
      })
    ));
    // Solar panels
    const panMat = new THREE.MeshPhongMaterial({ color: "#2255bb" });
    [-320, 320].forEach(px => {
      const p = new THREE.Mesh(new THREE.BoxGeometry(500,8,120), panMat);
      p.position.set(px, 0, 0);
      scGroup.add(p);
    });
    // Glow
    scGroup.add(new THREE.Mesh(
      new THREE.SphereGeometry(260, 16, 16),
      new THREE.MeshBasicMaterial({
        color: "#00ff88", transparent: true,
        opacity: 0.07, side: THREE.BackSide,
      })
    ));
    scGroup.position.set(scR, 0, 0);
    scene.add(scGroup);

    // ── Animation loop ─────────────────────────────────
    let animId, t = 0;
    const loop = () => {
      animId = requestAnimationFrame(loop);
      t += 0.016;
      earth.rotation.y     += 0.000080;
      cloudMesh.rotation.y += 0.000115;

      // Orbit spacecraft
      scGroup.position.set(
        scR * Math.cos(t * 0.007),
        scR * 0.1 * Math.sin(t * 0.007),
        scR * Math.sin(t * 0.007),
      );
      scGroup.lookAt(0, 0, 0);

      // Pulse cones
      coneGroup.children.forEach((c, i) => {
        if (c.material && c.material.opacity !== undefined) {
          c.material.opacity =
            0.07 + 0.05 * Math.abs(Math.sin(t * 1.5 + i));
        }
      });

      renderer.render(scene, camera);
    };
    loop();

    // ── Resize observer ────────────────────────────────
    const ro = new ResizeObserver(() => {
      const w = mount.clientWidth, h = mount.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    });
    ro.observe(mount);

    return () => {
      cancelAnimationFrame(animId);
      ro.disconnect();
      renderer.dispose();
      if (mount.contains(renderer.domElement))
        mount.removeChild(renderer.domElement);
    };
  }, [objects, conjunctions, showCones]);

  return (
    <div
      ref={mountRef}
      style={{
        position: "absolute", inset: 0,
        width: "100%", height: "100%",
      }}
    />
  );
}