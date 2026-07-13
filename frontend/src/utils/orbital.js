// Orbital mechanics utilities
// Used across the app for consistent calculations

export const R_EARTH = 6371;     // km
export const GM      = 398600.4418; // km³/s²
export const J2      = 1.08263e-3;

/**
 * Convert Cartesian ECI to orbital elements (simplified)
 * state = [x,y,z,vx,vy,vz] in km / km/s
 */
export function stateToElements(state) {
  const [x,y,z,vx,vy,vz] = state;
  const r = Math.sqrt(x**2 + y**2 + z**2);
  const v = Math.sqrt(vx**2 + vy**2 + vz**2);

  // Specific orbital energy
  const energy = v**2/2 - GM/r;

  // Semi-major axis
  const a = -GM / (2 * energy);

  // Orbital period (minutes)
  const T = 2 * Math.PI * Math.sqrt(a**3 / GM) / 60;

  // Mean motion (rev/day)
  const n = 1440 / T;

  // Angular momentum
  const hx = y*vz - z*vy;
  const hy = z*vx - x*vz;
  const hz = x*vy - y*vx;
  const h  = Math.sqrt(hx**2 + hy**2 + hz**2);

  // Inclination
  const incl = Math.acos(hz / h) * 180 / Math.PI;

  // Eccentricity vector
  const ex = (v**2/GM - 1/r)*x - (x*vx+y*vy+z*vz)/GM*vx;
  const ey = (v**2/GM - 1/r)*y - (x*vx+y*vy+z*vz)/GM*vy;
  const ecc = Math.sqrt(ex**2 + ey**2);

  // Altitude
  const alt = r - R_EARTH;

  // Apogee / Perigee
  const apo  = a * (1 + ecc) - R_EARTH;
  const peri = a * (1 - ecc) - R_EARTH;

  return {
    sma:        Math.round(a),
    period_min: Math.round(T * 10) / 10,
    mean_motion:Math.round(n * 10000) / 10000,
    inclination:Math.round(incl * 100) / 100,
    eccentricity:Math.round(ecc * 1e7) / 1e7,
    altitude_km: Math.round(alt * 10) / 10,
    apogee_km:   Math.round(apo),
    perigee_km:  Math.round(peri),
    velocity_kms:Math.round(v * 100) / 100,
  };
}

/**
 * ECI position → geographic lat/lon
 * Approximate (ignores GMST for now)
 */
export function eciToGeo(x, y, z) {
  const r   = Math.sqrt(x**2 + y**2 + z**2);
  const lat = Math.asin(z / r) * 180 / Math.PI;
  const lon = Math.atan2(y, x) * 180 / Math.PI;
  return { lat: Math.round(lat*100)/100,
           lon: Math.round(lon*100)/100 };
}

/**
 * Format time T+HH:MM:SS
 */
export function fmtMET(s) {
  const ss  = Math.floor(s);
  const h   = Math.floor(ss / 3600);
  const m   = Math.floor((ss % 3600) / 60);
  const sec = ss % 60;
  return `T+${String(h).padStart(2,"0")}:`
       + `${String(m).padStart(2,"0")}:`
       + `${String(sec).padStart(2,"0")}`;
}

/**
 * Risk colour mapping
 */
export const RISK_COLOR = {
  HIGH: "#ef4444",
  MED:  "#f97316",
  LOW:  "#3b82f6",
};

export const RISK_BG = {
  HIGH: "#3a0a0a",
  MED:  "#2a1a00",
  LOW:  "#0a2010",
};