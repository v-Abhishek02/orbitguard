import axios from "axios";

const BASE = import.meta.env.VITE_API_URL
          || "http://localhost:8000";

export const api = axios.create({ baseURL: BASE });

export const getHealth   = ()       => api.get("/health");
export const getObjects  = ()       => api.get("/objects");
export const detect      = (seqs)   => api.post("/detect",  { sequences: seqs });
export const predict     = (s, dt)  => api.post("/predict", { state0: s, dt_min: dt });
export const avoid       = (sc, db) => api.post("/avoid",   { sc_state: sc, debris_states: db });
export const runPipeline = (body)   => api.post("/pipeline", body);