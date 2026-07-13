import { useState, useCallback } from "react";

// Global state hook — shared across all components
// In 3 months replace this with Zustand for cleaner state management
let _listeners = [];
let _state = {
  objects:        [],
  conjunctions:   [],
  meta:           {},
  selectedObject: null,
  activePanel:    "dashboard",
  alerts:         [],
  showTrails:     true,
  showCones:      true,
  showLabels:     false,
  trailObjects:   new Set(),
  filterRisk:     "ALL",
  filterType:     "ALL",
  fuel:           500,
  manCount:       0,
  lastManeuver:   null,
  apiStatus:      "connecting",
  totalTime:      0,
  paused:         false,
};

function notifyAll() {
  _listeners.forEach(fn => fn({ ..._state }));
}

export function setState(patch) {
  _state = { ..._state, ...patch };
  notifyAll();
}

export function getState() {
  return _state;
}

export function useStore() {
  const [state, setLocalState] = useState({ ..._state });

  useCallback(() => {
    const listener = (s) => setLocalState(s);
    _listeners.push(listener);
    return () => {
      _listeners = _listeners.filter(l => l !== listener);
    };
  }, [])();

  const set = useCallback((patch) => {
    setState(patch);
    setLocalState(s => ({ ...s, ...patch }));
  }, []);

  return [state, set];
}