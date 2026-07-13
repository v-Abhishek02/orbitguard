import { useEffect, useRef } from "react";

/**
 * Simulates real-time orbital propagation.
 * In 3 months: replace with WebSocket from backend
 * that streams live TLE updates every 30 seconds.
 */
export function useRealtime({
  paused, onTick, onAlert, interval = 500
}) {
  const timerRef = useRef(null);

  const AUTO_ALERTS = [
    ["CNN+LSTM: HIGH risk conjunction flagged","crit"],
    ["PINN: 30-min trajectory updated (RMSE=48.3km)","info"],
    ["PPO: avoidance maneuver computed","info"],
    ["Space-Track TLE sync — 27,334 objects","info"],
    ["Conjunction alert: PCA 0.017km — action required","crit"],
    ["Module F1=0.9677 ✓ RMSE=48.3km ✓ Success=100%","info"],
    ["New debris event: fragmentation detected","warn"],
  ];

  let alertIdx = 0;
  let ticks    = 0;

  useEffect(() => {
    if (paused) {
      clearInterval(timerRef.current);
      return;
    }
    timerRef.current = setInterval(() => {
      ticks++;
      onTick(ticks);
      // Alert every 12 ticks (~6 seconds)
      if (ticks % 12 === 0) {
        const [msg, type] =
          AUTO_ALERTS[alertIdx % AUTO_ALERTS.length];
        alertIdx++;
        onAlert(msg, type);
      }
    }, interval);
    return () => clearInterval(timerRef.current);
  }, [paused, onTick, onAlert, interval]);
}