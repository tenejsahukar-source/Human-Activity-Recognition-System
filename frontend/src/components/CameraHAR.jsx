/**
 * CameraHAR.jsx
 * ─────────────
 * Drop-in React component that:
 *   1. Opens the webcam
 *   2. Captures frames every ~400 ms
 *   3. Sends them via WebSocket to the FastAPI backend
 *   4. Displays live activity, confidence, and next-action prediction
 *
 * Props
 * ─────
 *   wsUrl  – WebSocket URL  (default: ws://localhost:8000/ws/camera)
 *
 * Usage
 * ─────
 *   import CameraHAR from "./CameraHAR";
 *   <CameraHAR wsUrl="ws://localhost:8000/ws/camera" />
 */

import { useRef, useState, useEffect, useCallback } from "react";

const CAPTURE_INTERVAL_MS = 400;
const JPEG_QUALITY        = 0.75;

export default function CameraHAR({ wsUrl = "ws://localhost:8000/ws/camera" }) {
  const videoRef   = useRef(null);
  const canvasRef  = useRef(null);
  const wsRef      = useRef(null);
  const timerRef   = useRef(null);

  const [streaming,  setStreaming]  = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error,      setError]      = useState(null);
  const [wsStatus,   setWsStatus]   = useState("disconnected"); // connecting | open | closed | error

  // ── Open camera ────────────────────────────────────────────────────────────
  const startCamera = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    } catch (err) {
      setError("Camera access denied: " + err.message);
    }
  }, []);

  // ── Stop camera ────────────────────────────────────────────────────────────
  const stopCamera = useCallback(() => {
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(t => t.stop());
      videoRef.current.srcObject = null;
    }
  }, []);

  // ── Connect WebSocket ──────────────────────────────────────────────────────
  const connectWs = useCallback(() => {
    if (wsRef.current) wsRef.current.close();
    setWsStatus("connecting");
    const ws = new WebSocket(wsUrl);

    ws.onopen  = () => setWsStatus("open");
    ws.onclose = () => setWsStatus("closed");
    ws.onerror = () => setWsStatus("error");

    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        if (data.error) { setError(data.error); return; }
        setPrediction(data);
        setError(null);
      } catch { /* ignore parse errors */ }
    };

    wsRef.current = ws;
  }, [wsUrl]);

  // ── Frame capture loop ─────────────────────────────────────────────────────
  const captureLoop = useCallback(() => {
    if (!canvasRef.current || !videoRef.current || !wsRef.current) return;
    if (wsRef.current.readyState !== WebSocket.OPEN) return;

    const video  = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const b64 = canvas.toDataURL("image/jpeg", JPEG_QUALITY);
    wsRef.current.send(JSON.stringify({ type: "frame", image: b64 }));
  }, []);

  // ── Start / Stop ───────────────────────────────────────────────────────────
  const start = useCallback(async () => {
    await startCamera();
    connectWs();
    timerRef.current = setInterval(captureLoop, CAPTURE_INTERVAL_MS);
    setStreaming(true);
  }, [startCamera, connectWs, captureLoop]);

  const stop = useCallback(() => {
    clearInterval(timerRef.current);
    if (wsRef.current) wsRef.current.close();
    stopCamera();
    setStreaming(false);
    setPrediction(null);
  }, [stopCamera]);

  // Cleanup on unmount
  useEffect(() => () => stop(), [stop]);

  // ── Confidence bar color ───────────────────────────────────────────────────
  const confColor = (c) =>
    c > 0.75 ? "#22c55e" : c > 0.50 ? "#eab308" : "#ef4444";

  const conf = prediction?.confidence ?? 0;

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>🎥 Live Activity Recognition</h2>

      {/* Status pill */}
      <div style={{ ...styles.pill, background: statusColor(wsStatus) }}>
        WS: {wsStatus}
      </div>

      {/* Video + canvas (canvas is hidden, used only for capture) */}
      <video ref={videoRef} style={styles.video} muted playsInline />
      <canvas ref={canvasRef} style={{ display: "none" }} />

      {/* Controls */}
      <div style={styles.controls}>
        {!streaming ? (
          <button style={styles.btn} onClick={start}>▶ Start</button>
        ) : (
          <button style={{ ...styles.btn, background: "#dc2626" }} onClick={stop}>⏹ Stop</button>
        )}
      </div>

      {/* Error */}
      {error && <p style={styles.error}>{error}</p>}

      {/* Prediction card */}
      {prediction && (
        <div style={styles.card}>
          {!prediction.pose_detected ? (
            <p style={{ color: "#94a3b8", textAlign: "center" }}>
              👤 No person detected
            </p>
          ) : (
            <>
              <Row label="🏃 Activity" value={prediction.activity} big />

              {/* Confidence bar */}
              <div style={styles.barWrap}>
                <span style={styles.barLabel}>📊 Confidence</span>
                <div style={styles.barBg}>
                  <div style={{
                    ...styles.barFill,
                    width: `${(conf * 100).toFixed(1)}%`,
                    background: confColor(conf),
                  }} />
                </div>
                <span style={{ color: confColor(conf), fontWeight: 700 }}>
                  {(conf * 100).toFixed(1)}%
                </span>
              </div>

              <Row label="🔮 Next Action" value={prediction.next_action} />

              {/* All probabilities */}
              {prediction.all_probabilities && (
                <div style={styles.probs}>
                  <p style={styles.probTitle}>All Probabilities</p>
                  {Object.entries(prediction.all_probabilities)
                    .sort(([, a], [, b]) => b - a)
                    .map(([act, p]) => (
                      <div key={act} style={styles.probRow}>
                        <span style={{ color: "#94a3b8", width: 110, display: "inline-block" }}>{act}</span>
                        <div style={{ ...styles.barBg, flex: 1, height: 8 }}>
                          <div style={{
                            ...styles.barFill,
                            width: `${(p * 100).toFixed(1)}%`,
                            height: 8,
                            background: "#6366f1",
                          }} />
                        </div>
                        <span style={{ color: "#e2e8f0", marginLeft: 6, width: 44, textAlign: "right", fontSize: 12 }}>
                          {(p * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                </div>
              )}

              {/* History */}
              {prediction.history?.length > 0 && (
                <div style={styles.history}>
                  <p style={styles.probTitle}>⏱ Recent History</p>
                  <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                    {prediction.history.slice(-8).map((a, i) => (
                      <span key={i} style={styles.histBadge}>{a}</span>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ── Tiny sub-components ───────────────────────────────────────────────────────

function Row({ label, value, big }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
      <span style={{ color: "#94a3b8" }}>{label}</span>
      <span style={{ color: big ? "#f1f5f9" : "#a5b4fc", fontWeight: big ? 700 : 500, fontSize: big ? 18 : 15 }}>
        {value}
      </span>
    </div>
  );
}

function statusColor(s) {
  return s === "open" ? "#16a34a" : s === "connecting" ? "#ca8a04" : "#dc2626";
}

// ── Styles ────────────────────────────────────────────────────────────────────

const styles = {
  container: {
    maxWidth: 580,
    margin: "0 auto",
    padding: 24,
    fontFamily: "'Inter', sans-serif",
    color: "#f1f5f9",
    background: "#0f172a",
    minHeight: "100vh",
  },
  title: { fontSize: 22, fontWeight: 700, marginBottom: 12, textAlign: "center" },
  pill: {
    display: "inline-block", padding: "3px 10px", borderRadius: 999,
    fontSize: 12, marginBottom: 14, color: "#fff",
  },
  video: {
    width: "100%", borderRadius: 12,
    border: "2px solid #334155", background: "#1e293b",
  },
  controls: { display: "flex", justifyContent: "center", marginTop: 14, gap: 12 },
  btn: {
    padding: "10px 28px", borderRadius: 8, border: "none",
    background: "#6366f1", color: "#fff", fontWeight: 700,
    fontSize: 15, cursor: "pointer",
  },
  error: { color: "#f87171", marginTop: 8, textAlign: "center" },
  card: {
    marginTop: 20, background: "#1e293b", borderRadius: 14,
    padding: 20, border: "1px solid #334155",
  },
  barWrap: { display: "flex", alignItems: "center", gap: 10, marginBottom: 12 },
  barLabel: { color: "#94a3b8", whiteSpace: "nowrap" },
  barBg: { flex: 1, background: "#334155", borderRadius: 6, height: 12, overflow: "hidden" },
  barFill: { height: 12, borderRadius: 6, transition: "width 0.4s ease" },
  probs: { marginTop: 16, borderTop: "1px solid #334155", paddingTop: 12 },
  probTitle: { color: "#64748b", fontSize: 12, fontWeight: 600, marginBottom: 8, textTransform: "uppercase" },
  probRow: { display: "flex", alignItems: "center", gap: 8, marginBottom: 6 },
  history: { marginTop: 14, borderTop: "1px solid #334155", paddingTop: 12 },
  histBadge: {
    background: "#1e293b", border: "1px solid #475569",
    borderRadius: 6, padding: "2px 8px", fontSize: 12, color: "#cbd5e1",
  },
};
