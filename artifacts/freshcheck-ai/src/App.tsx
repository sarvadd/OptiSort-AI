import { useEffect, useMemo, useRef, useState } from "react";

declare global {
  interface Window {
    tmImage: any;
  }
}

const MODEL_URL = "https://teachablemachine.withgoogle.com/models/Nc5P3SYBJ/";
const HISTORY_KEY = "freshcheck_history_v1";
const MAX_HISTORY = 200;

type Prediction = {
  className: string;
  probability: number;
};

type Status =
  | "loading-libs"
  | "loading-model"
  | "ready"
  | "camera-on"
  | "image-loaded"
  | "error";

type ScanCategory = "fresh" | "ripe" | "rotten" | "other";

type ScanRecord = {
  id: string;
  label: string;
  confidence: number;
  category: ScanCategory;
  thumbnail: string;
  source: "camera" | "upload";
  timestamp: number;
};

const FRUIT_ICONS = ["🍎", "🍊", "🍌", "🍓", "🍇", "🍉", "🥝", "🍑", "🍍", "🥭"];

function categorize(label: string): ScanCategory {
  const lower = label.toLowerCase();
  if (lower.includes("rotten") || lower.includes("spoil") || lower.includes("bad"))
    return "rotten";
  if (lower.includes("ripe") || lower.includes("mature")) return "ripe";
  if (lower.includes("fresh") || lower.includes("good") || lower.includes("unripe"))
    return "fresh";
  return "other";
}

function getExplanation(label: string, confidence: number): string {
  if (confidence < 0.6) {
    return "The model isn't confident about this image. Try a clearer, well-lit photo of a single fruit.";
  }
  const cat = categorize(label);
  if (cat === "rotten")
    return "This fruit looks spoiled. Mark it as discard — it's not safe to sell or eat.";
  if (cat === "ripe")
    return "This fruit is perfectly ripe — move it to the front for quick sale.";
  if (cat === "fresh")
    return "This fruit looks fresh and ready to stock. A great healthy choice!";
  return "Detection complete. Check the result above.";
}

function categoryColor(cat: ScanCategory): string {
  if (cat === "rotten") return "#ef4444";
  if (cat === "ripe") return "#f97316";
  if (cat === "fresh") return "#16a34a";
  return "#6366f1";
}

function formatTime(ts: number): string {
  const d = new Date(ts);
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function dayKey(ts: number): string {
  const d = new Date(ts);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

function isSameDay(ts: number, ref: Date): boolean {
  const d = new Date(ts);
  return (
    d.getFullYear() === ref.getFullYear() &&
    d.getMonth() === ref.getMonth() &&
    d.getDate() === ref.getDate()
  );
}

function loadHistory(): ScanRecord[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    if (!raw) return [];
    const arr = JSON.parse(raw);
    if (!Array.isArray(arr)) return [];
    return arr;
  } catch {
    return [];
  }
}

function saveHistory(list: ScanRecord[]) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(list.slice(0, MAX_HISTORY)));
  } catch (e) {
    console.error("Could not save history", e);
  }
}

function makeThumbnail(source: HTMLCanvasElement | HTMLImageElement): string {
  const W = 160;
  const H = 160;
  const c = document.createElement("canvas");
  c.width = W;
  c.height = H;
  const ctx = c.getContext("2d");
  if (!ctx) return "";
  // cover-fit
  let sw = (source as any).width;
  let sh = (source as any).height;
  if (source instanceof HTMLImageElement) {
    sw = source.naturalWidth || source.width;
    sh = source.naturalHeight || source.height;
  }
  const scale = Math.max(W / sw, H / sh);
  const dw = sw * scale;
  const dh = sh * scale;
  const dx = (W - dw) / 2;
  const dy = (H - dh) / 2;
  ctx.drawImage(source, dx, dy, dw, dh);
  return c.toDataURL("image/jpeg", 0.7);
}

function FruitIconStrip() {
  return (
    <div className="flex flex-wrap items-center justify-center gap-2 sm:gap-3 text-2xl sm:text-3xl select-none">
      {FRUIT_ICONS.map((icon, i) => (
        <span
          key={i}
          className="inline-block transition-transform hover:scale-125"
          style={{ animation: `bob 3s ease-in-out ${i * 0.15}s infinite` }}
          aria-hidden
        >
          {icon}
        </span>
      ))}
    </div>
  );
}

export default function App() {
  const [status, setStatus] = useState<Status>("loading-libs");
  const [errorMsg, setErrorMsg] = useState<string>("");
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [cameraActive, setCameraActive] = useState(false);
  const [uploadPreview, setUploadPreview] = useState<string | null>(null);
  const [history, setHistory] = useState<ScanRecord[]>([]);
  const [savedFlash, setSavedFlash] = useState(false);
  const [filter, setFilter] = useState<"all" | ScanCategory>("all");

  const modelRef = useRef<any>(null);
  const webcamRef = useRef<any>(null);
  const webcamContainerRef = useRef<HTMLDivElement>(null);
  const uploadImgRef = useRef<HTMLImageElement>(null);
  const animationRef = useRef<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const lastUploadAutoSaved = useRef<string | null>(null);

  // Load history on mount
  useEffect(() => {
    setHistory(loadHistory());
  }, []);

  // Wait for TM library, then load the model
  useEffect(() => {
    let cancelled = false;
    const waitForLibs = async () => {
      const start = Date.now();
      while (!window.tmImage) {
        if (Date.now() - start > 15000)
          throw new Error("Timed out waiting for AI libraries to load.");
        await new Promise((r) => setTimeout(r, 100));
      }
    };
    (async () => {
      try {
        await waitForLibs();
        if (cancelled) return;
        setStatus("loading-model");
        const model = await window.tmImage.load(
          MODEL_URL + "model.json",
          MODEL_URL + "metadata.json",
        );
        if (cancelled) return;
        modelRef.current = model;
        setStatus("ready");
      } catch (e: any) {
        if (cancelled) return;
        console.error(e);
        setErrorMsg(
          e?.message ||
            "Failed to load the AI model. Check your connection and refresh.",
        );
        setStatus("error");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      if (webcamRef.current) {
        try {
          webcamRef.current.stop();
        } catch {}
      }
    };
  }, []);

  const startCamera = async () => {
    if (!modelRef.current) return;
    setErrorMsg("");
    setUploadPreview(null);
    setPredictions([]);
    try {
      const flip = true;
      const webcam = new window.tmImage.Webcam(360, 360, flip);
      await webcam.setup();
      await webcam.play();
      webcamRef.current = webcam;
      if (webcamContainerRef.current) {
        webcamContainerRef.current.innerHTML = "";
        webcamContainerRef.current.appendChild(webcam.canvas);
      }
      setCameraActive(true);
      setStatus("camera-on");
      const loop = async () => {
        if (!webcamRef.current) return;
        webcamRef.current.update();
        const preds = await modelRef.current.predict(webcamRef.current.canvas);
        setPredictions(preds);
        animationRef.current = requestAnimationFrame(loop);
      };
      loop();
    } catch (e: any) {
      console.error(e);
      setErrorMsg(
        "Could not access the camera. Please allow camera permissions or try uploading an image.",
      );
      setStatus("ready");
    }
  };

  const stopCamera = () => {
    if (animationRef.current) cancelAnimationFrame(animationRef.current);
    animationRef.current = null;
    if (webcamRef.current) {
      try {
        webcamRef.current.stop();
      } catch {}
      webcamRef.current = null;
    }
    if (webcamContainerRef.current) webcamContainerRef.current.innerHTML = "";
    setCameraActive(false);
    setStatus("ready");
  };

  const handleFile = (file: File) => {
    if (!modelRef.current) return;
    if (cameraActive) stopCamera();
    const reader = new FileReader();
    reader.onload = (ev) => {
      const dataUrl = ev.target?.result as string;
      lastUploadAutoSaved.current = null;
      setUploadPreview(dataUrl);
      setStatus("image-loaded");
      setPredictions([]);
      setTimeout(async () => {
        if (uploadImgRef.current) {
          try {
            const preds = await modelRef.current.predict(uploadImgRef.current);
            setPredictions(preds);
            // Auto-save uploads to history
            const sorted = [...preds].sort((a, b) => b.probability - a.probability);
            const top = sorted[0];
            if (top && lastUploadAutoSaved.current !== dataUrl) {
              lastUploadAutoSaved.current = dataUrl;
              const thumb = makeThumbnail(uploadImgRef.current);
              saveScan(top, thumb, "upload");
            }
          } catch (e: any) {
            console.error(e);
            setErrorMsg("Could not analyze the image. Try a different one.");
          }
        }
      }, 200);
    };
    reader.readAsDataURL(file);
  };

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    if (e.target) e.target.value = "";
  };

  const saveScan = (
    top: Prediction,
    thumbnail: string,
    source: "camera" | "upload",
  ) => {
    const record: ScanRecord = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      label: top.className,
      confidence: top.probability,
      category: categorize(top.className),
      thumbnail,
      source,
      timestamp: Date.now(),
    };
    setHistory((prev) => {
      const next = [record, ...prev].slice(0, MAX_HISTORY);
      saveHistory(next);
      return next;
    });
    setSavedFlash(true);
    setTimeout(() => setSavedFlash(false), 1500);
  };

  const captureFromCamera = () => {
    if (!webcamRef.current) return;
    const sorted = [...predictions].sort((a, b) => b.probability - a.probability);
    const top = sorted[0];
    if (!top) return;
    const thumb = makeThumbnail(webcamRef.current.canvas);
    saveScan(top, thumb, "camera");
  };

  const removeRecord = (id: string) => {
    setHistory((prev) => {
      const next = prev.filter((r) => r.id !== id);
      saveHistory(next);
      return next;
    });
  };

  const clearHistory = () => {
    if (!history.length) return;
    if (!window.confirm("Clear all scan history? This cannot be undone.")) return;
    setHistory([]);
    saveHistory([]);
  };

  const exportCSV = () => {
    if (!history.length) return;
    const header = ["timestamp", "date", "label", "category", "confidence_percent", "source"];
    const rows = history.map((r) => [
      new Date(r.timestamp).toISOString(),
      formatTime(r.timestamp),
      r.label,
      r.category,
      (r.confidence * 100).toFixed(1),
      r.source,
    ]);
    const csv = [header, ...rows]
      .map((row) =>
        row
          .map((v) => {
            const s = String(v).replace(/"/g, '""');
            return /[",\n]/.test(s) ? `"${s}"` : s;
          })
          .join(","),
      )
      .join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `freshcheck-history-${dayKey(Date.now())}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Stats
  const stats = useMemo(() => {
    const today = new Date();
    let fresh = 0,
      ripe = 0,
      rotten = 0,
      other = 0,
      todayCount = 0,
      confidenceSum = 0;
    const byDay: Record<string, number> = {};
    for (const r of history) {
      if (r.category === "fresh") fresh++;
      else if (r.category === "ripe") ripe++;
      else if (r.category === "rotten") rotten++;
      else other++;
      if (isSameDay(r.timestamp, today)) todayCount++;
      confidenceSum += r.confidence;
      const k = dayKey(r.timestamp);
      byDay[k] = (byDay[k] || 0) + 1;
    }
    const total = history.length;
    const avgConfidence = total ? (confidenceSum / total) * 100 : 0;
    const recentDays = Object.entries(byDay)
      .sort((a, b) => (a[0] < b[0] ? 1 : -1))
      .slice(0, 7);
    return { total, fresh, ripe, rotten, other, todayCount, avgConfidence, recentDays };
  }, [history]);

  const filteredHistory = useMemo(() => {
    if (filter === "all") return history;
    return history.filter((r) => r.category === filter);
  }, [history, filter]);

  const sorted = [...predictions].sort((a, b) => b.probability - a.probability);
  const top = sorted[0];
  const second = sorted[1];

  const isModelReady =
    status === "ready" || status === "camera-on" || status === "image-loaded";
  const showLoading = status === "loading-libs" || status === "loading-model";

  return (
    <>
      <style>{`
        @keyframes bob {
          0%, 100% { transform: translateY(0) rotate(-3deg); }
          50% { transform: translateY(-6px) rotate(3deg); }
        }
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulseRing {
          0% { box-shadow: 0 0 0 0 rgba(22, 163, 74, 0.45); }
          70% { box-shadow: 0 0 0 14px rgba(22, 163, 74, 0); }
          100% { box-shadow: 0 0 0 0 rgba(22, 163, 74, 0); }
        }
        @keyframes pop {
          0% { transform: scale(0.6); opacity: 0; }
          60% { transform: scale(1.05); opacity: 1; }
          100% { transform: scale(1); opacity: 1; }
        }
        .fade-up { animation: fadeUp 0.4s ease-out both; }
        .pop { animation: pop 0.35s ease-out both; }
      `}</style>

      <div className="min-h-screen flex flex-col items-center px-4 py-6 sm:py-10">
        {/* Header */}
        <header className="w-full max-w-4xl text-center mb-6 sm:mb-8 fade-up">
          <div className="inline-flex items-center gap-2 bg-white/70 backdrop-blur-md border border-white px-4 py-1.5 rounded-full text-xs sm:text-sm font-semibold text-emerald-700 shadow-sm mb-4">
            <span className="text-lg" aria-hidden>🌿</span>
            AI-powered fruit freshness check
          </div>
          <h1 className="text-4xl sm:text-6xl font-extrabold tracking-tight">
            <span className="bg-gradient-to-r from-emerald-600 via-orange-500 to-rose-500 bg-clip-text text-transparent">
              FreshCheck
            </span>{" "}
            <span className="text-gray-900">AI</span>
          </h1>
          <p className="mt-3 text-gray-600 text-sm sm:text-base max-w-xl mx-auto">
            Built for warehouses and grocers — scan, classify, and track your produce inventory in seconds.
          </p>
          <div className="mt-5">
            <FruitIconStrip />
          </div>
        </header>

        {/* Main scanner card */}
        <main className="w-full max-w-4xl bg-white/85 backdrop-blur-xl rounded-3xl shadow-xl border border-white p-5 sm:p-8 fade-up">
          {showLoading && (
            <div className="flex flex-col items-center justify-center py-10">
              <div
                className="w-14 h-14 rounded-full border-4 border-emerald-200 border-t-emerald-600 animate-spin"
                aria-label="Loading"
              />
              <p className="mt-4 text-gray-600 font-medium">
                {status === "loading-libs"
                  ? "Loading AI libraries..."
                  : "Loading freshness model..."}
              </p>
            </div>
          )}

          {status === "error" && (
            <div className="bg-rose-50 border border-rose-200 text-rose-800 rounded-2xl p-4 text-sm">
              <p className="font-semibold mb-1">Something went wrong</p>
              <p>{errorMsg}</p>
            </div>
          )}

          {isModelReady && (
            <>
              {/* Preview area */}
              <div className="relative w-full aspect-square sm:aspect-[4/3] max-w-md mx-auto rounded-2xl overflow-hidden bg-gradient-to-br from-emerald-50 via-orange-50 to-rose-50 border-2 border-dashed border-emerald-200 flex items-center justify-center">
                {!cameraActive && !uploadPreview && (
                  <div className="text-center px-6">
                    <div className="text-6xl mb-3" aria-hidden>📸</div>
                    <p className="text-gray-700 font-semibold">Ready to scan!</p>
                    <p className="text-gray-500 text-sm mt-1">
                      Start the camera or upload a fruit image
                    </p>
                  </div>
                )}
                <div
                  ref={webcamContainerRef}
                  className={`${cameraActive ? "block" : "hidden"} w-full h-full flex items-center justify-center [&>canvas]:w-full [&>canvas]:h-full [&>canvas]:object-cover`}
                />
                {uploadPreview && !cameraActive && (
                  <img
                    ref={uploadImgRef}
                    src={uploadPreview}
                    alt="Uploaded fruit"
                    crossOrigin="anonymous"
                    className="w-full h-full object-cover"
                  />
                )}
                {cameraActive && (
                  <div className="absolute top-3 left-3 flex items-center gap-1.5 bg-white/90 backdrop-blur px-2.5 py-1 rounded-full text-xs font-semibold text-emerald-700">
                    <span
                      className="w-2 h-2 rounded-full bg-emerald-500"
                      style={{ animation: "pulseRing 2s infinite" }}
                    />
                    LIVE
                  </div>
                )}
                {savedFlash && (
                  <div className="absolute bottom-3 right-3 bg-emerald-600 text-white text-xs font-semibold px-3 py-1.5 rounded-full shadow-lg pop">
                    Saved to history ✓
                  </div>
                )}
              </div>

              {/* Controls */}
              <div className="mt-5 flex flex-col sm:flex-row flex-wrap gap-3 justify-center">
                {!cameraActive ? (
                  <button
                    onClick={startCamera}
                    className="inline-flex items-center justify-center gap-2 px-5 py-3 rounded-xl bg-emerald-600 hover:bg-emerald-700 active:scale-[0.98] transition-all text-white font-semibold shadow-md"
                  >
                    <span aria-hidden>📷</span> Start Camera
                  </button>
                ) : (
                  <>
                    <button
                      onClick={captureFromCamera}
                      disabled={!top}
                      className="inline-flex items-center justify-center gap-2 px-5 py-3 rounded-xl bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed active:scale-[0.98] transition-all text-white font-semibold shadow-md"
                    >
                      <span aria-hidden>💾</span> Capture & Save
                    </button>
                    <button
                      onClick={stopCamera}
                      className="inline-flex items-center justify-center gap-2 px-5 py-3 rounded-xl bg-rose-500 hover:bg-rose-600 active:scale-[0.98] transition-all text-white font-semibold shadow-md"
                    >
                      <span aria-hidden>⏹</span> Stop Camera
                    </button>
                  </>
                )}

                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="inline-flex items-center justify-center gap-2 px-5 py-3 rounded-xl bg-white border-2 border-emerald-200 hover:border-emerald-400 hover:bg-emerald-50 active:scale-[0.98] transition-all text-emerald-700 font-semibold shadow-sm"
                >
                  <span aria-hidden>🖼️</span> Upload Image
                </button>

                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={onFileChange}
                  className="hidden"
                />
              </div>

              {/* Results */}
              {top && (
                <div className="mt-7 fade-up">
                  <div className="text-center mb-4">
                    <p className="text-xs uppercase tracking-widest text-gray-500 font-semibold">
                      Top Prediction
                    </p>
                    <p
                      className="text-3xl sm:text-4xl font-extrabold mt-1"
                      style={{ color: categoryColor(categorize(top.className)) }}
                    >
                      {top.probability < 0.6 ? "Not Sure 🤔" : top.className}
                    </p>
                    <p className="mt-1 text-gray-600 text-sm">
                      Confidence:{" "}
                      <span className="font-bold">
                        {(top.probability * 100).toFixed(1)}%
                      </span>
                    </p>
                  </div>

                  <div className="space-y-3 max-w-md mx-auto">
                    {[top, second].filter(Boolean).map((p, idx) => (
                      <div key={idx}>
                        <div className="flex justify-between text-sm font-medium text-gray-700 mb-1">
                          <span>{p.className}</span>
                          <span>{(p.probability * 100).toFixed(1)}%</span>
                        </div>
                        <div className="w-full h-3 bg-gray-100 rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all duration-300 ease-out"
                            style={{
                              width: `${(p.probability * 100).toFixed(1)}%`,
                              background: `linear-gradient(90deg, ${categoryColor(categorize(p.className))}, ${categoryColor(categorize(p.className))}cc)`,
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="mt-5 max-w-md mx-auto bg-emerald-50/70 border border-emerald-100 rounded-2xl p-4 text-sm text-gray-700 text-center">
                    {getExplanation(top.className, top.probability)}
                  </div>
                </div>
              )}

              {errorMsg && status !== "error" && (
                <div className="mt-4 bg-rose-50 border border-rose-200 text-rose-700 rounded-xl p-3 text-sm text-center">
                  {errorMsg}
                </div>
              )}
            </>
          )}
        </main>

        {/* Stats */}
        <section className="w-full max-w-4xl mt-6 sm:mt-8 grid grid-cols-2 sm:grid-cols-5 gap-3 fade-up">
          <StatCard label="Total Scans" value={stats.total} accent="#1f2937" />
          <StatCard label="Fresh" value={stats.fresh} accent="#16a34a" />
          <StatCard label="Ripe" value={stats.ripe} accent="#f97316" />
          <StatCard label="Discard" value={stats.rotten} accent="#ef4444" />
          <StatCard label="Today" value={stats.todayCount} accent="#6366f1" />
        </section>

        {stats.total > 0 && (
          <section className="w-full max-w-4xl mt-3 bg-white/80 backdrop-blur rounded-2xl border border-white p-4 sm:p-5 shadow-sm fade-up">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
              <div className="text-sm text-gray-700">
                <span className="font-semibold">Avg confidence:</span>{" "}
                <span className="font-bold text-emerald-700">
                  {stats.avgConfidence.toFixed(1)}%
                </span>
                <span className="mx-2 text-gray-300">•</span>
                <span className="font-semibold">Discard rate:</span>{" "}
                <span className="font-bold text-rose-600">
                  {stats.total ? ((stats.rotten / stats.total) * 100).toFixed(1) : "0.0"}%
                </span>
              </div>
              {stats.recentDays.length > 0 && (
                <div className="flex items-center gap-2 overflow-x-auto">
                  <span className="text-xs text-gray-500 font-semibold uppercase tracking-wider whitespace-nowrap">
                    Last days:
                  </span>
                  {stats.recentDays.map(([day, count]) => (
                    <span
                      key={day}
                      className="inline-flex items-center gap-1.5 bg-gray-100 text-gray-700 px-2.5 py-1 rounded-full text-xs whitespace-nowrap"
                    >
                      <span className="font-medium">{day.slice(5)}</span>
                      <span className="font-bold text-emerald-700">{count}</span>
                    </span>
                  ))}
                </div>
              )}
            </div>
          </section>
        )}

        {/* History */}
        <section className="w-full max-w-4xl mt-6 bg-white/85 backdrop-blur-xl rounded-3xl shadow-xl border border-white p-5 sm:p-8 fade-up">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-4">
            <div>
              <h2 className="text-xl sm:text-2xl font-bold text-gray-900">Scan History</h2>
              <p className="text-xs text-gray-500 mt-0.5">
                Stored locally on this device — use as your scan log and backup.
              </p>
            </div>
            <div className="flex gap-2">
              <button
                onClick={exportCSV}
                disabled={!history.length}
                className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-900 hover:bg-gray-800 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-semibold transition-colors"
              >
                <span aria-hidden>⬇</span> Export CSV
              </button>
              <button
                onClick={clearHistory}
                disabled={!history.length}
                className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-white border border-gray-300 hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed text-gray-700 text-sm font-semibold transition-colors"
              >
                Clear
              </button>
            </div>
          </div>

          {/* Filter chips */}
          {history.length > 0 && (
            <div className="flex flex-wrap gap-2 mb-4">
              {(["all", "fresh", "ripe", "rotten", "other"] as const).map((f) => {
                const count =
                  f === "all"
                    ? history.length
                    : history.filter((r) => r.category === f).length;
                const active = filter === f;
                const color =
                  f === "all" ? "#1f2937" : categoryColor(f as ScanCategory);
                return (
                  <button
                    key={f}
                    onClick={() => setFilter(f)}
                    className="px-3 py-1.5 rounded-full text-xs font-semibold border transition-all"
                    style={{
                      backgroundColor: active ? color : "white",
                      color: active ? "white" : color,
                      borderColor: color + "55",
                    }}
                  >
                    {f.charAt(0).toUpperCase() + f.slice(1)} ({count})
                  </button>
                );
              })}
            </div>
          )}

          {history.length === 0 ? (
            <div className="text-center py-10 text-gray-500">
              <div className="text-5xl mb-3" aria-hidden>📋</div>
              <p className="font-medium">No scans yet</p>
              <p className="text-sm mt-1">
                Upload an image or capture a snapshot to log your first scan.
              </p>
            </div>
          ) : filteredHistory.length === 0 ? (
            <div className="text-center py-8 text-gray-500 text-sm">
              No scans in this category.
            </div>
          ) : (
            <ul className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {filteredHistory.map((r) => (
                <li
                  key={r.id}
                  className="flex items-center gap-3 bg-white border border-gray-100 rounded-2xl p-3 hover:border-emerald-200 hover:shadow-sm transition-all"
                >
                  <img
                    src={r.thumbnail}
                    alt={r.label}
                    className="w-16 h-16 rounded-xl object-cover flex-shrink-0 border border-gray-100"
                  />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span
                        className="text-xs font-bold uppercase tracking-wider px-2 py-0.5 rounded-full text-white"
                        style={{ backgroundColor: categoryColor(r.category) }}
                      >
                        {r.category}
                      </span>
                      <span className="text-xs text-gray-500">
                        {r.source === "camera" ? "📷" : "🖼"} {formatTime(r.timestamp)}
                      </span>
                    </div>
                    <p
                      className="font-semibold text-gray-900 truncate mt-1"
                      title={r.label}
                    >
                      {r.label}
                    </p>
                    <p className="text-xs text-gray-500">
                      {(r.confidence * 100).toFixed(1)}% confidence
                    </p>
                  </div>
                  <button
                    onClick={() => removeRecord(r.id)}
                    className="opacity-50 hover:opacity-100 text-gray-500 hover:text-rose-600 text-lg p-1 transition-all flex-shrink-0"
                    aria-label="Delete scan"
                    title="Delete"
                  >
                    ×
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>

        <footer className="w-full max-w-4xl mt-6 sm:mt-8 text-center text-xs text-gray-500">
          <p>
            Built with TensorFlow.js + Teachable Machine. History is saved on this device only.
          </p>
        </footer>
      </div>
    </>
  );
}

function StatCard({
  label,
  value,
  accent,
}: {
  label: string;
  value: number;
  accent: string;
}) {
  return (
    <div className="bg-white/85 backdrop-blur rounded-2xl border border-white p-4 shadow-sm text-center">
      <p
        className="text-3xl sm:text-4xl font-extrabold tabular-nums"
        style={{ color: accent }}
      >
        {value}
      </p>
      <p className="text-xs uppercase tracking-wider font-semibold text-gray-500 mt-1">
        {label}
      </p>
    </div>
  );
}
