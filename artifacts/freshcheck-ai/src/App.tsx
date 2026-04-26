import { useEffect, useMemo, useRef, useState } from "react";

declare global {
  interface Window {
    tmImage: any;
    mobilenet: any;
  }
}

// Maps MobileNet (ImageNet) class names to friendly produce names.
// Longer keys are checked first so multi-word matches win.
const PRODUCE_KEYWORDS: Record<string, string> = {
  "granny smith": "apple",
  "custard apple": "custard apple",
  "head cabbage": "cabbage",
  "globe artichoke": "artichoke",
  "spaghetti squash": "squash",
  "acorn squash": "squash",
  "butternut squash": "squash",
  "bell pepper": "bell pepper",
  apple: "apple",
  banana: "banana",
  orange: "orange",
  lemon: "lemon",
  lime: "lime",
  strawberry: "strawberry",
  pineapple: "pineapple",
  ananas: "pineapple",
  fig: "fig",
  jackfruit: "jackfruit",
  pomegranate: "pomegranate",
  mango: "mango",
  papaya: "papaya",
  watermelon: "watermelon",
  cantaloupe: "cantaloupe",
  grape: "grape",
  cherry: "cherry",
  peach: "peach",
  pear: "pear",
  kiwi: "kiwi",
  blueberry: "blueberry",
  raspberry: "raspberry",
  avocado: "avocado",
  coconut: "coconut",
  tomato: "tomato",
  carrot: "carrot",
  eggplant: "eggplant",
  broccoli: "broccoli",
  cauliflower: "cauliflower",
  cabbage: "cabbage",
  potato: "potato",
  corn: "corn",
  cucumber: "cucumber",
  cuke: "cucumber",
  onion: "onion",
  garlic: "garlic",
  lettuce: "lettuce",
  mushroom: "mushroom",
  zucchini: "zucchini",
  courgette: "zucchini",
  squash: "squash",
  artichoke: "artichoke",
  asparagus: "asparagus",
};

const PRODUCE_KEYS_SORTED = Object.keys(PRODUCE_KEYWORDS).sort(
  (a, b) => b.length - a.length,
);

function matchProduceKeyword(className: string): string | null {
  const cls = className.toLowerCase();
  for (const k of PRODUCE_KEYS_SORTED) {
    if (cls.includes(k)) return PRODUCE_KEYWORDS[k];
  }
  return null;
}

async function detectProduceType(
  source: HTMLCanvasElement | HTMLImageElement,
  mn: any,
): Promise<string | null> {
  if (!mn) return null;
  try {
    const preds: Array<{ className: string; probability: number }> =
      await mn.classify(source, 5);
    for (const p of preds) {
      const m = matchProduceKeyword(p.className);
      if (m) return m;
    }
    return null;
  } catch (e) {
    console.error("MobileNet classify failed", e);
    return null;
  }
}

const MODEL_URL = "https://teachablemachine.withgoogle.com/models/Nc5P3SYBJ/";
const HISTORY_KEY = "freshcheck_history_v1";
const MAX_HISTORY = 200;
const CONFIDENCE_THRESHOLD = 0.6;

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

type ScanCategory = "fresh" | "overripe-discard";

type ScanRecord = {
  id: string;
  label: string;
  confidence: number;
  category: ScanCategory;
  /** MobileNet-detected fruit/vegetable name (e.g. "apple"). Optional for backward compat. */
  produceType?: string;
  thumbnail: string;
  source: "camera" | "upload";
  timestamp: number;
};

const FRUIT_ICONS = ["🍎", "🍊", "🍌", "🍓", "🍇", "🍉", "🥝", "🍑", "🍍", "🥭"];

function categorize(label: string): ScanCategory {
  const lower = label.toLowerCase();
  if (lower.includes("fresh") || lower.includes("good") || lower.includes("unripe"))
    return "fresh";
  return "overripe-discard";
}

function categoryLabel(cat: ScanCategory): string {
  return cat === "fresh" ? "Fresh" : "Overripe/Discard";
}

function getExplanation(cat: ScanCategory): string {
  if (cat === "fresh")
    return "This fruit looks fresh and ready to stock. A great healthy choice!";
  return "This fruit is past its prime. Mark it for discard or quick clearance.";
}

function categoryColor(cat: ScanCategory): string {
  return cat === "fresh" ? "#16a34a" : "#ef4444";
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

function lastNDayKeys(n: number): string[] {
  const arr: string[] = [];
  const today = new Date();
  for (let i = n - 1; i >= 0; i--) {
    const d = new Date(today);
    d.setDate(today.getDate() - i);
    arr.push(
      `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`,
    );
  }
  return arr;
}

const FRUIT_EMOJI_MAP: Record<string, string> = {
  apple: "🍎",
  banana: "🍌",
  orange: "🍊",
  strawberry: "🍓",
  grape: "🍇",
  watermelon: "🍉",
  kiwi: "🥝",
  peach: "🍑",
  pineapple: "🍍",
  mango: "🥭",
  lemon: "🍋",
  cherry: "🍒",
  pear: "🍐",
  coconut: "🥥",
  blueberry: "🫐",
  avocado: "🥑",
  tomato: "🍅",
  carrot: "🥕",
  eggplant: "🍆",
  broccoli: "🥦",
  potato: "🥔",
  corn: "🌽",
  pepper: "🫑",
  cucumber: "🥒",
  onion: "🧅",
  garlic: "🧄",
  lettuce: "🥬",
  mushroom: "🍄",
};

function produceEmoji(name: string): string {
  const lower = name.toLowerCase();
  for (const k of Object.keys(FRUIT_EMOJI_MAP)) {
    if (lower.includes(k)) return FRUIT_EMOJI_MAP[k];
  }
  return "🍽️";
}

function extractProduceName(label: string): string {
  // Strip freshness keywords to get the produce name (e.g. "Fresh Apple" -> "Apple")
  const cleaned = label
    .replace(
      /\b(fresh|rotten|spoiled|spoil|bad|good|ripe|unripe|mature|overripe|stale|old|new)\b/gi,
      "",
    )
    .replace(/\s+/g, " ")
    .trim();
  return (cleaned || label).toLowerCase();
}

type ProduceEntry = {
  name: string;
  total: number;
  days: Record<string, number>;
  // All scan timestamps for this produce (used for hourly drill-down)
  timestamps: number[];
  last: number;
};

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
    // Migrate older records (had "ripe" / "rotten" / "other" categories)
    return arr.map((r: any) => ({
      ...r,
      category:
        r.category === "fresh" ? "fresh" : "overripe-discard",
    }));
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

// Sample the image and decide if it's a blank/uniform background (white wall, black, etc.)
// Returns { blank, skinHeavy } so we can decide whether to skip prediction.
function analyzeFrame(
  source: HTMLCanvasElement | HTMLImageElement,
): { blank: boolean; skinHeavy: boolean } {
  const W = 48;
  const H = 48;
  const c = document.createElement("canvas");
  c.width = W;
  c.height = H;
  const ctx = c.getContext("2d");
  if (!ctx) return { blank: false, skinHeavy: false };
  try {
    ctx.drawImage(source, 0, 0, W, H);
  } catch {
    return { blank: false, skinHeavy: false };
  }
  let data: Uint8ClampedArray;
  try {
    data = ctx.getImageData(0, 0, W, H).data;
  } catch {
    return { blank: false, skinHeavy: false };
  }
  let sum = 0;
  let sumSq = 0;
  let n = 0;
  let skin = 0;
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const lum = 0.299 * r + 0.587 * g + 0.114 * b;
    sum += lum;
    sumSq += lum * lum;
    n++;
    // Loose skin-tone heuristic
    if (
      r > 95 &&
      g > 40 &&
      b > 20 &&
      r > g &&
      r > b &&
      r - Math.min(g, b) > 15 &&
      Math.abs(r - g) > 10
    ) {
      skin++;
    }
  }
  const mean = sum / n;
  const variance = Math.max(0, sumSq / n - mean * mean);
  const stddev = Math.sqrt(variance);
  // Uniform / blank: very low contrast across the frame.
  const blank = stddev < 14;
  // Skin-heavy: more than ~55% of pixels look like skin tones (likely just a person)
  const skinHeavy = skin / n > 0.55;
  return { blank, skinHeavy };
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
  const [nothingToScan, setNothingToScan] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(() => {
    try {
      return localStorage.getItem("optisort_sidebar_open") === "1";
    } catch {
      return false;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem("optisort_sidebar_open", sidebarOpen ? "1" : "0");
    } catch {}
  }, [sidebarOpen]);

  const modelRef = useRef<any>(null);
  const mobilenetRef = useRef<any>(null);
  const webcamRef = useRef<any>(null);
  const webcamContainerRef = useRef<HTMLDivElement>(null);
  const uploadImgRef = useRef<HTMLImageElement>(null);
  const animationRef = useRef<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const lastUploadAutoSaved = useRef<string | null>(null);

  useEffect(() => {
    setHistory(loadHistory());
  }, []);

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
        // Load MobileNet in background so we can identify what fruit/veg is in frame.
        // The freshness model only knows fresh/rotten, so we use MobileNet
        // (an ImageNet classifier) for the produce *type*.
        if (window.mobilenet) {
          window.mobilenet
            .load()
            .then((mn: any) => {
              if (!cancelled) mobilenetRef.current = mn;
            })
            .catch((err: any) => {
              console.error("MobileNet failed to load — produce names will be unknown.", err);
            });
        }
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
    setNothingToScan(false);
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
        const frame = analyzeFrame(webcamRef.current.canvas);
        const preds: Prediction[] = await modelRef.current.predict(
          webcamRef.current.canvas,
        );
        const sorted = [...preds].sort((a, b) => b.probability - a.probability);
        const top = sorted[0];
        // Decide if there's a fruit in frame
        const lowConfidence = !top || top.probability < CONFIDENCE_THRESHOLD;
        const empty = frame.blank || (frame.skinHeavy && lowConfidence) || lowConfidence;
        setNothingToScan(empty);
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
    setNothingToScan(false);
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
      setNothingToScan(false);
      setTimeout(async () => {
        if (uploadImgRef.current) {
          try {
            const frame = analyzeFrame(uploadImgRef.current);
            const preds: Prediction[] = await modelRef.current.predict(
              uploadImgRef.current,
            );
            const sorted = [...preds].sort((a, b) => b.probability - a.probability);
            const top = sorted[0];
            const lowConfidence = !top || top.probability < CONFIDENCE_THRESHOLD;
            const empty =
              frame.blank || (frame.skinHeavy && lowConfidence) || lowConfidence;
            setPredictions(preds);
            setNothingToScan(empty);
            // Auto-save uploads to history (only if a real fruit was detected)
            if (!empty && top && lastUploadAutoSaved.current !== dataUrl) {
              lastUploadAutoSaved.current = dataUrl;
              const thumb = makeThumbnail(uploadImgRef.current);
              const produceType = await detectProduceType(
                uploadImgRef.current,
                mobilenetRef.current,
              );
              saveScan(top, thumb, "upload", produceType);
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
    produceType: string | null,
  ) => {
    const record: ScanRecord = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      label: top.className,
      confidence: top.probability,
      category: categorize(top.className),
      produceType: produceType || undefined,
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

  const captureFromCamera = async () => {
    if (!webcamRef.current || nothingToScan) return;
    const sorted = [...predictions].sort((a, b) => b.probability - a.probability);
    const top = sorted[0];
    if (!top) return;
    const thumb = makeThumbnail(webcamRef.current.canvas);
    const produceType = await detectProduceType(
      webcamRef.current.canvas,
      mobilenetRef.current,
    );
    saveScan(top, thumb, "camera", produceType);
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
      categoryLabel(r.category),
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
    a.download = `optisort-history-${dayKey(Date.now())}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const stats = useMemo(() => {
    const today = new Date();
    let fresh = 0;
    let overripe = 0;
    let todayCount = 0;
    let confidenceSum = 0;
    const byDay: Record<string, number> = {};
    for (const r of history) {
      if (r.category === "fresh") fresh++;
      else overripe++;
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
    return { total, fresh, overripe, todayCount, avgConfidence, recentDays };
  }, [history]);

  const produceBreakdown = useMemo<ProduceEntry[]>(() => {
    const map: Record<string, ProduceEntry> = {};
    for (const r of history) {
      // Prefer the MobileNet-detected produce type; fall back to extracting
      // from the freshness label for older records, or "unknown" if neither.
      const fallback = extractProduceName(r.label);
      const isFreshnessOnly = /^(fresh|rotten|ripe|overripe|nothing|spoiled|good|bad)$/i.test(
        fallback,
      );
      const name =
        r.produceType || (isFreshnessOnly ? "unknown" : fallback) || "unknown";
      if (!map[name]) {
        map[name] = {
          name,
          total: 0,
          days: {},
          timestamps: [],
          last: 0,
        };
      }
      const e = map[name];
      e.total++;
      const dk = dayKey(r.timestamp);
      e.days[dk] = (e.days[dk] || 0) + 1;
      e.timestamps.push(r.timestamp);
      if (r.timestamp > e.last) e.last = r.timestamp;
    }
    return Object.values(map).sort((a, b) => b.total - a.total);
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
  const showResult = !!top && !nothingToScan;

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

      {/* Collapsible produce-counts sidebar */}
      <aside
        aria-hidden={!sidebarOpen}
        className={`fixed top-0 left-0 h-full z-40 w-80 max-w-[85vw] bg-white/95 backdrop-blur-xl shadow-2xl border-r border-gray-200 overflow-y-auto transform transition-transform duration-300 ${sidebarOpen ? "translate-x-0" : "-translate-x-full"}`}
      >
        <div className="sticky top-0 bg-white/95 backdrop-blur p-4 border-b border-gray-100 flex items-center justify-between z-10">
          <div>
            <h3 className="font-bold text-gray-900 text-lg">Produce Counts</h3>
            <p className="text-xs text-gray-500">Auto-tracked from your scans</p>
          </div>
          <button
            onClick={() => setSidebarOpen(false)}
            className="p-1.5 rounded-lg hover:bg-gray-100 text-gray-600"
            aria-label="Close sidebar"
          >
            ✕
          </button>
        </div>
        <div className="p-4 space-y-3">
          {produceBreakdown.length === 0 ? (
            <div className="text-center py-10 text-gray-500">
              <div className="text-4xl mb-2" aria-hidden>
                🥗
              </div>
              <p className="text-sm font-medium">No produce yet</p>
              <p className="text-xs mt-1">
                Scan a fruit or vegetable and it'll show up here automatically.
              </p>
            </div>
          ) : (
            produceBreakdown.map((p) => <ProduceCard key={p.name} produce={p} />)
          )}
        </div>
      </aside>

      {/* Mobile backdrop */}
      {sidebarOpen && (
        <button
          onClick={() => setSidebarOpen(false)}
          aria-label="Close sidebar"
          className="lg:hidden fixed inset-0 bg-black/40 z-30"
        />
      )}

      {/* Sidebar toggle */}
      <button
        onClick={() => setSidebarOpen((v) => !v)}
        className={`fixed top-4 z-50 inline-flex items-center gap-2 bg-white shadow-lg border border-gray-200 px-3 py-2 rounded-xl font-semibold text-sm text-gray-700 hover:bg-emerald-50 hover:border-emerald-300 transition-all ${sidebarOpen ? "left-[calc(20rem+1rem)] max-[680px]:left-[calc(85vw+0.5rem)]" : "left-4"}`}
        aria-label="Toggle produce counts sidebar"
        aria-expanded={sidebarOpen}
      >
        <span className="text-lg" aria-hidden>
          {sidebarOpen ? "✕" : "📊"}
        </span>
        <span className="hidden sm:inline">
          {sidebarOpen ? "Hide" : "Counts"}
        </span>
        {!sidebarOpen && produceBreakdown.length > 0 && (
          <span className="ml-1 inline-flex items-center justify-center min-w-[20px] h-5 px-1.5 rounded-full bg-emerald-600 text-white text-xs font-bold">
            {produceBreakdown.length}
          </span>
        )}
      </button>

      <div className="min-h-screen flex flex-col items-center px-4 py-6 sm:py-10">
        {/* Header */}
        <header className="w-full max-w-4xl text-center mb-6 sm:mb-8 fade-up">
          <div className="inline-flex items-center gap-2 bg-white/70 backdrop-blur-md border border-white px-4 py-1.5 rounded-full text-xs sm:text-sm font-semibold text-emerald-700 shadow-sm mb-4">
            <span className="text-lg" aria-hidden>🌿</span>
            AI-powered fruit freshness check
          </div>
          <h1 className="text-4xl sm:text-6xl font-extrabold tracking-tight">
            <span className="bg-gradient-to-r from-emerald-600 via-orange-500 to-rose-500 bg-clip-text text-transparent">
              OptiSort
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
                      disabled={!top || nothingToScan}
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

              {/* Nothing to scan */}
              {nothingToScan && (cameraActive || uploadPreview) && (
                <div className="mt-7 max-w-md mx-auto fade-up text-center">
                  <p className="text-xs uppercase tracking-widest text-gray-500 font-semibold">
                    Result
                  </p>
                  <p className="text-3xl sm:text-4xl font-extrabold mt-1 text-gray-500">
                    Nothing to scan 🫥
                  </p>
                  <div className="mt-4 bg-gray-50 border border-gray-200 rounded-2xl p-4 text-sm text-gray-600">
                    Point the camera at a fruit, or upload an image with a clear fruit in frame. Empty backgrounds and people-only photos are skipped.
                  </div>
                </div>
              )}

              {/* Results */}
              {showResult && (
                <div className="mt-7 fade-up">
                  <div className="text-center mb-4">
                    <p className="text-xs uppercase tracking-widest text-gray-500 font-semibold">
                      Top Prediction
                    </p>
                    <p
                      className="text-3xl sm:text-4xl font-extrabold mt-1"
                      style={{ color: categoryColor(categorize(top.className)) }}
                    >
                      {top.className}
                    </p>
                    <p className="mt-1 text-gray-600 text-sm">
                      Category:{" "}
                      <span
                        className="font-bold"
                        style={{ color: categoryColor(categorize(top.className)) }}
                      >
                        {categoryLabel(categorize(top.className))}
                      </span>{" "}
                      · Confidence:{" "}
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
                    {getExplanation(categorize(top.className))}
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
        <section className="w-full max-w-4xl mt-6 sm:mt-8 grid grid-cols-2 sm:grid-cols-4 gap-3 fade-up">
          <StatCard label="Total Scans" value={stats.total} accent="#1f2937" />
          <StatCard label="Fresh" value={stats.fresh} accent="#16a34a" />
          <StatCard label="Overripe / Discard" value={stats.overripe} accent="#ef4444" />
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
                  {stats.total ? ((stats.overripe / stats.total) * 100).toFixed(1) : "0.0"}%
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
              {(["all", "fresh", "overripe-discard"] as const).map((f) => {
                const count =
                  f === "all"
                    ? history.length
                    : history.filter((r) => r.category === f).length;
                const active = filter === f;
                const color =
                  f === "all" ? "#1f2937" : categoryColor(f as ScanCategory);
                const labelText =
                  f === "all"
                    ? "All"
                    : f === "fresh"
                      ? "Fresh"
                      : "Overripe/Discard";
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
                    {labelText} ({count})
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
                        {categoryLabel(r.category)}
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

        <footer className="w-full max-w-4xl mt-6 sm:mt-8 text-center text-xs text-gray-500 space-y-1">
          <p>
            Built with TensorFlow.js + Teachable Machine. History is saved on this device only.
          </p>
          <p className="text-gray-400">OptiSort is AI and can make mistakes.</p>
        </footer>
      </div>
    </>
  );
}

function HourlyLineChart({
  hourly,
  dayLabel,
}: {
  hourly: number[];
  dayLabel: string;
}) {
  const W = 280;
  const H = 110;
  const padL = 24;
  const padR = 8;
  const padT = 10;
  const padB = 22;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;
  const max = Math.max(1, ...hourly);
  const points = hourly.map((c, i) => {
    const x = padL + (i / 23) * innerW;
    const y = padT + innerH - (c / max) * innerH;
    return [x, y] as const;
  });
  const path = points.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x},${y}`).join(" ");
  const area = `${path} L${padL + innerW},${padT + innerH} L${padL},${padT + innerH} Z`;
  // y-axis ticks (0, mid, max)
  const yTicks = [0, Math.ceil(max / 2), max];
  return (
    <div>
      <p className="text-[10px] uppercase tracking-wider font-bold text-gray-500 mb-1">
        {dayLabel} · scans by hour
      </p>
      <div className="bg-white rounded-lg border border-gray-200 p-2">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto">
          {/* grid lines */}
          {yTicks.map((t) => {
            const y = padT + innerH - (t / max) * innerH;
            return (
              <g key={t}>
                <line
                  x1={padL}
                  y1={y}
                  x2={W - padR}
                  y2={y}
                  stroke="#e5e7eb"
                  strokeDasharray="2 3"
                />
                <text
                  x={padL - 4}
                  y={y + 3}
                  textAnchor="end"
                  fontSize="9"
                  fill="#9ca3af"
                >
                  {t}
                </text>
              </g>
            );
          })}
          {/* x-axis labels (every 4 hours) */}
          {[0, 4, 8, 12, 16, 20].map((h) => {
            const x = padL + (h / 23) * innerW;
            return (
              <text
                key={h}
                x={x}
                y={H - 6}
                textAnchor="middle"
                fontSize="9"
                fill="#9ca3af"
              >
                {h.toString().padStart(2, "0")}
              </text>
            );
          })}
          {/* area + line */}
          <path d={area} fill="rgba(16, 185, 129, 0.15)" />
          <path
            d={path}
            fill="none"
            stroke="#16a34a"
            strokeWidth="2"
            strokeLinejoin="round"
            strokeLinecap="round"
          />
          {/* points */}
          {points.map(([x, y], i) =>
            hourly[i] > 0 ? (
              <circle key={i} cx={x} cy={y} r="2.5" fill="#16a34a">
                <title>{`${i.toString().padStart(2, "0")}:00 — ${hourly[i]} scan${hourly[i] === 1 ? "" : "s"}`}</title>
              </circle>
            ) : null,
          )}
        </svg>
      </div>
    </div>
  );
}

function ProduceCard({ produce }: { produce: ProduceEntry }) {
  const [expanded, setExpanded] = useState(false);
  const [selectedDay, setSelectedDay] = useState<string | null>(null);

  const last7 = lastNDayKeys(7).map((d) => ({
    day: d,
    count: produce.days[d] || 0,
  }));
  const maxDay = Math.max(1, ...last7.map((d) => d.count));

  const hourly = useMemo<number[]>(() => {
    if (!selectedDay) return [];
    const buckets = new Array(24).fill(0);
    for (const ts of produce.timestamps) {
      if (dayKey(ts) === selectedDay) {
        buckets[new Date(ts).getHours()]++;
      }
    }
    return buckets;
  }, [selectedDay, produce.timestamps]);

  const dayLabel = selectedDay
    ? new Date(selectedDay + "T00:00:00").toLocaleDateString(undefined, {
        weekday: "short",
        month: "short",
        day: "numeric",
      })
    : "";

  return (
    <div className="bg-white border border-gray-200 rounded-xl overflow-hidden shadow-sm">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="w-full p-3 flex items-center gap-3 hover:bg-gray-50 transition-colors text-left"
        aria-expanded={expanded}
      >
        <span className="text-3xl flex-shrink-0" aria-hidden>
          {produceEmoji(produce.name)}
        </span>
        <div className="flex-1 min-w-0">
          <p className="font-semibold text-gray-900 truncate capitalize">
            {produce.name}
          </p>
          <p className="text-xs text-gray-500 mt-0.5">
            {produce.total} scan{produce.total === 1 ? "" : "s"} ·{" "}
            {Object.keys(produce.days).length} day
            {Object.keys(produce.days).length === 1 ? "" : "s"}
          </p>
        </div>
        <div className="text-right flex-shrink-0">
          <p className="text-2xl font-extrabold text-gray-900 tabular-nums leading-none">
            {produce.total}
          </p>
          <p className="text-xs text-gray-400 mt-1">{expanded ? "▲" : "▼"}</p>
        </div>
      </button>
      {expanded && (
        <div className="p-3 border-t border-gray-100 bg-gray-50/60 space-y-3">
          <div>
            <div className="flex items-center justify-between mb-2">
              <p className="text-[10px] uppercase tracking-wider font-bold text-gray-500">
                Last 7 days
              </p>
              <p className="text-[10px] text-gray-400 italic">tap a bar</p>
            </div>
            <div className="flex items-end gap-1 h-24">
              {last7.map((d) => {
                const isSelected = selectedDay === d.day;
                const isToday = d.day === dayKey(Date.now());
                return (
                  <button
                    key={d.day}
                    onClick={() =>
                      setSelectedDay((cur) =>
                        cur === d.day ? null : d.count > 0 ? d.day : cur,
                      )
                    }
                    disabled={d.count === 0}
                    className={`flex-1 flex flex-col items-center gap-1 min-w-0 rounded transition-all ${
                      d.count > 0
                        ? "cursor-pointer hover:opacity-80"
                        : "cursor-not-allowed opacity-60"
                    }`}
                    title={`${d.day}: ${d.count} scan${d.count === 1 ? "" : "s"}`}
                  >
                    <span className="text-[10px] font-bold text-gray-700 tabular-nums h-3">
                      {d.count || ""}
                    </span>
                    <div className="w-full flex items-end h-14 bg-white rounded-sm border border-gray-200">
                      <div
                        className={`w-full rounded-sm transition-all ${
                          isSelected
                            ? "bg-gradient-to-t from-emerald-700 to-emerald-500 ring-2 ring-emerald-300"
                            : "bg-gradient-to-t from-emerald-500 to-emerald-400"
                        }`}
                        style={{
                          height: `${Math.max(d.count > 0 ? 8 : 0, (d.count / maxDay) * 100)}%`,
                        }}
                      />
                    </div>
                    <span
                      className={`text-[10px] tabular-nums ${isToday ? "font-bold text-emerald-700" : "text-gray-500"}`}
                    >
                      {d.day.slice(8)}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {selectedDay && (
            <div className="fade-up">
              <HourlyLineChart hourly={hourly} dayLabel={dayLabel} />
            </div>
          )}

          <p className="text-[10px] text-gray-500 text-center pt-1">
            Last scan: {formatTime(produce.last)}
          </p>
        </div>
      )}
    </div>
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
