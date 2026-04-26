import { useEffect, useRef, useState } from "react";

declare global {
  interface Window {
    tmImage: any;
  }
}

const MODEL_URL = "https://teachablemachine.withgoogle.com/models/Nc5P3SYBJ/";

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

const FRUIT_ICONS = ["🍎", "🍊", "🍌", "🍓", "🍇", "🍉", "🥝", "🍑", "🍍", "🥭"];

function getExplanation(label: string, confidence: number): string {
  const lower = label.toLowerCase();
  if (confidence < 0.6) {
    return "The model isn't confident about this image. Try a clearer, well-lit photo of a single fruit.";
  }
  if (lower.includes("rotten") || lower.includes("spoil") || lower.includes("bad")) {
    return "This fruit looks spoiled. It's safer to throw it away rather than eat it.";
  }
  if (lower.includes("ripe") || lower.includes("mature")) {
    return "This fruit is perfectly ripe — enjoy it now for the best flavor!";
  }
  if (lower.includes("fresh") || lower.includes("good") || lower.includes("unripe")) {
    return "This fruit looks fresh and ready to eat. A great healthy choice!";
  }
  return "Detection complete. Check the result above.";
}

function labelColor(label: string): string {
  const lower = label.toLowerCase();
  if (lower.includes("rotten") || lower.includes("spoil") || lower.includes("bad"))
    return "#ef4444";
  if (lower.includes("ripe")) return "#f97316";
  if (lower.includes("fresh") || lower.includes("good")) return "#16a34a";
  return "#6366f1";
}

function FruitIconStrip() {
  return (
    <div className="flex flex-wrap items-center justify-center gap-2 sm:gap-3 text-2xl sm:text-3xl select-none">
      {FRUIT_ICONS.map((icon, i) => (
        <span
          key={i}
          className="inline-block transition-transform hover:scale-125"
          style={{
            animation: `bob 3s ease-in-out ${i * 0.15}s infinite`,
          }}
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

  const modelRef = useRef<any>(null);
  const webcamRef = useRef<any>(null);
  const webcamContainerRef = useRef<HTMLDivElement>(null);
  const uploadImgRef = useRef<HTMLImageElement>(null);
  const animationRef = useRef<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Wait for TM library, then load the model
  useEffect(() => {
    let cancelled = false;

    const waitForLibs = async () => {
      const start = Date.now();
      while (!window.tmImage) {
        if (Date.now() - start > 15000) {
          throw new Error("Timed out waiting for AI libraries to load.");
        }
        await new Promise((r) => setTimeout(r, 100));
      }
    };

    (async () => {
      try {
        await waitForLibs();
        if (cancelled) return;
        setStatus("loading-model");

        const modelURL = MODEL_URL + "model.json";
        const metadataURL = MODEL_URL + "metadata.json";
        const model = await window.tmImage.load(modelURL, metadataURL);

        if (cancelled) return;
        modelRef.current = model;
        setStatus("ready");
      } catch (e: any) {
        if (cancelled) return;
        console.error(e);
        setErrorMsg(
          e?.message || "Failed to load the AI model. Check your connection and refresh.",
        );
        setStatus("error");
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  // Cleanup on unmount
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

      // Mount the canvas
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
    if (webcamContainerRef.current) {
      webcamContainerRef.current.innerHTML = "";
    }
    setCameraActive(false);
    setStatus("ready");
  };

  const handleFile = (file: File) => {
    if (!modelRef.current) return;
    if (cameraActive) stopCamera();

    const reader = new FileReader();
    reader.onload = (ev) => {
      const dataUrl = ev.target?.result as string;
      setUploadPreview(dataUrl);
      setStatus("image-loaded");
      setPredictions([]);

      // Wait for image to render then predict
      setTimeout(async () => {
        if (uploadImgRef.current) {
          try {
            const preds = await modelRef.current.predict(uploadImgRef.current);
            setPredictions(preds);
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
  };

  // Sort predictions desc
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
        .fade-up { animation: fadeUp 0.4s ease-out both; }
      `}</style>

      <div className="min-h-screen flex flex-col items-center px-4 py-6 sm:py-10">
        {/* Header */}
        <header className="w-full max-w-3xl text-center mb-6 sm:mb-8 fade-up">
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
            Snap a photo or upload an image, and our AI tells you if your fruit is{" "}
            <span className="font-semibold text-emerald-600">fresh</span>,{" "}
            <span className="font-semibold text-orange-500">ripe</span>, or{" "}
            <span className="font-semibold text-rose-500">rotten</span> in seconds.
          </p>
          <div className="mt-5">
            <FruitIconStrip />
          </div>
        </header>

        {/* Main card */}
        <main className="w-full max-w-3xl bg-white/85 backdrop-blur-xl rounded-3xl shadow-xl border border-white p-5 sm:p-8 fade-up">
          {/* Status / loading / errors */}
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

                {/* Webcam mounts here */}
                <div
                  ref={webcamContainerRef}
                  className={`${cameraActive ? "block" : "hidden"} w-full h-full flex items-center justify-center [&>canvas]:w-full [&>canvas]:h-full [&>canvas]:object-cover`}
                />

                {/* Uploaded image preview */}
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
              </div>

              {/* Controls */}
              <div className="mt-5 flex flex-col sm:flex-row gap-3 justify-center">
                {!cameraActive ? (
                  <button
                    onClick={startCamera}
                    className="inline-flex items-center justify-center gap-2 px-5 py-3 rounded-xl bg-emerald-600 hover:bg-emerald-700 active:scale-[0.98] transition-all text-white font-semibold shadow-md"
                  >
                    <span aria-hidden>📷</span> Start Camera
                  </button>
                ) : (
                  <button
                    onClick={stopCamera}
                    className="inline-flex items-center justify-center gap-2 px-5 py-3 rounded-xl bg-rose-500 hover:bg-rose-600 active:scale-[0.98] transition-all text-white font-semibold shadow-md"
                  >
                    <span aria-hidden>⏹</span> Stop Camera
                  </button>
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
                      style={{ color: labelColor(top.className) }}
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

                  {/* Bars */}
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
                              background: `linear-gradient(90deg, ${labelColor(p.className)}, ${labelColor(p.className)}cc)`,
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Explanation */}
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

        {/* Footer */}
        <footer className="w-full max-w-3xl mt-6 sm:mt-8 text-center text-xs text-gray-500">
          <p>
            Built with TensorFlow.js + Teachable Machine. Best results with bright lighting and a single fruit in frame.
          </p>
        </footer>
      </div>
    </>
  );
}
