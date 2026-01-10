import { useState } from "react";
import { Upload, Image, X, FileImage, CheckCircle2, AlertCircle, Activity } from "lucide-react";
import { Button } from "@/components/ui/button";
import ScanDoodles from "@/components/ScanDoodles";

interface AnalysisResult {
  prediction: string;
  confidence: number;
  explanation?: string;
  gradcam_image?: string;
  all_probabilities?: Record<string, number>;
  body_part_detection?: {
    body_part: string;
    confidence: number;
  };
  fracture_detection?: {
    fracture_detected: boolean;
    label: string;
    confidence: number;
    probability?: number;
  };
  stage_a_score?: number;
  stage_b_classification?: string;
}

const UploadSection = () => {
  const [isDragging, setIsDragging] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadedImageUrl, setUploadedImageUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      setUploadedFile(file);
      setUploadedImageUrl(URL.createObjectURL(file));
      setAnalysisResult(null);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setUploadedFile(file);
      setUploadedImageUrl(URL.createObjectURL(file));
      setAnalysisResult(null);
    }
  };

  const removeFile = () => {
    setUploadedFile(null);
    setUploadedImageUrl(null);
    setAnalysisResult(null);
  };

  const handleAnalyze = async () => {
    if (!uploadedFile || isAnalyzing) return;

    const baseUrl = (import.meta as any).env?.VITE_API_BASE_URL || "http://127.0.0.1:8000";
    const url = `${baseUrl.replace(/\/$/, "")}/api/analyze`;

    const form = new FormData();
    form.append("file", uploadedFile);
    form.append("mode", "unified");

    try {
      setIsAnalyzing(true);
      const res = await fetch(url, {
        method: "POST",
        body: form,
      });

      const data = await res.json().catch(() => null);
      if (!res.ok) {
        const msg = data?.detail || data?.error || "Request failed";
        throw new Error(msg);
      }

      setAnalysisResult(data);
    } catch (e: any) {
      alert(e?.message || "Analysis failed");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <section id="diagnosis" className="py-24 relative bg-muted/30">
      <div className="container mx-auto px-6">
        {/* Section Header */}
        <div className="text-center max-w-2xl mx-auto mb-16">
          <span className="inline-block px-4 py-1.5 rounded-full bg-secondary text-secondary-foreground text-sm font-medium mb-4">
            Try It Now
          </span>
          <h2 className="text-3xl sm:text-4xl font-bold text-foreground mb-4">
            Upload & <span className="text-primary font-semibold">Scan</span>
          </h2>
          <p className="text-muted-foreground text-lg">
            Upload your medical image and let our AI analyze it for potential conditions.
          </p>
        </div>

        {/* Upload Card */}
        {!analysisResult && (
          <div className="max-w-2xl mx-auto">
            <div
              className={`relative rounded-3xl border-2 border-dashed transition-all duration-300 ${isDragging
                ? "border-primary bg-primary/5 scale-[1.02]"
                : uploadedFile
                  ? "border-accent bg-accent/5"
                  : "border-border hover:border-primary/50 hover:bg-muted/50"
                }`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <div className="absolute inset-0 rounded-3xl overflow-hidden">
                <ScanDoodles />
              </div>

              {/* Animated border */}
              {isDragging && (
                <div className="absolute inset-0 rounded-3xl overflow-hidden">
                  <div className="absolute inset-0 bg-gradient-to-r from-primary via-accent to-primary bg-[length:200%_100%] animate-shimmer opacity-20" />
                </div>
              )}

              <div className="relative z-10 p-12 text-center">
                {!uploadedFile ? (
                  <>
                    {/* Upload icon */}
                    <div className="w-20 h-20 mx-auto mb-6 rounded-2xl bg-primary/10 flex items-center justify-center">
                      <Upload className={`w-10 h-10 text-primary transition-transform ${isDragging ? "scale-110" : ""}`} />
                    </div>

                    <h3 className="text-xl font-semibold text-foreground mb-2">
                      {isDragging ? "Drop your image here" : "Drag & drop your image"}
                    </h3>
                    <p className="text-muted-foreground mb-6">
                      Supports DICOM, PNG, JPEG formats (Max 10MB)
                    </p>

                    {/* File input */}
                    <label htmlFor="file-upload">
                      <Button variant="hero" size="lg" className="cursor-pointer" asChild>
                        <span>
                          <Image className="w-5 h-5" />
                          Browse Files
                        </span>
                      </Button>
                    </label>
                    <input
                      id="file-upload"
                      type="file"
                      accept="image/*"
                      onChange={handleFileSelect}
                      className="hidden"
                    />
                  </>
                ) : (
                  <>
                    {/* Uploaded file preview */}
                    <div className="flex items-center justify-center gap-4 mb-6">
                      <div className="w-16 h-16 rounded-2xl bg-accent/10 flex items-center justify-center">
                        <FileImage className="w-8 h-8 text-accent" />
                      </div>
                      <div className="text-left">
                        <p className="font-semibold text-foreground">{uploadedFile.name}</p>
                        <p className="text-sm text-muted-foreground">
                          {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB
                        </p>
                      </div>
                      <button
                        onClick={removeFile}
                        className="w-10 h-10 rounded-full bg-destructive/10 flex items-center justify-center hover:bg-destructive/20 transition-colors"
                      >
                        <X className="w-5 h-5 text-destructive" />
                      </button>
                    </div>

                    <div className="flex items-center justify-center gap-2 text-accent mb-6">
                      <CheckCircle2 className="w-5 h-5" />
                      <span className="font-medium">File ready for analysis</span>
                    </div>

                    <Button variant="medical" size="lg" onClick={handleAnalyze} disabled={isAnalyzing}>
                      {isAnalyzing ? (
                        <>
                          <Activity className="w-5 h-5 animate-pulse" />
                          Analyzing...
                        </>
                      ) : (
                        "Start AI Analysis"
                      )}
                    </Button>
                  </>
                )}
              </div>
            </div>

            {/* Supported formats */}
            <div className="mt-6 flex items-center justify-center gap-6 text-sm text-muted-foreground">
              <span>Supported:</span>
              <div className="flex gap-2">
                {["DICOM", "PNG", "JPEG", "JPG"].map((format) => (
                  <span
                    key={format}
                    className="px-2 py-1 rounded bg-secondary text-secondary-foreground text-xs font-medium"
                  >
                    {format}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Analysis Results */}
        {analysisResult && uploadedImageUrl && (
          <div className="max-w-6xl mx-auto space-y-8 animate-in fade-in duration-500">
            {/* Side-by-side Images */}
            <div className="grid md:grid-cols-2 gap-6">
              {/* Original X-ray */}
              <div className="glass-card p-6 space-y-4">
                <h3 className="text-xl font-semibold text-foreground text-center">
                  Original X-ray
                </h3>
                <div className="aspect-square rounded-2xl overflow-hidden bg-muted/50 flex items-center justify-center">
                  <img
                    src={uploadedImageUrl}
                    alt="Original X-ray"
                    className="w-full h-full object-contain"
                  />
                </div>
              </div>

              {/* Grad-CAM Heatmap */}
              <div className="glass-card p-6 space-y-4">
                <h3 className="text-xl font-semibold text-foreground text-center">
                  Explainability (Grad-CAM)
                </h3>
                <div className="aspect-square rounded-2xl overflow-hidden bg-muted/50 flex items-center justify-center">
                  {analysisResult.gradcam_image ? (
                    <img
                      src={analysisResult.gradcam_image}
                      alt="Grad-CAM Heatmap"
                      className="w-full h-full object-contain"
                    />
                  ) : (
                    <div className="text-center text-muted-foreground p-8">
                      <AlertCircle className="w-12 h-12 mx-auto mb-2 opacity-50" />
                      <p>Heatmap not available</p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Prediction Results */}
            <div className="glass-card p-8 space-y-6">
              <h3 className="text-2xl font-bold text-foreground text-center mb-6">
                AI Analysis Results
              </h3>

              {/* Body Part Detection - Simple */}
              {analysisResult.body_part_detection && (
                <div className="p-6 rounded-xl bg-secondary/30 border border-border">
                  <p className="text-sm text-muted-foreground mb-2">Body Part Detected</p>
                  <p className="text-2xl font-bold text-foreground">
                    {analysisResult.body_part_detection.body_part}
                  </p>
                </div>
              )}

              {/* Main Prediction - Simple, No Percentages */}
              <div className="p-6 rounded-xl bg-gradient-to-br from-primary/10 to-accent/10 border border-primary/20">
                <p className="text-sm text-muted-foreground mb-2">Diagnosis</p>
                <p className="text-2xl font-bold text-foreground">
                  {analysisResult.prediction}
                </p>
                {analysisResult.explanation && (
                  <p className="text-sm text-muted-foreground mt-3">
                    {analysisResult.explanation}
                  </p>
                )}
              </div>

              {/* Disclaimer */}
              <div className="mt-6 p-4 rounded-xl bg-primary/5 border border-primary/20">
                <p className="text-xs text-muted-foreground flex items-start gap-2">
                  <AlertCircle className="w-4 h-4 text-primary flex-shrink-0 mt-0.5" />
                  <span>
                    This AI analysis is for informational purposes only and should not replace professional medical advice.
                    Always consult a qualified healthcare professional for medical diagnosis and treatment.
                  </span>
                </p>
              </div>

              {/* Analyze Another Button */}
              <div className="text-center pt-4">
                <Button variant="outline" size="lg" onClick={removeFile}>
                  Analyze Another Image
                </Button>
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
};

export default UploadSection;
