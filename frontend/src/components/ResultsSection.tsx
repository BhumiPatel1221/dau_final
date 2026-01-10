import { AlertCircle, CheckCircle2, Info } from "lucide-react";

const mockResults = [
  { label: "Pneumonia", probability: 85, status: "high" },
  { label: "Tuberculosis", probability: 12, status: "low" },
  { label: "Normal", probability: 3, status: "low" },
];

const ResultsSection = () => {
  return (
    <section className="py-24 relative">
      <div className="container mx-auto px-6">
        {/* Section Header */}
        <div className="text-center max-w-2xl mx-auto mb-16">
          <span className="inline-block px-4 py-1.5 rounded-full bg-secondary text-secondary-foreground text-sm font-medium mb-4">
            Sample Output
          </span>
          <h2 className="text-3xl sm:text-4xl font-bold text-foreground mb-4">
            AI Analysis <span className="text-gradient opacity-80">Results</span>
          </h2>
          <p className="text-muted-foreground text-lg">
            See how our AI presents diagnostic results with confidence scores and visual explanations.
          </p>
        </div>

        <div className="max-w-4xl mx-auto grid md:grid-cols-2 gap-8">
          {/* Left - Heatmap Placeholder */}
          <div className="glass-card p-6">
            <h3 className="font-semibold text-foreground mb-4 flex items-center gap-2">
              <Info className="w-5 h-5 text-primary" />
              Grad-CAM Heatmap
            </h3>
            <div className="aspect-square rounded-2xl bg-gradient-to-br from-muted to-muted/50 relative overflow-hidden">
              {/* Mock X-ray with heatmap overlay */}
              <div className="absolute inset-0 flex items-center justify-center">
                <svg viewBox="0 0 200 200" className="w-full h-full p-8 opacity-20">
                  <rect x="70" y="40" width="60" height="120" rx="30" fill="currentColor" />
                  <circle cx="80" cy="70" r="20" fill="currentColor" />
                  <circle cx="120" cy="70" r="20" fill="currentColor" />
                </svg>
              </div>
              
              {/* Heatmap overlay */}
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="w-32 h-32 rounded-full bg-gradient-to-br from-red-500/50 via-yellow-500/30 to-transparent blur-xl" />
              </div>
              
              {/* Indicator */}
              <div className="absolute top-4 right-4 px-3 py-1.5 rounded-full bg-destructive/20 text-destructive text-xs font-medium flex items-center gap-1">
                <AlertCircle className="w-3 h-3" />
                Affected Area
              </div>
              
              {/* Caption */}
              <div className="absolute bottom-4 left-4 right-4">
                <p className="text-xs text-muted-foreground text-center bg-background/80 backdrop-blur-sm rounded-lg px-3 py-2">
                  Highlighted region shows where the AI detected anomalies
                </p>
              </div>
            </div>
          </div>

          {/* Right - Results Panel */}
          <div className="glass-card p-6">
            <h3 className="font-semibold text-foreground mb-6 flex items-center gap-2">
              <CheckCircle2 className="w-5 h-5 text-accent" />
              Diagnosis Probability
            </h3>

            <div className="space-y-6">
              {mockResults.map((result, index) => (
                <div key={result.label} className="space-y-2" style={{ animationDelay: `${index * 0.15}s` }}>
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-foreground">{result.label}</span>
                    <span
                      className={`text-sm font-semibold ${
                        result.probability > 50
                          ? "text-destructive"
                          : result.probability > 20
                          ? "text-yellow-600"
                          : "text-accent"
                      }`}
                    >
                      {result.probability}%
                    </span>
                  </div>
                  <div className="h-3 rounded-full bg-muted overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-1000 ${
                        result.probability > 50
                          ? "bg-gradient-to-r from-red-400 to-red-500"
                          : result.probability > 20
                          ? "bg-gradient-to-r from-yellow-400 to-yellow-500"
                          : "bg-gradient-to-r from-accent to-accent/80"
                      }`}
                      style={{ width: `${result.probability}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>

            {/* Confidence Score */}
            <div className="mt-8 p-4 rounded-2xl bg-secondary/50 border border-border">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-muted-foreground">Model Confidence</span>
                <span className="text-2xl font-bold text-foreground">94.5%</span>
              </div>
              <p className="text-xs text-muted-foreground">
                High confidence indicates reliable prediction based on image quality and model certainty.
              </p>
            </div>

            {/* Disclaimer */}
            <div className="mt-6 p-4 rounded-xl bg-primary/5 border border-primary/20">
              <p className="text-xs text-muted-foreground flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-primary flex-shrink-0 mt-0.5" />
                <span>
                  This is a demo interface. Results shown are for illustration purposes only. 
                  Always consult a qualified healthcare professional for medical diagnosis.
                </span>
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ResultsSection;
