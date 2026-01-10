import { Button } from "@/components/ui/button";
import { Upload, ArrowRight, ShieldCheck, Zap, Users } from "lucide-react";
import { useNavigate } from "react-router-dom";
import heroImage from "@/assets/hero-medical.jpg";

const HeroSection = () => {
  const navigate = useNavigate();
  return (
    <section id="home" className="relative min-h-screen flex items-center pt-24 pb-16 hero-gradient overflow-hidden">
      <div className="container mx-auto px-6">
        <div className="grid lg:grid-cols-2 gap-12 lg:gap-16 items-center">
          {/* Left Content */}
          <div className="space-y-8 animate-fade-up">
            {/* Badge */}
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-secondary border border-primary/10">
              <span className="w-2 h-2 rounded-full bg-accent animate-pulse" />
              <span className="text-sm font-medium text-secondary-foreground">
                AI-Powered Medical Diagnosis
              </span>
            </div>

            {/* Headline */}
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold leading-tight text-foreground">
              Comprehensive{" "}
              <span className="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">Health Care</span>
              <br />& AI-Powered{" "}
              <span className="relative inline-block">
                Diagnosis
                <svg
                  className="absolute -bottom-2 left-0 w-full"
                  height="8"
                  viewBox="0 0 100 8"
                  preserveAspectRatio="none"
                >
                  <path
                    d="M0 7 Q25 0 50 7 Q75 0 100 7"
                    stroke="hsl(180 60% 45%)"
                    strokeWidth="3"
                    fill="none"
                    strokeLinecap="round"
                  />
                </svg>
              </span>
            </h1>

            {/* Subheading */}
            <p className="text-lg sm:text-xl text-muted-foreground max-w-xl leading-relaxed">
              Revolutionizing early disease detection using advanced AI and medical imaging. 
              Get accurate analysis of X-rays and medical scans in seconds, powered by 
              state-of-the-art deep learning technology.
            </p>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-start">
              <Button
                variant="hero"
                size="xl"
                className="group"
                onClick={() => navigate("/dashboard")}
              >
                <Upload className="w-5 h-5" />
                Upload X-ray
                <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
              </Button>
            </div>

            {/* Trust Indicators */}
            
          </div>

          {/* Right Content - Hero Image */}
          <div className="relative animate-fade-up stagger-2">
            <div className="relative rounded-3xl overflow-hidden shadow-medical-lg">
              {/* Glow effect behind image */}
              <div className="absolute inset-0 bg-gradient-to-br from-primary/20 to-accent/20 blur-3xl scale-110" />
              
              {/* Main image container */}
              <div className="relative rounded-3xl overflow-hidden border-2 border-primary/10">
                <img
                  src={heroImage}
                  alt="Medical professionals using AI diagnostic tools"
                  className="w-full h-auto object-cover"
                />
                
                {/* Overlay gradient */}
                <div className="absolute inset-0 bg-gradient-to-t from-background/20 to-transparent" />
              </div>
            </div>

            {/* Floating stats card - Detection Certainty */}
            <div className="absolute -left-6 top-1/4 glass-card p-4 animate-float" style={{ animationDelay: "1s" }}>
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary to-accent flex items-center justify-center">
                  <ShieldCheck className="w-6 h-6 text-primary-foreground" />
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Detection</p>
                  <p className="text-sm font-semibold text-foreground">Certainty</p>
                </div>
              </div>
            </div>

            <div className="absolute -right-4 bottom-1/4 glass-card p-4 animate-float" style={{ animationDelay: "2s" }}>
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-accent to-accent/80 flex items-center justify-center">
                  <Zap className="w-6 h-6 text-accent-foreground" />
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Processing</p>
                  <p className="text-sm font-semibold text-foreground">Real-time</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
