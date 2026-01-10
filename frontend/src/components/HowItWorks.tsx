import { Upload, Brain, FileCheck, ArrowRight } from "lucide-react";

const steps = [
  {
    icon: Upload,
    title: "Upload Image",
    description: "Simply drag and drop your X-ray or medical scan image. We support DICOM, PNG, and JPEG formats.",
    step: "01",
  },
  {
    icon: Brain,
    title: "AI Analysis",
    description: "Our advanced deep learning model processes your image using state-of-the-art algorithms in seconds.",
    step: "02",
  },
  {
    icon: FileCheck,
    title: "Instant Results",
    description: "Receive detailed analysis with confidence scores, heatmaps, and actionable medical insights.",
    step: "03",
  },
];

const HowItWorks = () => {
  return (
    <section id="about" className="py-24 relative bg-muted/30">
      <div className="container mx-auto px-6">
        {/* Section Header */}
        <div className="text-center max-w-2xl mx-auto mb-16">
          <span className="inline-block px-4 py-1.5 rounded-full bg-secondary text-secondary-foreground text-sm font-medium mb-4">
            Simple Process
          </span>
          <h2 className="text-3xl sm:text-4xl font-bold text-foreground mb-4">
            How It <span className="text-primary font-semibold">Works</span>
          </h2>
          <p className="text-muted-foreground text-lg">
            Get accurate medical diagnosis in three simple steps. Our AI-powered platform makes healthcare accessible to everyone.
          </p>
        </div>

        {/* Steps Grid */}
        <div className="grid md:grid-cols-3 gap-8 relative">
          {/* Connecting line */}
          <div className="hidden md:block absolute top-24 left-1/6 right-1/6 h-0.5 bg-gradient-to-r from-primary/20 via-primary to-primary/20" />
          
          {steps.map((step, index) => (
            <div
              key={step.title}
              className="relative group"
              style={{ animationDelay: `${index * 0.15}s` }}
            >
              <div className="glass-card p-8 hover-lift h-full">
                {/* Step number */}
                <div className="absolute -top-4 -right-4 w-12 h-12 rounded-full bg-gradient-to-br from-primary to-accent flex items-center justify-center shadow-medical-md">
                  <span className="text-sm font-bold text-primary-foreground">{step.step}</span>
                </div>

                {/* Icon */}
                <div className="w-16 h-16 rounded-2xl bg-secondary flex items-center justify-center mb-6 group-hover:bg-primary/10 transition-colors duration-300">
                  <step.icon className="w-8 h-8 text-primary" />
                </div>

                {/* Content */}
                <h3 className="text-xl font-semibold text-foreground mb-3">{step.title}</h3>
                <p className="text-muted-foreground leading-relaxed">{step.description}</p>

                {/* Arrow indicator */}
                {index < steps.length - 1 && (
                  <div className="hidden md:flex absolute -right-4 top-1/2 -translate-y-1/2 z-10">
                    <div className="w-8 h-8 rounded-full bg-card border border-border flex items-center justify-center">
                      <ArrowRight className="w-4 h-4 text-primary" />
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default HowItWorks;
