import { Bone, Wind, Activity } from "lucide-react";

const diseases = [
  {
    icon: Bone,
    title: "Fracture Detection",
    description: "Accurately identify bone fractures, micro-fractures, and skeletal abnormalities from X-ray images with high precision.",
    accuracy: "97.8%",
    color: "from-blue-500 to-cyan-500",
  },
  {
    icon: Wind,
    title: "Pneumonia",
    description: "Detect signs of pneumonia and lung infections through chest X-ray analysis using advanced pattern recognition.",
    accuracy: "96.5%",
    color: "from-teal-500 to-green-500",
  },
  {
    icon: Activity,
    title: "Tuberculosis",
    description: "Early detection of tuberculosis indicators in chest radiographs, crucial for timely treatment and prevention.",
    accuracy: "98.2%",
    color: "from-primary to-accent",
  },
];

const DiseasesSection = () => {
  return (
    <section id="services" className="py-24 relative">
      <div className="container mx-auto px-6">
        {/* Section Header */}
        <div className="text-center max-w-2xl mx-auto mb-16">
          <span className="inline-block px-4 py-1.5 rounded-full bg-secondary text-secondary-foreground text-sm font-medium mb-4">
            Diagnostic Capabilities
          </span>
          <h2 className="text-3xl sm:text-4xl font-bold text-foreground mb-4">
            Diseases We <span className="text-primary font-semibold">Detect</span>
          </h2>
          <p className="text-muted-foreground text-lg">
            Our AI models are trained on millions of medical images to detect various conditions with clinical-grade accuracy.
          </p>
        </div>

        {/* Disease Cards */}
        <div className="grid md:grid-cols-3 gap-8">
          {diseases.map((disease, index) => (
            <div
              key={disease.title}
              className="group relative"
              style={{ animationDelay: `${index * 0.15}s` }}
            >
              <div className="glass-card p-8 h-full hover-lift overflow-hidden">
                {/* Subtle background gradient always visible at low opacity */}
                <div className={`absolute inset-0 bg-gradient-to-br ${disease.color} opacity-5 transition-opacity duration-500`} />
                
                {/* Icon container */}
                <div className={`relative w-16 h-16 rounded-2xl bg-gradient-to-br ${disease.color} flex items-center justify-center mb-6 shadow-medical-sm group-hover:shadow-medical-glow transition-all duration-300`}>
                  <disease.icon className="w-8 h-8 text-white" />
                </div>

                {/* Content */}
                <h3 className="text-xl font-semibold text-foreground mb-3">{disease.title}</h3>
                <p className="text-muted-foreground leading-relaxed mb-6">{disease.description}</p>

                {/* Accuracy badge removed as requested */}
              </div>
            </div>
          ))}
        </div>

        {/* Additional info */}
      </div>
    </section>
  );
};

export default DiseasesSection;
