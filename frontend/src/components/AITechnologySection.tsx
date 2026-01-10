import { Cpu, Layers, Eye, GitBranch } from "lucide-react";

const technologies = [
  {
    icon: Cpu,
    title: "Deep Learning",
    description: "Neural networks trained on millions of medical images for pattern recognition",
  },
  {
    icon: Layers,
    title: "DenseNet Architecture",
    description: "State-of-the-art convolutional networks with dense connections for feature extraction",
  },
  {
    icon: Eye,
    title: "Grad-CAM Visualization",
    description: "Visual explanations showing exactly which regions influenced the diagnosis",
  },
  {
    icon: GitBranch,
    title: "Transfer Learning",
    description: "Pre-trained models fine-tuned on specialized medical imaging datasets",
  },
];

const AITechnologySection = () => {
  return (
    <section className="py-24 relative bg-muted/30 overflow-hidden">
      {/* Background decoration */}
      <div className="absolute inset-0 opacity-30">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-primary/10 rounded-full blur-3xl" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-accent/10 rounded-full blur-3xl" />
      </div>

      <div className="container mx-auto px-6 relative">
        <div className="grid lg:grid-cols-2 gap-16 items-center">
          {/* Left Content */}
          <div className="space-y-8">
            <div>
              <span className="inline-block px-4 py-1.5 rounded-full bg-secondary text-secondary-foreground text-sm font-medium mb-4">
                Our Technology
              </span>
              <h2 className="text-3xl sm:text-4xl font-bold text-foreground mb-4">
                Powered by Advanced{" "}
                <span className="text-gradient">AI Technology</span>
              </h2>
              <p className="text-muted-foreground text-lg leading-relaxed">
                We leverage cutting-edge deep learning architectures to deliver accurate, 
                explainable, and trustworthy medical image analysis.
              </p>
            </div>

            {/* Technology grid */}
            <div className="grid sm:grid-cols-2 gap-6">
              {technologies.map((tech, index) => (
                <div
                  key={tech.title}
                  className="flex gap-4 p-4 rounded-2xl bg-card/50 border border-border/50 hover:bg-card transition-colors duration-300"
                  style={{ animationDelay: `${index * 0.1}s` }}
                >
                  <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center flex-shrink-0">
                    <tech.icon className="w-6 h-6 text-primary" />
                  </div>
                  <div>
                    <h4 className="font-semibold text-foreground mb-1">{tech.title}</h4>
                    <p className="text-sm text-muted-foreground">{tech.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Right Content - AI Visualization */}
          <div className="relative">
            <div className="glass-card p-8">
              {/* Neural network visualization */}
              <div className="aspect-square rounded-2xl bg-gradient-to-br from-muted to-muted/50 flex items-center justify-center overflow-hidden relative">
                {/* Animated neural network illustration */}
                <svg
                  viewBox="0 0 400 400"
                  className="w-full h-full"
                  style={{ maxWidth: "400px" }}
                >
                  {/* Input layer nodes */}
                  {[80, 140, 200, 260, 320].map((y, i) => (
                    <g key={`input-${i}`}>
                      <circle
                        cx="50"
                        cy={y}
                        r="15"
                        fill="hsl(var(--primary) / 0.2)"
                        stroke="hsl(var(--primary))"
                        strokeWidth="2"
                        className="animate-pulse"
                        style={{ animationDelay: `${i * 0.2}s` }}
                      />
                    </g>
                  ))}

                  {/* Hidden layer 1 */}
                  {[100, 160, 220, 280].map((y, i) => (
                    <g key={`hidden1-${i}`}>
                      <circle
                        cx="150"
                        cy={y}
                        r="12"
                        fill="hsl(var(--accent) / 0.2)"
                        stroke="hsl(var(--accent))"
                        strokeWidth="2"
                        className="animate-pulse"
                        style={{ animationDelay: `${i * 0.15 + 0.3}s` }}
                      />
                    </g>
                  ))}

                  {/* Hidden layer 2 */}
                  {[120, 200, 280].map((y, i) => (
                    <g key={`hidden2-${i}`}>
                      <circle
                        cx="250"
                        cy={y}
                        r="12"
                        fill="hsl(var(--primary) / 0.3)"
                        stroke="hsl(var(--primary))"
                        strokeWidth="2"
                        className="animate-pulse"
                        style={{ animationDelay: `${i * 0.2 + 0.5}s` }}
                      />
                    </g>
                  ))}

                  {/* Output layer */}
                  {[160, 240].map((y, i) => (
                    <g key={`output-${i}`}>
                      <circle
                        cx="350"
                        cy={y}
                        r="18"
                        fill="hsl(var(--accent) / 0.3)"
                        stroke="hsl(var(--accent))"
                        strokeWidth="3"
                        className="animate-pulse"
                        style={{ animationDelay: `${i * 0.3 + 0.7}s` }}
                      />
                    </g>
                  ))}

                  {/* Connection lines */}
                  <g stroke="hsl(var(--primary) / 0.15)" strokeWidth="1">
                    {[80, 140, 200, 260, 320].map((y1) =>
                      [100, 160, 220, 280].map((y2) => (
                        <line key={`l1-${y1}-${y2}`} x1="65" y1={y1} x2="138" y2={y2} />
                      ))
                    )}
                    {[100, 160, 220, 280].map((y1) =>
                      [120, 200, 280].map((y2) => (
                        <line key={`l2-${y1}-${y2}`} x1="162" y1={y1} x2="238" y2={y2} />
                      ))
                    )}
                    {[120, 200, 280].map((y1) =>
                      [160, 240].map((y2) => (
                        <line key={`l3-${y1}-${y2}`} x1="262" y1={y1} x2="332" y2={y2} />
                      ))
                    )}
                  </g>
                </svg>

                {/* Labels */}
                <div className="absolute bottom-4 left-4 right-4 flex justify-between text-xs text-muted-foreground font-medium">
                  <span>Input</span>
                  <span>Hidden Layers</span>
                  <span>Output</span>
                </div>
              </div>

              {/* Caption */}
              <p className="text-center text-sm text-muted-foreground mt-4">
                DenseNet-121 Architecture Visualization
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default AITechnologySection;
