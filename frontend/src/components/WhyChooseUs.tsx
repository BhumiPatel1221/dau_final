import { CheckCircle, Clock, MapPin, ShieldCheck, HeartPulse, Award } from "lucide-react";

const benefits = [
  {
    icon: CheckCircle,
    title: "98%+ Accuracy",
    description: "Clinical-grade diagnostic accuracy validated against expert radiologists",
  },
  {
    icon: Clock,
    title: "30 Second Results",
    description: "Get instant analysis without waiting hours or days for results",
  },
  {
    icon: MapPin,
    title: "Rural Healthcare",
    description: "Bringing expert-level diagnosis to underserved and remote communities",
  },
  {
    icon: ShieldCheck,
    title: "HIPAA Compliant",
    description: "Your medical data is encrypted and handled with utmost security",
  },
  {
    icon: HeartPulse,
    title: "24/7 Availability",
    description: "Access diagnostic services anytime, anywhere in the world",
  },
  {
    icon: Award,
    title: "FDA Pathway",
    description: "Designed following regulatory guidelines for medical devices",
  },
];

const WhyChooseUs = () => {
  return (
    <section className="py-24 relative">
      <div className="container mx-auto px-6">
        {/* Section Header */}
        <div className="text-center max-w-2xl mx-auto mb-16">
          <span className="inline-block px-4 py-1.5 rounded-full bg-secondary text-secondary-foreground text-sm font-medium mb-4">
            Why MediAI
          </span>
          <h2 className="text-3xl sm:text-4xl font-bold text-foreground mb-4">
            Why <span className="text-gradient">Choose Us</span>
          </h2>
          <p className="text-muted-foreground text-lg">
            We're committed to making advanced medical diagnostics accessible, accurate, and affordable for everyone.
          </p>
        </div>

        {/* Benefits Grid */}
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {benefits.map((benefit, index) => (
            <div
              key={benefit.title}
              className="group p-6 rounded-2xl border border-border bg-card hover:bg-secondary/30 transition-all duration-300 hover:shadow-medical-md"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div className="flex items-start gap-4">
                <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center flex-shrink-0 group-hover:bg-primary/20 transition-colors">
                  <benefit.icon className="w-6 h-6 text-primary" />
                </div>
                <div>
                  <h3 className="font-semibold text-foreground mb-2">{benefit.title}</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">{benefit.description}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default WhyChooseUs;
