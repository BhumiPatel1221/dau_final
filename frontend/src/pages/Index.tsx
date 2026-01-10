import Navbar from "@/components/Navbar";
import HeroSection from "@/components/HeroSection";
import HowItWorks from "@/components/HowItWorks";
import DiseasesSection from "@/components/DiseasesSection";
import Footer from "@/components/Footer";
import FloatingShapes from "@/components/FloatingShapes";
import MedicalDoodles from "@/components/MedicalDoodles";
import Reveal from "@/components/Reveal";

const Index = () => {
  return (
    <div className="min-h-screen bg-background relative overflow-x-hidden">
      {/* Floating background shapes */}
      <FloatingShapes />

      {/* Medical doodles */}
      <MedicalDoodles />
      
      <div className="relative z-10">
        {/* Navbar */}
        <Navbar />
        
        {/* Main Content */}
        <main>
          <HeroSection />
          <Reveal delayMs={50}>
            <HowItWorks />
          </Reveal>
          <Reveal delayMs={100}>
            <DiseasesSection />
          </Reveal>
        </main>
        
        {/* Footer */}
        <Reveal delayMs={50}>
          <Footer />
        </Reveal>
      </div>
    </div>
  );
};

export default Index;
