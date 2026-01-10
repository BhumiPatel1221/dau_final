import Navbar from "@/components/Navbar";
import UploadSection from "@/components/UploadSection";
import FloatingShapes from "@/components/FloatingShapes";

const Dashboard = () => {
  return (
    <div className="min-h-screen bg-background relative overflow-x-hidden">
      <FloatingShapes />
      <Navbar />
      <main>
        <UploadSection />
      </main>
    </div>
  );
};

export default Dashboard;
