const MedicalDoodles = () => {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none select-none" aria-hidden="true">
      <svg
        className="absolute -top-10 -left-10 w-[520px] h-[520px] opacity-[0.08] text-primary animate-float"
        viewBox="0 0 600 600"
        fill="none"
        style={{ animationDuration: "16s" }}
      >
        <rect x="90" y="110" width="360" height="260" rx="32" stroke="currentColor" strokeWidth="2" />
        <path
          d="M120 160H420"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeDasharray="6 10"
        />
        <path
          d="M140 220c60-70 110-70 170 0s110 70 170 0"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="[stroke-dasharray:520] [stroke-dashoffset:520] animate-doodle-draw"
          style={{ animationDuration: "6s" }}
        />
        <path
          d="M150 290h70l20-30 25 60 25-60 25 60 25-45h70"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="[stroke-dasharray:520] [stroke-dashoffset:520] animate-doodle-draw"
          style={{ animationDuration: "7s", animationDelay: "0.5s" }}
        />
      </svg>

      <svg
        className="absolute top-24 right-10 w-[320px] h-[320px] opacity-[0.07] text-accent animate-float"
        viewBox="0 0 400 400"
        fill="none"
        style={{ animationDuration: "20s", animationDelay: "1.2s" }}
      >
        <path
          d="M200 70c-38 0-70 34-70 78 0 33 18 61 44 73l26 14 26-14c26-12 44-40 44-73 0-44-32-78-70-78Z"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinejoin="round"
          className="[stroke-dasharray:640] [stroke-dashoffset:640] animate-doodle-draw"
          style={{ animationDuration: "8s" }}
        />
        <path
          d="M200 92v130"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          className="[stroke-dasharray:180] [stroke-dashoffset:180] animate-doodle-draw"
          style={{ animationDuration: "5s", animationDelay: "0.4s" }}
        />
        <path
          d="M170 150h60"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          className="[stroke-dasharray:120] [stroke-dashoffset:120] animate-doodle-draw"
          style={{ animationDuration: "5s", animationDelay: "0.6s" }}
        />
      </svg>

      <svg
        className="absolute bottom-16 left-1/2 -translate-x-1/2 w-[760px] h-[160px] opacity-[0.06] text-primary animate-doodle-pulse"
        viewBox="0 0 800 180"
        fill="none"
        style={{ animationDuration: "7s" }}
      >
        <path
          d="M20 110h160l30-55 36 110 36-90 30 35h170l32-70 38 125 38-85 26 30h156"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>

      <svg
        className="absolute -bottom-10 -right-10 w-[520px] h-[520px] opacity-[0.06] text-primary animate-float"
        viewBox="0 0 600 600"
        fill="none"
        style={{ animationDuration: "22s", animationDelay: "0.8s" }}
      >
        <path
          d="M250 130c-55 40-45 110 10 150 42 30 68 74 60 126-2 16-7 33-16 49"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          className="[stroke-dasharray:520] [stroke-dashoffset:520] animate-doodle-draw"
          style={{ animationDuration: "9s" }}
        />
        <path
          d="M350 130c55 40 45 110-10 150-42 30-68 74-60 126 2 16 7 33 16 49"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          className="[stroke-dasharray:520] [stroke-dashoffset:520] animate-doodle-draw"
          style={{ animationDuration: "9s", animationDelay: "0.7s" }}
        />
        <path
          d="M250 210h100"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeDasharray="8 12"
        />
        <path
          d="M250 260h100"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeDasharray="8 12"
        />
        <path
          d="M250 310h100"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeDasharray="8 12"
        />
      </svg>
    </div>
  );
};

export default MedicalDoodles;
