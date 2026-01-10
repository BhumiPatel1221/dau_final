const ScanDoodles = () => {
  return (
    <div className="absolute inset-0 pointer-events-none select-none" aria-hidden="true">
      <svg
        className="absolute inset-0 w-full h-full opacity-[0.06] text-primary"
        viewBox="0 0 1200 520"
        fill="none"
        preserveAspectRatio="none"
      >
        <path
          d="M0 80H1200"
          stroke="currentColor"
          strokeWidth="1"
          strokeDasharray="6 10"
        />
        <path
          d="M0 160H1200"
          stroke="currentColor"
          strokeWidth="1"
          strokeDasharray="6 10"
        />
        <path
          d="M0 240H1200"
          stroke="currentColor"
          strokeWidth="1"
          strokeDasharray="6 10"
        />
        <path
          d="M0 320H1200"
          stroke="currentColor"
          strokeWidth="1"
          strokeDasharray="6 10"
        />
      </svg>

      <div className="absolute inset-x-0 top-0 h-full overflow-hidden opacity-[0.08]">
        <div className="absolute -left-1/2 top-0 h-full w-[200%] bg-gradient-to-r from-transparent via-primary/30 to-transparent animate-scan-sweep" />
      </div>

      <svg
        className="absolute -bottom-2 left-1/2 -translate-x-1/2 w-[520px] h-[120px] opacity-[0.08] text-accent animate-doodle-pulse"
        viewBox="0 0 520 120"
        fill="none"
      >
        <path
          d="M10 70h90l18-32 22 65 22-55 16 20h120l20-45 24 80 24-55 14 16h114"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </div>
  );
};

export default ScanDoodles;
