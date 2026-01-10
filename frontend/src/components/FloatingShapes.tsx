const FloatingShapes = () => {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {/* Top right large blob */}
      <div 
        className="floating-shape w-[600px] h-[600px] -top-40 -right-40"
        style={{ animationDelay: "0s" }}
      />
      
      {/* Bottom left blob */}
      <div 
        className="floating-shape w-[400px] h-[400px] -bottom-20 -left-20"
        style={{ animationDelay: "5s" }}
      />
      
      {/* Center right small blob */}
      <div 
        className="floating-shape w-[300px] h-[300px] top-1/2 right-20"
        style={{ animationDelay: "10s", opacity: 0.2 }}
      />
      
      {/* Top left small blob */}
      <div 
        className="floating-shape w-[200px] h-[200px] top-40 left-20"
        style={{ animationDelay: "7s", opacity: 0.15 }}
      />
      
      {/* Medical cross patterns */}
      <svg
        className="absolute top-20 right-1/4 w-16 h-16 text-primary/10 animate-float"
        viewBox="0 0 24 24"
        fill="currentColor"
        style={{ animationDelay: "2s" }}
      >
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-2 10h-4v4h-2v-4H7v-2h4V7h2v4h4v2z" />
      </svg>
      
      <svg
        className="absolute bottom-40 left-1/3 w-12 h-12 text-accent/10 animate-float"
        viewBox="0 0 24 24"
        fill="currentColor"
        style={{ animationDelay: "4s" }}
      >
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-2 10h-4v4h-2v-4H7v-2h4V7h2v4h4v2z" />
      </svg>
      
      {/* DNA helix pattern */}
      <svg
        className="absolute top-1/3 left-10 w-8 h-8 text-primary/5 animate-float"
        viewBox="0 0 24 24"
        fill="currentColor"
        style={{ animationDelay: "6s" }}
      >
        <circle cx="12" cy="12" r="10" />
      </svg>
    </div>
  );
};

export default FloatingShapes;
