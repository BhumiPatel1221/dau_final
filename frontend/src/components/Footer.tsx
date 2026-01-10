 const Footer = () => {
  return (
    <footer id="contact" className="bg-muted/50 border-t border-border">
      <div className="container mx-auto px-6 py-16">
        <div className="flex flex-col items-center text-center">
          <div className="flex items-center gap-2 mb-4">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-accent flex items-center justify-center shadow-medical-sm">
              <svg
                viewBox="0 0 24 24"
                fill="none"
                className="w-6 h-6 text-primary-foreground"
                stroke="currentColor"
                strokeWidth="2"
              >
                <path d="M12 4v16M4 12h16" strokeLinecap="round" />
              </svg>
            </div>
            <span className="text-xl font-bold text-foreground">
              Medi<span className="text-primary">AI</span>
            </span>
          </div>
          <p className="text-muted-foreground text-sm leading-relaxed max-w-xl">
            Revolutionizing healthcare with AI-powered medical imaging analysis.
          </p>
        </div>
      </div>
    </footer>
  );
 };

export default Footer;
