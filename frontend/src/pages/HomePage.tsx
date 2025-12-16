import { Link } from "react-router-dom";
import { Brain, Sparkles, Layers, LineChart, Zap, GitCompare, Server, GraduationCap, ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import heroBg from "@/assets/hero-bg.jpg";

const features = [
  {
    icon: Sparkles,
    title: "Advanced NLP Preprocessing Pipeline",
    description: "The system applies a comprehensive Natural Language Processing pipeline to clean and normalize raw user input. This includes lowercasing, URL removal, HTML tag removal, emoji removal, special character and number removal, repeated character normalization, tokenization, stopword removal, and lemmatization. This enables the system to handle informal, noisy, and user-generated text effectively before classification.",
  },
  {
    icon: Layers,
    title: "Multi-Model Machine Learning Architecture",
    description: "Instead of relying on a single classifier, the system uses six lightweight machine learning models: Logistic Regression for linear classification, Linear SVM for high-dimensional sparse text data, Multinomial Naive Bayes for probabilistic text classification, Complement Naive Bayes for imbalanced data, Bernoulli Naive Bayes for binary features, and SGD Classifier for scalable learning. This memory-efficient approach enables deployment on resource-constrained environments.",
  },
  {
    icon: LineChart,
    title: "Transparent Model Comparison",
    description: "The application displays predictions from all models along with their accuracy and F1-score. This transparency allows users to understand how different algorithms interpret the same input and why one model performs better than others.",
  },
  {
    icon: Zap,
    title: "Real-Time Inference System",
    description: "User input is processed in real time through a backend ML API. The system performs preprocessing, feature extraction, and prediction instantly without retraining, ensuring fast and efficient responses.",
  },
  {
    icon: GitCompare,
    title: "Best Model Selection Logic",
    description: "Based on evaluation metrics collected during training, the system identifies and highlights the best-performing model. This mirrors how real-world ML systems select models for deployment.",
  },
  {
    icon: Brain,
    title: "Consistent Training and Inference Pipeline",
    description: "The same preprocessing and feature extraction steps used during training are reused during inference, ensuring consistency, reliability, and stable predictions.",
  },
  {
    icon: Server,
    title: "Full-Stack ML System Design",
    description: "The project follows a full-stack architecture: Frontend for user interaction and visualization, FastAPI-based backend ML microservice, and pretrained machine learning models loaded at startup. This separation improves scalability and maintainability.",
  },
  {
    icon: GraduationCap,
    title: "Educational and Explainable ML Focus",
    description: "Beyond prediction, the system is designed to be educational. It helps users and learners understand NLP preprocessing, machine learning model behavior, and performance evaluation.",
  },
];

const techStack = [
  {
    category: "Frontend",
    tech: "React + TypeScript + Vite",
    icon: "⚛️",
    achievements: [
      "Built responsive UI with Tailwind CSS and shadcn/ui components",
      "Implemented real-time classification with dynamic prediction confidence display",
      "Created split-screen layout for optimal data visualization",
      "Added gradient animations and modern design system"
    ]
  },
  {
    category: "Backend",
    tech: "FastAPI + Python 3.13",
    icon: "⚡",
    achievements: [
      "Developed RESTful API with CORS middleware",
      "Implemented lifespan events for efficient model loading",
      "Created preprocessing pipeline with complete NLP steps",
      "Added environment-based configuration with python-dotenv"
    ]
  },
  {
    category: "NLP Processing",
    tech: "NLTK + scikit-learn",
    icon: "🔤",
    achievements: [
      "Implemented TF-IDF vectorization with unigrams + bigrams",
      "Built comprehensive text preprocessing (tokenization, lemmatization, stopword removal)",
      "Created emoji and URL removal for social media text",
      "Normalized repeated characters and special symbols"
    ]
  },
  {
    category: "Machine Learning",
    tech: "6 Lightweight ML Models",
    icon: "🤖",
    achievements: [
      "Trained SGD Classifier (72.20% accuracy, 71.87% F1) - Best Model",
      "Trained Logistic Regression (71.83% accuracy, 71.76% F1)",
      "Trained Linear SVM (71.49% accuracy, 71.56% F1)",
      "Trained Complement Naive Bayes (71.43% accuracy, 70.91% F1)",
      "Trained Multinomial Naive Bayes (71.11% accuracy, 70.00% F1)",
      "Trained Bernoulli Naive Bayes (56.39% accuracy, 58.47% F1)"
    ]
  },
  {
    category: "Dataset",
    tech: "20 Newsgroups",
    icon: "📰",
    achievements: [
      "Trained on 18,846 documents across 20 categories",
      "Handled diverse topics: tech, politics, sports, science, religion",
      "Preprocessed with complete NLP pipeline",
      "Achieved cross-validated performance metrics"
    ]
  },
  {
    category: "Deployment",
    tech: "Memory-Safe Model Persistence",
    icon: "💾",
    achievements: [
      "Saved all 6 lightweight models individually as .pkl files",
      "Implemented sequential model loading for 512MB RAM constraint",
      "Created SafeModelLoader with garbage collection after each prediction",
      "Stored TF-IDF vectorizer and metadata for consistent inference",
      "Optimized for Render Free Tier deployment"
    ]
  },
];

const HomePage = () => {
  return (
    <div className="min-h-screen bg-background">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-secondary/90 backdrop-blur-md border-b border-secondary">
        <div className="container mx-auto px-4 h-16 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2">
            <Brain className="h-8 w-8 text-primary" />
            <span className="font-semibold text-lg text-secondary-foreground">NewsSense</span>
          </Link>
          <Link to="/classify">
            <Button variant="default" size="sm">
              Try Classification
              <ArrowRight className="ml-1 h-4 w-4" />
            </Button>
          </Link>
        </div>
      </nav>

      {/* Hero Section with Background Image */}
      <section 
        className="relative pt-32 pb-24 px-4 overflow-hidden"
        style={{
          backgroundImage: `url(${heroBg})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
        }}
      >
        {/* Overlay for better text readability */}
        <div className="absolute inset-0 bg-secondary/70 backdrop-blur-[2px]" />
        
        <div className="container mx-auto text-center max-w-4xl relative z-10">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/20 border border-primary/30 text-secondary-foreground text-sm mb-6 animate-fade-in">
            <Brain className="h-4 w-4 text-primary" />
            <span>Powered by Machine Learning</span>
          </div>
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-secondary-foreground mb-6 leading-tight animate-fade-in" style={{ animationDelay: '0.1s' }}>
            ML-Based Multiclass
            <span className="block text-primary drop-shadow-lg">News Classification</span>
          </h1>
          <p className="text-lg md:text-xl text-secondary-foreground/90 mb-6 max-w-3xl mx-auto leading-relaxed animate-fade-in" style={{ animationDelay: '0.2s' }}>
            An end-to-end NLP and Machine Learning system that cleans noisy text,
            compares multiple ML models, and transparently shows predictions and performance.
          </p>
          
          {/* Developer Credit */}
          <div className="flex justify-center mb-10 animate-fade-in" style={{ animationDelay: '0.25s' }}>
            <div className="inline-flex items-center gap-3 px-6 py-3 rounded-xl bg-gradient-to-r from-primary/20 via-primary/15 to-primary/20 border border-primary/30 backdrop-blur-sm shadow-lg hover:shadow-primary/20 transition-all duration-300 hover:scale-105">
              <GraduationCap className="h-5 w-5 text-primary" />
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-secondary-foreground/80">Developed by</span>
                <span className="text-base font-bold bg-gradient-to-r from-primary to-primary/70 bg-clip-text text-transparent">
                  Naveen Arul
                </span>
              </div>
            </div>
          </div>

          <div className="flex flex-col sm:flex-row gap-4 justify-center animate-fade-in" style={{ animationDelay: '0.3s' }}>
            <Link to="/classify">
              <Button variant="hero" size="xl" className="w-full sm:w-auto glow-primary">
                Try News Classification
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
            </Link>
            <a href="#features">
              <Button variant="hero-outline" size="xl" className="w-full sm:w-auto">
                Learn More
              </Button>
            </a>
          </div>
        </div>

        {/* Decorative gradient at bottom */}
        <div className="absolute bottom-0 left-0 right-0 h-24 bg-gradient-to-t from-background to-transparent" />
      </section>

      {/* Features Section with Background */}
      <section 
        id="features" 
        className="py-20 px-4 relative"
        style={{
          backgroundImage: `url(${heroBg})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundAttachment: 'fixed',
        }}
      >
        {/* Overlay */}
        <div className="absolute inset-0 bg-background/95" />
        
        <div className="container mx-auto max-w-7xl relative z-10">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
              System Features
            </h2>
            <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
              A comprehensive machine learning pipeline designed for transparency, education, and real-world application.
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 gap-6">
            {features.map((feature, index) => (
              <div
                key={index}
                className="group bg-card/80 backdrop-blur-sm rounded-xl p-6 border border-border card-shadow hover:card-shadow-hover transition-all duration-300 hover:-translate-y-1"
              >
                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0 w-12 h-12 rounded-lg bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center group-hover:from-primary/30 group-hover:to-accent/30 transition-all duration-300">
                    <feature.icon className="h-6 w-6 text-primary" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-card-foreground mb-2 group-hover:text-primary transition-colors">
                      {feature.title}
                    </h3>
                    <p className="text-muted-foreground text-sm leading-relaxed">
                      {feature.description}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Tech Stack Section */}
      <section 
        className="py-20 px-4 relative overflow-hidden"
        style={{
          backgroundImage: `url(${heroBg})`,
          backgroundSize: 'cover',
          backgroundPosition: 'bottom center',
        }}
      >
        {/* Overlay */}
        <div className="absolute inset-0 bg-secondary/85 backdrop-blur-sm" />
        
        <div className="container mx-auto max-w-7xl relative z-10">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold text-secondary-foreground mb-4">
              Technology Stack & Achievements
            </h2>
            <p className="text-secondary-foreground/80 text-lg">
              Built with modern technologies and comprehensive ML implementation.
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {techStack.map((item, index) => (
              <div
                key={index}
                className="group bg-gradient-to-br from-card/95 to-card/85 backdrop-blur-md rounded-2xl p-6 border-2 border-border/50 hover:border-primary/40 card-shadow hover:card-shadow-hover transition-all duration-300 hover:-translate-y-2 relative overflow-hidden"
              >
                <div className="absolute top-0 right-0 w-32 h-32 bg-primary/5 rounded-full blur-3xl group-hover:bg-primary/10 transition-all duration-500"></div>
                <div className="relative">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="text-4xl">{item.icon}</div>
                    <div className="flex-1">
                      <p className="text-xs font-bold text-primary uppercase tracking-wider mb-1">
                        {item.category}
                      </p>
                      <p className="text-card-foreground font-bold text-lg">
                        {item.tech}
                      </p>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
                      Key Achievements:
                    </p>
                    <ul className="space-y-2">
                      {item.achievements.map((achievement, idx) => (
                        <li key={idx} className="flex items-start gap-2 text-sm text-muted-foreground">
                          <span className="text-primary mt-1">✓</span>
                          <span className="flex-1 leading-relaxed">{achievement}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 bg-background relative">
        <div className="container mx-auto max-w-3xl text-center">
          <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-6">
            Ready to Classify News?
          </h2>
          <p className="text-muted-foreground text-lg mb-8">
            Enter any news article and see how our multi-model system processes and classifies it in real time.
          </p>
          <Link to="/classify">
            <Button variant="default" size="xl" className="shadow-lg hover:shadow-xl transition-shadow">
              Try News Classification
              <ArrowRight className="ml-2 h-5 w-5" />
            </Button>
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 px-4 border-t border-border bg-muted/20">
        <div className="container mx-auto text-center">
          <div className="flex items-center justify-center gap-2 mb-2">
            <Brain className="h-5 w-5 text-primary" />
            <span className="font-semibold text-foreground">NewsSense</span>
          </div>
          <p className="text-sm text-muted-foreground">
            ML-Based Multiclass News Classification System
          </p>
        </div>
      </footer>
    </div>
  );
};

export default HomePage;
