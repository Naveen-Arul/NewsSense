import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { Brain, ArrowLeft, Send, Loader2, Trophy, AlertCircle, Sparkles, TrendingUp, Award, Target, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";
import heroBg from "@/assets/hero-bg.jpg";

interface ModelResult {
  model: string;
  prediction: string;
  prediction_confidence: number;
}

interface PredictionResponse {
  cleaned_text: string;
  results: ModelResult[];
  best_model: string;
}

interface ModelInfo {
  name: string;
  accuracy: number;
  f1_score: number;
}

const ClassifyPage = () => {
  const [inputText, setInputText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [modelMetrics, setModelMetrics] = useState<ModelInfo[]>([]);
  const { toast } = useToast();

  // Fetch model metrics on component mount
  useEffect(() => {
    const fetchModelMetrics = async () => {
      try {
        const response = await fetch("http://localhost:8000/models");
        const data = await response.json();
        setModelMetrics(data.models);
      } catch (err) {
        console.error("Failed to fetch model metrics:", err);
      }
    };
    fetchModelMetrics();
  }, []);

  const handleSubmit = async () => {
    if (!inputText.trim()) {
      toast({
        title: "Input Required",
        description: "Please enter a news article to classify.",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: inputText }),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      
      // API returns results array directly
      setResult(data);
      toast({
        title: "Classification Complete",
        description: "Your news article has been successfully classified.",
      });
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to connect to the API";
      setError(errorMessage);
      toast({
        title: "Classification Failed",
        description: "Could not connect to the backend API. Please ensure the server is running.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const formatPercentage = (value: number) => {
    return (value * 100).toFixed(2) + "%";
  };

  // Get best model by accuracy from training
  const bestModelByAccuracy = modelMetrics.length > 0 
    ? modelMetrics.reduce((prev, current) => (prev.accuracy > current.accuracy) ? prev : current)
    : null;

  // Get best model by prediction confidence from current prediction
  const bestModelByConfidence = result 
    ? result.results.reduce((prev, current) => (prev.prediction_confidence > current.prediction_confidence) ? prev : current)
    : null;

  return (
    <div 
      className="min-h-screen relative"
      style={{
        backgroundImage: `url(${heroBg})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        backgroundAttachment: 'fixed',
      }}
    >
      {/* Background overlay */}
      <div className="absolute inset-0 bg-background/90 backdrop-blur-[3px]" />

      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-secondary/90 backdrop-blur-md border-b border-secondary">
        <div className="container mx-auto px-4 h-16 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2">
            <Brain className="h-8 w-8 text-primary" />
            <span className="font-semibold text-lg text-secondary-foreground">NewsSense</span>
          </Link>
          <Link to="/">
            <Button variant="ghost" size="sm" className="text-secondary-foreground hover:text-primary">
              <ArrowLeft className="mr-1 h-4 w-4" />
              Back to Home
            </Button>
          </Link>
        </div>
      </nav>

      <main className="pt-24 pb-16 px-4 relative z-10">
        <div className="container mx-auto max-w-7xl">
          {/* Header */}
          <div className="text-center mb-10">
            <h1 className="text-3xl md:text-4xl font-bold text-foreground mb-3">
              News Classification
            </h1>
            <p className="text-muted-foreground text-lg">
              Enter a news article below to see how our ML models classify it.
            </p>
          </div>

          {/* Two Column Layout */}
          <div className="grid lg:grid-cols-2 gap-8">
            {/* Left Side - Input & Cleaned Text */}
            <div className="space-y-6">
              {/* Input Section */}
              <div className="bg-gradient-to-br from-card/90 to-card/70 backdrop-blur-md rounded-2xl border-2 border-border/50 card-shadow p-6 relative overflow-hidden hover:border-primary/30 transition-all duration-300">
                <div className="absolute top-0 left-0 w-32 h-32 bg-primary/5 rounded-full blur-3xl"></div>
                <div className="relative">
                  <label htmlFor="news-input" className="flex items-center gap-2 text-sm font-semibold text-card-foreground mb-3">
                    <Sparkles className="h-4 w-4 text-primary" />
                    News Article Text
                  </label>
                <Textarea
                  id="news-input"
                  placeholder="Paste or type your news article here..."
                  className="min-h-[400px] resize-y mb-4 bg-background/80"
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  disabled={isLoading}
                />
                  <Button
                    onClick={handleSubmit}
                    disabled={isLoading || !inputText.trim()}
                    className="w-full bg-gradient-to-r from-primary to-primary/80 hover:from-primary/90 hover:to-primary/70 shadow-lg shadow-primary/30 hover:shadow-xl hover:shadow-primary/40 transition-all duration-300 hover:scale-[1.02]"
                    size="lg"
                  >
                    {isLoading ? (
                      <>
                        <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                        Classifying...
                      </>
                    ) : (
                      <>
                        <Zap className="mr-2 h-5 w-5" />
                        Classify News
                      </>
                    )}
                  </Button>
                </div>
              </div>

              {/* Cleaned Text */}
              {result && (
                <div className="bg-gradient-to-br from-card/90 to-card/70 backdrop-blur-md rounded-2xl border-2 border-green-500/30 card-shadow p-6 animate-fade-in relative overflow-hidden">
                  <div className="absolute top-0 right-0 w-24 h-24 bg-green-500/10 rounded-full blur-2xl"></div>
                  <div className="relative">
                    <div className="flex items-center gap-2 mb-3">
                      <div className="w-8 h-8 rounded-lg bg-green-500/20 flex items-center justify-center">
                        <Sparkles className="w-4 h-4 text-green-600" />
                      </div>
                      <h2 className="text-lg font-bold text-card-foreground">
                        Preprocessed Text
                      </h2>
                    </div>
                    <div className="bg-gradient-to-br from-muted/60 to-muted/40 rounded-xl p-4 max-h-[200px] overflow-y-auto border border-border/50">
                      <p className="text-sm text-muted-foreground font-mono leading-relaxed">
                        {result.cleaned_text}
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Right Side - Model Performance & Prediction Results */}
            <div className="space-y-6">
              {/* Model Performance Metrics (Always Visible) */}
              <div className="bg-gradient-to-br from-card/90 to-card/70 backdrop-blur-md rounded-2xl border-2 border-primary/20 card-shadow p-6 relative overflow-hidden">
                <div className="absolute top-0 right-0 w-32 h-32 bg-primary/5 rounded-full blur-3xl"></div>
                <div className="relative">
                  <div className="flex items-center gap-2 mb-6">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary/20 to-primary/10 flex items-center justify-center">
                      <Trophy className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                      <h2 className="text-lg font-bold text-card-foreground">
                        Model Performance
                      </h2>
                      <p className="text-xs text-muted-foreground">Test dataset results</p>
                    </div>
                  </div>
                  {modelMetrics.length > 0 ? (
                    <div className="space-y-3">
                      {modelMetrics.map((model, index) => {
                        const isBest = bestModelByAccuracy?.name === model.name;
                        const modelIcons = [Award, Target, Zap, TrendingUp];
                        const ModelIcon = modelIcons[index % modelIcons.length];
                        return (
                          <div
                            key={index}
                            className={`group relative flex items-center justify-between p-4 rounded-xl transition-all duration-300 hover:scale-[1.02] ${
                              isBest 
                                ? "bg-gradient-to-r from-primary/20 to-primary/10 border-2 border-primary/40 shadow-lg shadow-primary/20" 
                                : "bg-muted/40 border border-border/50 hover:border-primary/30 hover:bg-muted/60"
                            }`}
                          >
                            <div className="flex items-center gap-3">
                              <div className={`w-8 h-8 rounded-lg flex items-center justify-center transition-all ${
                                isBest ? "bg-primary/20" : "bg-background/50 group-hover:bg-primary/10"
                              }`}>
                                {isBest ? (
                                  <Trophy className="h-4 w-4 text-primary" />
                                ) : (
                                  <ModelIcon className="h-4 w-4 text-muted-foreground group-hover:text-primary" />
                                )}
                              </div>
                              <span className={`font-semibold text-sm ${
                                isBest ? "text-primary" : "text-card-foreground"
                              }`}>
                                {model.name}
                              </span>
                              {isBest && (
                                <span className="px-2 py-0.5 text-xs font-bold bg-primary/20 text-primary rounded-full border border-primary/30">
                                  BEST
                                </span>
                              )}
                            </div>
                            <div className="flex gap-6 text-sm">
                              <div className="text-right">
                                <span className="text-muted-foreground text-xs block mb-1">Accuracy</span>
                                <span className={`font-mono font-bold ${
                                  isBest ? "text-primary" : "text-card-foreground"
                                }`}>{formatPercentage(model.accuracy)}</span>
                              </div>
                              <div className="text-right">
                                <span className="text-muted-foreground text-xs block mb-1">F1-Score</span>
                                <span className={`font-mono font-bold ${
                                  isBest ? "text-primary" : "text-card-foreground"
                                }`}>{formatPercentage(model.f1_score)}</span>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <div className="flex items-center gap-2 text-muted-foreground">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <p className="text-sm">Loading model metrics...</p>
                    </div>
                  )}
                </div>
              </div>

              {/* Error State */}
              {error && (
                <div className="bg-destructive/10 backdrop-blur-sm border border-destructive/30 rounded-xl p-6 animate-fade-in">
                  <div className="flex items-start gap-3">
                    <AlertCircle className="h-5 w-5 text-destructive mt-0.5" />
                    <div>
                      <h3 className="font-semibold text-destructive mb-1">Connection Error</h3>
                      <p className="text-destructive/80 text-sm">
                        {error}. Make sure the backend API is running at http://localhost:8000
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Model Results Table */}
              {result && (
                <>
                  <div className="bg-gradient-to-br from-card/90 to-card/70 backdrop-blur-md rounded-2xl border-2 border-primary/20 card-shadow p-6 animate-fade-in relative overflow-hidden">
                    <div className="absolute bottom-0 left-0 w-32 h-32 bg-accent/10 rounded-full blur-3xl"></div>
                    <div className="relative">
                      <div className="flex items-center gap-2 mb-6">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-accent/20 to-accent/10 flex items-center justify-center">
                          <Brain className="h-5 w-5 text-accent" />
                        </div>
                        <div>
                          <h2 className="text-lg font-bold text-card-foreground">
                            Model Predictions
                          </h2>
                          <p className="text-xs text-muted-foreground">Real-time classification results</p>
                        </div>
                      </div>
                    <div className="overflow-x-auto">
                      <table className="w-full">
                        <thead>
                          <tr className="border-b border-border">
                            <th className="text-left py-3 px-4 text-sm font-semibold text-muted-foreground uppercase tracking-wider">
                              Model
                            </th>
                            <th className="text-left py-3 px-4 text-sm font-semibold text-muted-foreground uppercase tracking-wider">
                              Predicted Topic
                            </th>
                            <th className="text-right py-3 px-4 text-sm font-semibold text-muted-foreground uppercase tracking-wider">
                              Prediction Confidence
                            </th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.results.map((modelResult, index) => {
                            const isBest = bestModelByConfidence?.model === modelResult.model;
                            return (
                              <tr
                                key={index}
                                className={`border-b border-border/50 last:border-b-0 transition-colors ${
                                  isBest ? "bg-primary/10" : "hover:bg-muted/30"
                                }`}
                              >
                                <td className="py-4 px-4">
                                  <div className="flex items-center gap-2">
                                    {isBest && (
                                      <Trophy className="h-4 w-4 text-primary" />
                                    )}
                                    <span className={`font-medium text-sm ${isBest ? "text-primary" : "text-card-foreground"}`}>
                                      {modelResult.model}
                                    </span>
                                  </div>
                                </td>
                                <td className="py-4 px-4">
                                  <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-accent/20 text-accent-foreground">
                                    {modelResult.prediction}
                                  </span>
                                </td>
                                <td className="py-4 px-4 text-right">
                                  <span className={`font-mono text-sm font-semibold ${
                                    modelResult.prediction_confidence >= 0.7 ? "text-green-600" :
                                    modelResult.prediction_confidence >= 0.5 ? "text-yellow-600" :
                                    "text-red-600"
                                  }`}>
                                    {formatPercentage(modelResult.prediction_confidence)}
                                  </span>
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                    </div>
                  </div>

                  {/* Best Model Highlight - Based on Confidence */}
                  <div className="bg-gradient-to-br from-primary/20 via-primary/15 to-primary/10 backdrop-blur-md border-2 border-primary/40 rounded-2xl p-6 animate-fade-in relative overflow-hidden shadow-xl shadow-primary/20">
                    <div className="absolute top-0 right-0 w-40 h-40 bg-primary/20 rounded-full blur-3xl"></div>
                    <div className="relative flex items-center gap-4">
                      <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary/30 to-primary/20 flex items-center justify-center shadow-lg">
                        <Trophy className="h-8 w-8 text-primary animate-pulse" />
                      </div>
                      <div className="flex-1">
                        <p className="text-sm text-primary font-bold uppercase tracking-wide mb-1 flex items-center gap-1">
                          <Sparkles className="h-3 w-3" />
                          Highest Confidence
                        </p>
                        <p className="text-2xl font-bold text-foreground">{bestModelByConfidence?.model}</p>
                      </div>
                      <div className="text-right bg-primary/20 rounded-xl px-4 py-3 border border-primary/30">
                        <p className="text-xs text-primary font-semibold uppercase tracking-wide mb-1">Score</p>
                        <p className="text-3xl font-black bg-gradient-to-r from-primary to-primary/70 bg-clip-text text-transparent">
                          {bestModelByConfidence && formatPercentage(bestModelByConfidence.prediction_confidence)}
                        </p>
                      </div>
                    </div>
                  </div>
                </>
              )}

              {/* Placeholder when no results */}
              {!result && !error && (
                <div className="bg-gradient-to-br from-card/60 to-card/40 backdrop-blur-sm rounded-2xl border-2 border-dashed border-border/50 p-12 text-center relative overflow-hidden group hover:border-primary/30 transition-all duration-300">
                  <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-accent/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                  <div className="relative">
                    <div className="w-20 h-20 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-muted/40 to-muted/20 flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                      <Brain className="h-10 w-10 text-muted-foreground/40 group-hover:text-primary/60 transition-colors duration-300" />
                    </div>
                    <p className="text-muted-foreground font-medium mb-2">
                      Awaiting Classification
                    </p>
                    <p className="text-sm text-muted-foreground/60">
                      Enter news text and click "Classify News" to see predictions
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="py-6 px-4 border-t border-border bg-card/80 backdrop-blur-sm relative z-10">
        <div className="container mx-auto text-center">
          <div className="flex items-center justify-center gap-2 mb-1">
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

export default ClassifyPage;
