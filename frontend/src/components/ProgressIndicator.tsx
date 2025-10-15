// frontend/src/components/ProgressIndicator.tsx
import { useEffect, useState } from 'react';

interface ProgressIndicatorProps {
  onComplete: (result: any) => void;
  onError: (error: string) => void;
  config: any; // BacktestConfig
}

export default function ProgressIndicator({ onComplete, onError, config }: ProgressIndicatorProps) {
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState("Initializing...");
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    // Create EventSource for SSE
    const eventSource = new EventSource(
      `http://127.0.0.1:8000/api/backtest-model-stream?${new URLSearchParams({
        ...config,
        model_spec: JSON.stringify(config.model_spec)
      })}`
    );

    // Alternative: Use POST with fetch API for EventSource
    const startSSE = async () => {
      try {
        const response = await fetch("http://127.0.0.1:8000/api/backtest-model-stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(config),
        });

        if (!response.ok) throw new Error("Failed to start backtesting");

        const reader = response.body?.getReader();
        const decoder = new TextDecoder();

        while (true && reader) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = JSON.parse(line.slice(6));
              
              setProgress(data.progress);
              setMessage(data.message);

              // Handle completion
              if (data.completed) {
                setIsComplete(true);
                onComplete(data.result);
              }

              // Handle error
              if (data.error) {
                onError(data.detail || data.message);
              }
            }
          }
        }
      } catch (err) {
        onError(err instanceof Error ? err.message : "Stream failed");
      }
    };

    startSSE();

    return () => {
      // Cleanup if needed
    };
  }, [config, onComplete, onError]);

  return (
    <div className="w-full max-w-2xl mx-auto p-6 bg-white rounded-lg shadow-md">
      <div className="mb-4">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-gray-700">{message}</span>
          <span className="text-sm font-bold text-purple-600">{Math.round(progress)}%</span>
        </div>
        
        {/* Progress bar */}
        <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden">
          <div
            className={`h-full transition-all duration-300 ease-out ${
              isComplete ? 'bg-green-500' : 'bg-purple-600'
            }`}
            style={{ width: `${progress}%` }}
          >
            {/* Animated shimmer effect */}
            <div className="h-full w-full animate-pulse bg-gradient-to-r from-transparent via-white/30 to-transparent" />
          </div>
        </div>
      </div>

      {/* Status indicator */}
      <div className="flex items-center justify-center mt-4">
        {!isComplete ? (
          <div className="flex items-center space-x-2">
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-purple-600" />
            <span className="text-sm text-gray-600">Processing...</span>
          </div>
        ) : (
          <div className="flex items-center space-x-2 text-green-600">
            <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
            <span className="text-sm font-medium">Complete!</span>
          </div>
        )}
      </div>
    </div>
  );
}