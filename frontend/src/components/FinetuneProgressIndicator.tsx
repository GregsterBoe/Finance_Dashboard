import { useEffect, useState, useRef } from 'react';

interface FinetuneIterationResult {
  iteration: number;
  config: Record<string, any>;
  score: number | null;
  metric: string;
  is_best: boolean;
}

interface FinetuneResponse {
  status: string;
  ticker: string;
  best_config: Record<string, any>;
  best_score: number | null;
  metric: string;
  all_results: FinetuneIterationResult[];
  total_iterations: number;
  timestamp: string;
}

interface FinetuneProgressIndicatorProps {
  ticker: string;
  startDate: string;
  endDate: string;
  nIterations: number;
  onComplete: (result: FinetuneResponse) => void;
  onError: (error: string) => void;
}

export type { FinetuneResponse, FinetuneIterationResult };

export default function FinetuneProgressIndicator({
  ticker,
  startDate,
  endDate,
  nIterations,
  onComplete,
  onError,
}: FinetuneProgressIndicatorProps) {
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('Initializing...');
  const [isComplete, setIsComplete] = useState(false);
  const [currentIteration, setCurrentIteration] = useState(0);
  const [totalIterations, setTotalIterations] = useState(nIterations);
  const [currentScore, setCurrentScore] = useState<number | null>(null);
  const [bestScore, setBestScore] = useState<number | null>(null);
  const [iterationResults, setIterationResults] = useState<
    Array<{ iteration: number; score: number | null; is_best: boolean; config: Record<string, any> }>
  >([]);

  const abortControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    abortControllerRef.current = controller;

    const startSSE = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/api/finetune-lstm-stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ticker,
            start_date: startDate,
            end_date: endDate,
            n_iterations: nIterations,
          }),
          signal: controller.signal,
        });

        if (!response.ok) throw new Error('Failed to start finetuning');

        const reader = response.body?.getReader();
        const decoder = new TextDecoder();

        while (reader) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            try {
              const data = JSON.parse(line.slice(6));

              setProgress(data.progress);
              setMessage(data.message);

              if (data.iteration) setCurrentIteration(data.iteration);
              if (data.total_iterations) setTotalIterations(data.total_iterations);
              if (data.current_score !== undefined) setCurrentScore(data.current_score);
              if (data.best_score !== undefined) setBestScore(data.best_score);

              // Track results for the live table
              if (data.current_score !== undefined && data.iteration) {
                setIterationResults((prev) => {
                  if (prev.some((r) => r.iteration === data.iteration && r.score !== null)) return prev;
                  return [
                    ...prev,
                    {
                      iteration: data.iteration,
                      score: data.current_score,
                      is_best: data.best_score === data.current_score,
                      config: data.current_config || data.best_config || {},
                    },
                  ];
                });
              }

              if (data.completed) {
                setIsComplete(true);
                onComplete(data.result);
              }

              if (data.error) {
                onError(data.detail || data.message);
              }
            } catch {
              // skip malformed lines
            }
          }
        }
      } catch (err) {
        if ((err as Error).name === 'AbortError') return;
        onError(err instanceof Error ? err.message : 'Stream failed');
      }
    };

    startSSE();

    return () => {
      controller.abort();
    };
  }, [ticker, startDate, endDate, nIterations, onComplete, onError]);

  const sortedResults = [...iterationResults].sort(
    (a, b) => (b.score ?? -Infinity) - (a.score ?? -Infinity)
  );

  return (
    <div className="w-full max-w-3xl mx-auto p-6 bg-white rounded-lg shadow-md space-y-5">
      {/* Header */}
      <div className="flex justify-between items-center">
        <span className="text-sm font-medium text-gray-700">{message}</span>
        <span className="text-sm font-bold text-yellow-600">{Math.round(progress)}%</span>
      </div>

      {/* Progress bar */}
      <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden">
        <div
          className={`h-full transition-all duration-300 ease-out ${
            isComplete ? 'bg-green-500' : 'bg-yellow-500'
          }`}
          style={{ width: `${progress}%` }}
        >
          <div className="h-full w-full animate-pulse bg-gradient-to-r from-transparent via-white/30 to-transparent" />
        </div>
      </div>

      {/* Scores summary */}
      {(currentIteration > 0) && (
        <div className="grid grid-cols-3 gap-3 text-center text-sm">
          <div className="bg-gray-50 rounded p-2">
            <div className="text-gray-500">Iteration</div>
            <div className="font-semibold">{currentIteration} / {totalIterations}</div>
          </div>
          <div className="bg-gray-50 rounded p-2">
            <div className="text-gray-500">Current Score</div>
            <div className="font-semibold">
              {currentScore !== null ? currentScore.toFixed(4) : '—'}
            </div>
          </div>
          <div className="bg-yellow-50 rounded p-2 border border-yellow-200">
            <div className="text-yellow-700">Best Score</div>
            <div className="font-semibold text-yellow-800">
              {bestScore !== null ? bestScore.toFixed(4) : '—'}
            </div>
          </div>
        </div>
      )}

      {/* Live results table */}
      {sortedResults.length > 0 && (
        <div className="max-h-48 overflow-y-auto border rounded">
          <table className="w-full text-xs">
            <thead className="bg-gray-100 sticky top-0">
              <tr>
                <th className="p-1.5 text-left">#</th>
                <th className="p-1.5 text-left">Score</th>
                <th className="p-1.5 text-left">Hidden</th>
                <th className="p-1.5 text-left">Layers</th>
                <th className="p-1.5 text-left">Seq Len</th>
                <th className="p-1.5 text-left">LR</th>
                <th className="p-1.5 text-left">Bidir</th>
              </tr>
            </thead>
            <tbody>
              {sortedResults.map((r) => (
                <tr
                  key={r.iteration}
                  className={r.is_best ? 'bg-yellow-50 font-medium' : 'hover:bg-gray-50'}
                >
                  <td className="p-1.5">{r.iteration}</td>
                  <td className="p-1.5">{r.score !== null ? r.score.toFixed(4) : 'failed'}</td>
                  <td className="p-1.5">{r.config.hidden_size ?? '—'}</td>
                  <td className="p-1.5">{r.config.num_layers ?? '—'}</td>
                  <td className="p-1.5">{r.config.sequence_length ?? '—'}</td>
                  <td className="p-1.5">{r.config.learning_rate != null ? Number(r.config.learning_rate).toFixed(5) : '—'}</td>
                  <td className="p-1.5">{r.config.bidirectional != null ? (r.config.bidirectional ? 'Y' : 'N') : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Status indicator */}
      <div className="flex items-center justify-center">
        {!isComplete ? (
          <div className="flex items-center space-x-2">
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-yellow-600" />
            <span className="text-sm text-gray-600">Searching...</span>
          </div>
        ) : (
          <div className="flex items-center space-x-2 text-green-600">
            <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                clipRule="evenodd"
              />
            </svg>
            <span className="text-sm font-medium">Finetuning Complete!</span>
          </div>
        )}
      </div>
    </div>
  );
}
