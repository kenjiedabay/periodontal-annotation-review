interface ErrorStateProps {
  message: string;
  onDismiss?: () => void;
  onRetry?: () => void;
}

export default function ErrorState({ message, onDismiss, onRetry }: ErrorStateProps) {
  return (
    <div className="error-state" role="alert">
      <div>
        <strong>Something went wrong</strong>
        <p>{message}</p>
      </div>
      <div className="error-actions">
        {onRetry && <button type="button" className="secondary-btn" onClick={onRetry}>Try again</button>}
        {onDismiss && <button type="button" className="text-btn" onClick={onDismiss}>Dismiss</button>}
      </div>
    </div>
  );
}
