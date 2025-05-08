import React, { useState, Component } from 'react';
import '../styles/stockAnalyzer.css';
import ChatbotPopup from './ChatbotPopup';

// Error Boundary Component
class ErrorBoundary extends Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }

    componentDidCatch(error, errorInfo) {
        console.error('Error caught by boundary:', error, errorInfo);
    }

    render() {
        if (this.state.hasError) {
            return (
                <div className="error-boundary">
                    <h2>Something went wrong.</h2>
                    <p>Please try again or contact support if the problem persists.</p>
                    <button onClick={() => this.setState({ hasError: false, error: null })}>
                        Try Again
                    </button>
                </div>
            );
        }

        return this.props.children;
    }
}

const StockAnalyzer = () => {
    const [ticker, setTicker] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [analysis, setAnalysis] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        
        try {
            const response = await fetch('http://localhost:4000/api/stock/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ ticker: ticker.toUpperCase() }),
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.details || 'Failed to analyze stock');
            }
            
            const data = await response.json();
            console.log('Full response data:', data);
            console.log('Predictions data:', data.predictions);
            console.log('Error value:', data.predictions.error);
            console.log('MAPE value:', data.predictions.mape);
            setAnalysis(data);
        } catch (err) {
            console.error('Error in handleSubmit:', err);
            setError(err.message);
            setAnalysis(null);
        } finally {
            setLoading(false);
        }
    };

    return (
        <ErrorBoundary>
            <div className="stock-analyzer-page">
                <div className="header">
                    <h1>Stock Analyzer</h1>
                </div>
                <div className="analyzer-container">
                    <form onSubmit={handleSubmit} className="analyzer-form">
                        <div className="input-group">
                            <label htmlFor="ticker">Enter Stock Ticker</label>
                            <input
                                type="text"
                                id="ticker"
                                value={ticker}
                                onChange={(e) => setTicker(e.target.value)}
                                placeholder="Enter stock ticker (e.g., AAPL)"
                                className="ticker-input"
                                required
                            />
                        </div>
                        <button type="submit" className="analyze-button" disabled={loading}>
                            {loading ? 'Analyzing...' : 'Analyze'}
                        </button>
                    </form>

                    {error && (
                        <div className="error-message">
                            {error}
                        </div>
                    )}

                    {analysis && (
                        <div className="analysis-results">
                            <h2>Analysis for {analysis.ticker}</h2>
                            
                            <div className="current-price">
                                <h3>Current Price</h3>
                                <p>${analysis.current_price.toFixed(2)}</p>
                            </div>

                            <div className="risk-metrics">
                                <h3>Risk Metrics</h3>
                                <ul>
                                    <li>20-Day Volatility: {(analysis.risk_metrics.Current_Volatility_20d * 100).toFixed(2)}%</li>
                                    <li>60-Day Volatility: {(analysis.risk_metrics.Current_Volatility_60d * 100).toFixed(2)}%</li>
                                    <li>Sharpe Ratio: {analysis.risk_metrics.Sharpe_Ratio.toFixed(2)}</li>
                                    <li>Sortino Ratio: {analysis.risk_metrics.Sortino_Ratio.toFixed(2)}</li>
                                    <li>Value at Risk (95%): {(analysis.risk_metrics.VaR_95 * 100).toFixed(2)}%</li>
                                    <li>Maximum Drawdown: {(analysis.risk_metrics.Max_Drawdown * 100).toFixed(2)}%</li>
                                </ul>
                            </div>

                            <div className="recommendations">
                                <h3>Trading Recommendations</h3>
                                <p>Action: {analysis.recommendations.action}</p>
                                <p>Confidence: {analysis.recommendations.confidence.toFixed(2)}%</p>
                                <h4>Reasoning:</h4>
                                <ul>
                                    {analysis.recommendations.reasoning.map((reason, index) => (
                                        <li key={index}>{reason}</li>
                                    ))}
                                </ul>
                            </div>

                            <div className="model-performance">
                                <h3>Model Performance</h3>
                                <ul>
                                    <li>Prediction Error: ${analysis.predictions.error.toFixed(2)}</li>
                                    <li>Mean Absolute Percentage Error (MAPE): {(analysis.predictions.mape * 100).toFixed(2)}%</li>
                                </ul>
                            </div>
                        </div>
                    )}
                </div>
            </div>
            <ChatbotPopup />
        </ErrorBoundary>
    );
};

export default StockAnalyzer; 