'''
Stock Analyzer Service

A FastAPI service that provides users stock analysis based on the ticker symbol inputted.

Output:
- Current Price
- Risk Metrics
- Trading Recommendations
- Predictions
- Plots

'''
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
from io import BytesIO
import model  # your existing model
import pandas as pd
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StockRequest(BaseModel):
    """
    Stock request data model.
    
    Attributes:
        ticker (str): The stock ticker symbol to analyze
        risk_tolerance (str): User's risk tolerance level (default: "moderate")
        owns_stock (bool): Whether the user already owns the stock (default: False)
    """
    ticker: str
    risk_tolerance: str = "moderate"
    owns_stock: bool = False

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string
    
    Args:
        fig (matplotlib.figure.Figure): The figure to convert
        
    Returns:
        str: Base64 encoded string of the figure image
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

@app.post("/analyze")
async def analyze_stock(request: StockRequest):
    """
    Analyze stock data and provide predictions and recommendations.
    
    Args:
        request (StockRequest): The stock request containing ticker and preferences
        
    Returns:
        dict: Analysis results including risk metrics, recommendations, and predictions
        
    Raises:
        HTTPException: If the stock data cannot be retrieved or analyzed
    """
    try:
        logger.info(f"Received request for ticker: {request.ticker}")
        # Fetch and process data
        logger.info("Fetching daily data...")
        df = model.fetch_daily_data(request.ticker)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for ticker {request.ticker}")

        logger.info("Engineering features...")
        df = model.engineer_features(df)

        logger.info("Calculating risk metrics...")
        df = model.calculate_risk_metrics(df)

        # Get risk summary
        logger.info("Getting risk summary...")
        risk_summary = model.get_risk_summary(df)

        # Generate predictions
        logger.info("Generating predictions...")
        error, mape, actual, predictions = model.validate(df, 0.2)
        future_prices, future_dates = model.predict_future(df)

        # Log prediction values
        logger.info(f"Prediction error: {error}")
        logger.info(f"MAPE: {mape}")

        # Generate recommendations
        logger.info("Generating recommendations...")
        recommendations = model.generate_trading_recommendations(
            df,
            risk_summary,
            predictions,
            request.risk_tolerance
        )

        # Generate and convert plots to base64
        logger.info("Generating plots...")
        try:
            fig = plt.figure(figsize=(15, 10))
            model.plot_risk_metrics(df)
            risk_metrics_plot = fig_to_base64(fig)
            logger.info("Risk metrics plot generated successfully")
        except Exception as e:
            logger.error(f"Error generating plot: {str(e)}")
            risk_metrics_plot = None

        # Create response
        response = {
            "ticker": request.ticker,
            "current_price": float(df['Close'].iloc[-1]),
            "risk_metrics": risk_summary,
            "recommendations": recommendations,
            "predictions": {
                "error": float(error),
                "mape": float(mape),
                "future_prices": [float(x) for x in future_prices],
                "future_dates": [str(x) for x in future_dates]
            },
            "plots": {
                "risk_metrics": risk_metrics_plot
            }
        }
        # Log response values
        logger.info("Response structure:")
        logger.info(f"Has plots: {'plots' in response}")
        logger.info(f"Has risk_metrics plot: {response['plots'] and 'risk_metrics' in response['plots']}")
        logger.info("Analysis completed successfully")
        return response

    except Exception as e:
        logger.error(f"Error analyzing stock: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    """
    Run FastAPI application with uvicron server.
    """
    import uvicorn
    logger.info("Starting stock analyzer service...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
