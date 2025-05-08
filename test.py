import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import yfinance as yf
from finta import TA    
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error
import time



def fetch_daily_data(symbol):
    '''
    Fetches daily stock data with retry mechanism and error handling
    @param symbol: stock symbol to fetch data for
    @return: pandas DataFrame containing stock data
    '''
    try:
        print(f"Fetching data for {symbol}...")
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                # Create ticker object
                ticker = yf.Ticker(symbol)
                
                # Get data for past 2.5 years
                end_date = datetime.now()
                start_date = end_date - timedelta(days=913)  # ~2.5 years
                
                # Fetch data
                df = ticker.history(start=start_date, end=end_date)
                
                # Verify we have data
                if df is None or df.empty:
                    if attempt < max_retries - 1:
                        print(f"Attempt {attempt + 1}: No data received. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"Error: Unable to fetch data for {symbol} after {max_retries} attempts")
                        return None
                
                # Verify we have enough data points
                min_required_points = 100  # Adjust as needed
                if len(df) < min_required_points:
                    print(f"Error: Insufficient data points for {symbol}. Got {len(df)}, need at least {min_required_points}")
                    return None
                
                # Verify we have all required columns
                required_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
                missing_columns = required_columns - set(df.columns)
                if missing_columns:
                    print(f"Error: Missing required columns: {missing_columns}")
                    return None
                
                # Check for missing values
                if df.isnull().any().any():
                    print("Warning: Data contains missing values. Filling with forward fill method...")
                    df = df.ffill()  # Forward fill missing values
                
                print(f"Successfully fetched {len(df)} data points for {symbol}")
                return df
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Error: Failed to fetch data for {symbol} after {max_retries} attempts: {str(e)}")
                    return None
                    
    except Exception as e:
        print(f"Unexpected error in fetch_daily_data: {str(e)}")
        return None

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Optimized feature engineering with essential technical indicators
    @param df: pandas DF containing stock data
    @return: pandas DF containing engineered features
    '''
    # Use vectorized operations for basic indicators
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    
    # Calculate essential indicators
    df['RSI'] = TA.RSI(df)
    df['MACD'] = TA.MACD(df)['MACD']
    df['MACD_SIGNAL'] = TA.MACD(df)['SIGNAL']
    
    # Drop rows with NaN values
    df = df.iloc[200:, :]
    df['Target'] = df.Close.shift(-1)
    df.dropna(inplace=True)
    
    return df

def train_test_split(data: pd.DataFrame, perc: float) -> tuple[np.ndarray, np.ndarray]:
    '''
    Splits data into train and test sets
    @param data: pandas DF containing stock data
    @param perc: percentage of data for test set
    @return: train set, test set
    '''
    ret = data.values
    n = int(len(data) * (1-perc))
    return ret[:n], ret[n:]

def xgb_predict(train: pd.DataFrame, val: pd.DataFrame, model=None) -> tuple[float, XGBRegressor]:
    '''
    Optimized prediction function that reuses the trained model
    @param train: pandas DF containing training data
    @param val: pandas DF as a validation set
    @param model: Optional pre-trained model
    @return: tuple of (prediction, model)
    '''
    if model is None:
        train = np.array(train)
        val = np.array(val)
        
        X = train[:,:-1]
        y = train[:,-1]
        
        model = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=500,
            learning_rate=0.1,
            colsample_bytree=0.7,
            max_depth=3,
            gamma=1,
            n_jobs=-1
        )
        model.fit(X, y)
    
    val = val.reshape(1, -1)
    pred = model.predict(val)
    return pred[0], model

def mape(actual, pred) -> float:
    ''' 
    Calculates the mean absolute percentage error between the actual and predicted values 
    @param actual: array containing actual values 
    @param pred: array containing predicted values
    @return: float mape
    '''
    actual = np.array(actual)
    pred = np.array(pred)
    mape = np.mean(np.abs((actual - pred) / actual) * 100)
    return mape 

def validate(data, perc):
    '''
    Optimized validation function that reuses the trained model
    @param data: df containing stock data
    @param perc: percentage of data for test set
    @return: tuple of (error, MAPE, actual values, predictions)
    '''
    try:
        # Check if we have enough data
        if data is None or data.empty or len(data) < 50:  # Minimum required data points
            print("Not enough data for validation (minimum 50 data points required)")
            return 0.0, 0.0, [], []
            
        predictions = []
        train, test = train_test_split(data, perc)
        
        # Check if test set is empty
        if len(test) == 0:
            print("Test set is empty")
            return 0.0, 0.0, [], []
            
        history = train.copy()
        model = None
        
        try:
            for i in range(len(test)):
                X_test = test[i,:-1].reshape(1, -1)
                y_test = test[i,-1]
                
                pred, model = xgb_predict(history, X_test, model)
                predictions.append(pred)
                
                history = np.vstack([history, test[i]])
            
            # Convert to numpy arrays
            predictions = np.array(predictions)
            actual = test[:,-1]
            
            # Check if we have valid data
            if len(actual) == 0 or len(predictions) == 0:
                print("No valid predictions or actual values")
                return 0.0, 0.0, [], []
            
            # Calculate error metrics
            error = root_mean_squared_error(actual, predictions)
            MAPE = mape(actual, predictions)
            
            return error, MAPE, actual, predictions
            
        except Exception as e:
            print(f"Error in validation loop: {str(e)}")
            return 0.0, 0.0, [], []
            
    except Exception as e:
        print(f"Error in validate function: {str(e)}")
        return 0.0, 0.0, [], []

def calculate_risk_metrics(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Calculates various risk metrics for the stock
    @param df: DataFrame with price data and features
    @return: DataFrame with added risk metrics
    '''
    # Calculating daily returns
    df['Daily_Returns'] = df['Close'].pct_change()
    
    # Volatility calculations (20 and 60 day)
    df['Volatility_20d'] = df['Daily_Returns'].rolling(window=20).std() * np.sqrt(252)  # Annualized
    df['Volatility_60d'] = df['Daily_Returns'].rolling(window=60).std() * np.sqrt(252)
    
    # Calculate Sharpe Ratio (assuming risk-free rate of 0.02)
    risk_free_rate = 0.02
    excess_returns = df['Daily_Returns'] - risk_free_rate/252
    df['Sharpe_Ratio'] = (excess_returns.mean() * 252) / (df['Daily_Returns'].std() * np.sqrt(252))
    
    # Calculate Sortino Ratio (downside deviation only)
    negative_returns = df['Daily_Returns'].copy()
    negative_returns[negative_returns > 0] = 0
    downside_std = negative_returns.std() * np.sqrt(252)
    df['Sortino_Ratio'] = (excess_returns.mean() * 252) / downside_std if downside_std != 0 else 0
    
    # Maximum Drawdown
    rolling_max = df['Close'].rolling(window=252, min_periods=1).max()
    daily_drawdown = df['Close']/rolling_max - 1.0
    df['Max_Drawdown'] = daily_drawdown.rolling(window=252, min_periods=1).min()
    
    # Value at Risk (VaR) - 95% confidence
    df['VaR_95'] = df['Daily_Returns'].quantile(0.05)
    
    
    return df

def get_risk_summary(df: pd.DataFrame) -> dict:
    '''
    Creates a summary of current risk metrics
    @param df: DataFrame with risk metrics
    @return: Dictionary with current risk metrics
    '''
    return {
        'Current_Volatility_20d': df['Volatility_20d'].iloc[-1],
        'Current_Volatility_60d': df['Volatility_60d'].iloc[-1],
        'Sharpe_Ratio': df['Sharpe_Ratio'].iloc[-1],
        'Sortino_Ratio': df['Sortino_Ratio'].iloc[-1],
        'VaR_95': df['VaR_95'].iloc[-1],
        'Max_Drawdown': df['Max_Drawdown'].iloc[-1],
    }

def plot_risk_metrics(df: pd.DataFrame):
    '''
    Creates visualizations for risk metrics
    @param df: DataFrame with risk metrics
    '''
    # Convert numpy array to DataFrame if needed
    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df, columns=['Volatility_20d', 'Volatility_60d', 'Max_Drawdown'])
    
    fig, (ax1) = plt.subplots( 1,1, figsize=(15, 10))
    
    # Plot volatility
    ax1.plot(df.index, df['Volatility_20d'], label='20-day Volatility')
    ax1.plot(df.index, df['Volatility_60d'], label='60-day Volatility')
    ax1.set_title('Historical Volatility')
    ax1.legend()
    
    plt.tight_layout()

def generate_trading_recommendations(df: pd.DataFrame, risk_summary: dict, predictions: np.ndarray, 
                                  risk_tolerance: str = 'moderate', owns_stock: bool = False) -> dict:
    '''
    Generates trading recommendations based on technical indicators, risk metrics, and predictions
    @param df: DataFrame with all metrics and indicators
    @param risk_summary: Dictionary containing current risk metrics
    @param predictions: Numpy array of predictions
    @param risk_tolerance: String indicating risk tolerance ('conservative', 'moderate', 'aggressive')
    @param owns_stock: Boolean indicating if the stock is currently owned
    @return: Dictionary containing recommendations and reasoning
    '''
    # Get latest data
    latest = df.iloc[-1]
    
    # Define risk tolerance thresholds with balanced values
    risk_thresholds = {
        'conservative': {
            'max_volatility': 0.35,
            'min_sharpe': 0.5,
            'max_var': 0.04,
            'min_return': 0.03,
            'stop_loss': -0.08
        },
        'moderate': {
            'max_volatility': 0.50,
            'min_sharpe': 0.2,
            'max_var': 0.06,
            'min_return': 0.02,
            'stop_loss': -0.12
        },
        'aggressive': {
            'max_volatility': 0.70,
            'min_sharpe': -0.2,
            'max_var': 0.08,
            'min_return': 0.01,
            'stop_loss': -0.15
        }
    }
    
    # Get thresholds for given risk tolerance
    thresholds = risk_thresholds[risk_tolerance]
    
    # Initialize recommendation
    recommendation = {
        'action': None,
        'confidence': 0,
        'reasoning': [],
        'risk_assessment': [],
        'technical_signals': [],
        'prediction_signals': [],
        'position_analysis': []
    }
    
    # 1. Risk Assessment with balanced scoring
    risk_score = 0
    
    # Volatility scoring with trend consideration
    vol_trend = df['Volatility_20d'].iloc[-5:].mean() - df['Volatility_20d'].iloc[-10:-5].mean()
    if risk_summary['Current_Volatility_20d'] < thresholds['max_volatility']:
        risk_score += 1
        recommendation['risk_assessment'].append("Volatility within acceptable range")
    else:
        if vol_trend < 0:  # Volatility is decreasing
            risk_score -= 0.3
            recommendation['risk_assessment'].append("Elevated but decreasing volatility")
        else:
            risk_score -= 0.5
            recommendation['risk_assessment'].append("Elevated volatility - monitor closely")
    
    # Sharpe Ratio with trend consideration
    sharpe_trend = df['Sharpe_Ratio'].iloc[-5:].mean() - df['Sharpe_Ratio'].iloc[-10:-5].mean()
    if risk_summary['Sharpe_Ratio'] > thresholds['min_sharpe']:
        risk_score += 1
        recommendation['risk_assessment'].append("Good risk-adjusted returns")
    else:
        if sharpe_trend > 0:  # Sharpe ratio is improving
            risk_score -= 0.3
            recommendation['risk_assessment'].append("Risk-adjusted returns improving")
        else:
            risk_score -= 0.5
            recommendation['risk_assessment'].append("Below target risk-adjusted returns")
    
    # VaR with market context
    if abs(risk_summary['VaR_95']) < thresholds['max_var']:
        risk_score += 1
        recommendation['risk_assessment'].append("Value at Risk within acceptable range")
    else:
        if abs(risk_summary['VaR_95']) < abs(df['VaR_95'].mean()):  # Better than average
            risk_score -= 0.3
            recommendation['risk_assessment'].append("VaR elevated but better than average")
        else:
            risk_score -= 0.5
            recommendation['risk_assessment'].append("VaR above acceptable range")
    
    # 2. Enhanced Technical Analysis
    tech_score = 0
    
    # RSI with trend consideration
    rsi_trend = df['RSI'].iloc[-5:].mean() - df['RSI'].iloc[-10:-5].mean()
    if latest['RSI'] < 30:
        tech_score += 1.5 if rsi_trend > 0 else 1  # Extra weight if RSI is improving
        recommendation['technical_signals'].append("Oversold condition (RSI)")
    elif latest['RSI'] > 70:
        tech_score -= 1.5 if rsi_trend < 0 else 1  # Extra weight if RSI is decreasing
        recommendation['technical_signals'].append("Overbought condition (RSI)")
    elif 40 <= latest['RSI'] <= 60:  # Neutral zone
        tech_score += 0.5
        recommendation['technical_signals'].append("RSI in neutral zone")
    
    # MACD with trend strength
    macd_trend = df['MACD'].iloc[-5:].mean() - df['MACD'].iloc[-10:-5].mean()
    if latest['MACD'] > latest['MACD_SIGNAL']:
        tech_score += 1.5 if macd_trend > 0 else 1  # Extra weight if trend is strengthening
        recommendation['technical_signals'].append("Positive MACD crossover")
    else:
        tech_score -= 1.5 if macd_trend < 0 else 1
        recommendation['technical_signals'].append("Negative MACD crossover")
    
    # Moving Average analysis with multiple timeframes
    ma_score = 0
    if latest['Close'] > latest['SMA200']:
        ma_score += 1
        recommendation['technical_signals'].append("Price above 200-day SMA")
    if latest['Close'] > latest['SMA50']:
        ma_score += 0.5
        recommendation['technical_signals'].append("Price above 50-day SMA")
    tech_score += ma_score
    
    # 3. Enhanced Prediction Analysis
    pred_score = 0
    current_price = latest['Close']
    
    # Calculate average predicted return with trend consideration
    avg_prediction = np.mean(predictions[-5:])
    predicted_return = (avg_prediction - current_price) / current_price
    
    # Consider prediction stability
    pred_volatility = np.std(predictions[-5:]) / np.mean(predictions[-5:])
    pred_trend = predictions[-1] - predictions[-5]
    
    if predicted_return > thresholds['min_return']:
        pred_score += 1
        if pred_volatility < 0.02:  # Low volatility in predictions
            pred_score += 0.5
            recommendation['prediction_signals'].append(
                f"Stable positive prediction: {predicted_return:.1%} potential return"
            )
        else:
            recommendation['prediction_signals'].append(
                f"Short-term prediction shows {predicted_return:.1%} potential return"
            )
    
    # Trend analysis with momentum
    if predictions[-1] > predictions[-2]:
        pred_score += 1
        if pred_trend > 0:  # Stronger upward trend
            pred_score += 0.5
            recommendation['prediction_signals'].append("Strong upward prediction trend")
        else:
            recommendation['prediction_signals'].append("Upward prediction trend")
    else:
        pred_score -= 1
        if pred_trend < 0:  # Stronger downward trend
            pred_score -= 0.5
            recommendation['prediction_signals'].append("Strong downward prediction trend")
        else:
            recommendation['prediction_signals'].append("Downward prediction trend")
    
    # 4. Position Analysis with market context
    if owns_stock:
        # Check stop loss with trend consideration
        recent_return = (current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]
        if recent_return < thresholds['stop_loss']:
            if recent_return > df['Daily_Returns'].mean():  # Better than average daily return
                recommendation['position_analysis'].append(
                    f"Stop loss triggered but better than average: {recent_return:.1%}"
                )
                pred_score -= 1.5
            else:
                recommendation['position_analysis'].append(
                    f"Stop loss triggered: {recent_return:.1%}"
                )
                pred_score -= 2
        
        # Check if holding is still profitable with trend context
        if predicted_return < 0:
            if pred_trend > 0:  # Improving trend
                recommendation['position_analysis'].append(
                    "Negative return predicted but trend improving"
                )
                pred_score -= 0.5
            else:
                recommendation['position_analysis'].append(
                    "Negative future return predicted, consider taking profits"
                )
                pred_score -= 1
        else:
            recommendation['position_analysis'].append(
                "Positive future return predicted, consider holding position"
            )
    else:
        # Enhanced entry analysis
        if predicted_return > thresholds['min_return'] and tech_score > 0:
            if pred_volatility < 0.02 and macd_trend > 0:  # Stable predictions and strong trend
                recommendation['position_analysis'].append(
                    "Strong entry point: stable predictions and positive technical signals"
                )
            else:
                recommendation['position_analysis'].append(
                    "Good entry point: positive technical signals and predicted returns"
                )
        else:
            recommendation['position_analysis'].append(
                "Wait for better entry point"
            )
    
    # 5. Generate Final Recommendation with balanced weighting
    total_score = (risk_score * 1.2) + (tech_score * 1.0) + (pred_score * 0.8)  # Adjusted weights
    max_possible_score = (3 * 1.2) + (3 * 1.0) + (2 * 0.8)  # Adjusted max score
    confidence = (total_score / max_possible_score) * 100
    
    # Modified action logic with trend consideration
    if owns_stock:
        if confidence < -30 and pred_trend < 0:  # Need both negative confidence and trend
            action = "SELL"
            recommendation['reasoning'].append("Negative indicators and trend suggest exiting position")
        else:
            action = "HOLD"
            recommendation['reasoning'].append("Current indicators support maintaining position")
    else:
        if confidence >= 0 and pred_trend > 0:  # Need both positive confidence and trend
            action = "BUY"
            recommendation['reasoning'].append("Positive indicators and trend suggest entering position")
        else:
            action = "NO ACTION"
            recommendation['reasoning'].append("Current indicators suggest waiting for better entry")
    
    recommendation['action'] = action
    recommendation['confidence'] = confidence
    
    return recommendation

def print_recommendation(recommendation: dict):
    '''
    Prints the trading recommendation in a formatted way
    @param recommendation: Dictionary containing recommendation details
    '''
    print("\n" + "="*80)
    print("STOCK TRADING RECOMMENDATION")
    print("="*80)
    
    print(f"\nRECOMMENDATION: {recommendation['action']}")
    print(f"Confidence Level: {recommendation['confidence']:.1f}%")
    
    # Simple confidence explanation
    if recommendation['confidence'] >= 0:
        print("→ This is a positive signal, suggesting potential for growth")
    else:
        print("→ This is a cautious signal, suggesting waiting for better conditions")
    
    print("\n" + "-"*80)
    print("MARKET ANALYSIS")
    print("-"*80)
    
    print("\nTechnical Indicators (What the charts are telling us):")
    # Dictionary of technical signal explanations in simple terms
    signal_explanations = {
        "Oversold condition (RSI)": "The stock price has dropped significantly and might be a good time to buy",
        "Overbought condition (RSI)": "The stock price has risen significantly and might be due for a pullback",
        "RSI in neutral zone": "The stock is trading in a balanced range, neither overbought nor oversold",
        "Positive MACD crossover": "The stock is showing signs of upward momentum and potential growth",
        "Negative MACD crossover": "The stock is showing signs of downward momentum and potential decline",
        "Price above 200-day SMA": "The stock is in a long-term uptrend, which is generally positive",
        "Price above 50-day SMA": "The stock is showing positive recent performance"
    }
    
    for item in recommendation['technical_signals']:
        print(f"\n• {item}")
        if item in signal_explanations:
            print(f"  → {signal_explanations[item]}")
    
    print("\n" + "-"*80)
    print("PREDICTION OUTLOOK")
    print("-"*80)
    
    print("\nWhat to Expect:")
    for item in recommendation['prediction_signals']:
        print(f"• {item}")
    
    print("\n" + "-"*80)
    print("RISK ASSESSMENT")
    print("-"*80)
    
    print("\nCurrent Market Conditions:")
    for item in recommendation['risk_assessment']:
        print(f"• {item}")
    
    print("\nRisk Metrics (In Simple Terms):")
    risk_explanations = {
        "Volatility": "How much the stock price moves up and down - higher means more risk",
        "Sharpe Ratio": "How good the returns are compared to the risk taken - higher is better",
        "Sortino Ratio": "How good the returns are compared to the bad days - higher is better",
        "VaR (Value at Risk)": "The worst-case scenario for potential losses",
        "Maximum Drawdown": "The biggest price drop the stock has experienced"
    }
    
    for metric, explanation in risk_explanations.items():
        print(f"• {metric}: {explanation}")
    
    print("\n" + "-"*80)
    print("POSITION ADVICE")
    print("-"*80)
    
    print("\nWhat This Means For You:")
    for item in recommendation['position_analysis']:
        print(f"• {item}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Create a simple summary based on the action and confidence
    if recommendation['action'] == "BUY":
        print("\nThe overall analysis suggests this might be a good time to consider buying:")
        print("• Technical indicators are showing positive signals")
        print("• Risk levels are within acceptable ranges")
        print("• Future predictions are optimistic")
    elif recommendation['action'] == "SELL":
        print("\nThe overall analysis suggests this might be a good time to consider selling:")
        print("• Technical indicators are showing negative signals")
        print("• Risk levels are elevated")
        print("• Future predictions are concerning")
    elif recommendation['action'] == "HOLD":
        print("\nThe overall analysis suggests holding your current position:")
        print("• Current indicators are mixed but not strongly negative")
        print("• It might be better to wait for clearer signals")
    else:  # NO ACTION
        print("\nThe overall analysis suggests waiting for better conditions:")
        print("• Current market conditions are unclear")
        print("• It's better to wait for more positive signals")
    
    print("="*80 + "\n")

def predict_future(df: pd.DataFrame, days: int = 10) -> tuple[list, list]:
    '''
    Optimized future predictions with batched processing
    @param df: DataFrame containing historical data and features
    @param days: Number of days to predict into the future
    @return: Tuple of (predictions list, dates list)
    '''
    predictions = []
    dates = []
    current_data = df.iloc[-1:].copy()
    last_date = df.index[-1]
    
    # Train model once
    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        learning_rate=0.1,
        colsample_bytree=0.7,
        max_depth=3,
        gamma=1,
        n_jobs=-1
    )
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    model.fit(X, y)
    
    # Process predictions in batches
    for i in range(0, days, 5):
        batch_days = min(5, days - i)
        batch_predictions = []
        
        for j in range(batch_days):
            next_date = last_date + timedelta(days=i+j+1)
            dates.append(next_date)
            
            features = current_data.iloc[:, :-1]
            pred = model.predict(features)[0]
            batch_predictions.append(pred)
            
            # Update current data
            current_data['Close'] = pred
            current_data['Target'] = pred
            
            # Update only essential indicators
            current_data['SMA200'] = (df['Close'].iloc[-199:].sum() + pred) / 200
            current_data['SMA50'] = (df['Close'].iloc[-49:].sum() + pred) / 50
            current_data['EMA20'] = pred * 0.1 + current_data['EMA20'].iloc[-1] * 0.9
        
        predictions.extend(batch_predictions)
    
    return predictions, dates

def display_future_predictions(predictions: list, dates: list, current_price: float):
    '''
    Displays detailed future predictions in the terminal
    @param predictions: List of predicted prices
    @param dates: List of dates for predictions
    @param current_price: Current stock price
    '''
    print("\n" + "="*80)
    print("10-DAY PRICE PREDICTION FORECAST")
    print("="*80)
    
    # Print header
    print(f"\nCurrent Price: ${current_price:.2f}")
    print("\nDaily Predictions:")
    print("-"*80)
    print(f"{'Day':<8} {'Predicted Price':<15} {'Change':<10} {'% Change':<10}")
    print("-"*80)
    
    # Print daily predictions
    for i, (date, price) in enumerate(zip(dates, predictions)):
        change = price - current_price
        pct_change = (change / current_price) * 100
        print(f"Day {i+1:<3} ${price:<14.2f} {change:>+8.2f} {pct_change:>+8.2f}%")
    
    # Print trend analysis
    print("\nTrend Analysis:")
    print("-"*80)
    overall_change = predictions[-1] - current_price
    overall_pct = (overall_change / current_price) * 100
    
    if overall_change > 0:
        trend = "BULLISH"
    elif overall_change < 0:
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"
    
    print(f"Overall Trend: {trend}")
    print(f"10-Day Price Target: ${predictions[-1]:.2f}")
    print(f"Expected Change: {overall_change:+.2f} ({overall_pct:+.2f}%)")
    
    # Calculate volatility of predictions
    pred_volatility = np.std(predictions) / np.mean(predictions) * 100
    print(f"Predicted Volatility: {pred_volatility:.2f}%")
    
    print("="*80 + "\n")

def main():
    try:
        print("\nFetching stock data...")
        # Fetch data with reduced time range
        df = fetch_daily_data('AAPL')
        
        # Check if we have valid data
        if df is None or df.empty:
            print("Error: No data available for analysis. Please check your internet connection or try again later.")
            return
            
        print("Processing data...")
        df.columns = df.columns.droplevel(1)
        
        # Convert to float32 for memory efficiency
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        
        # Engineer features with optimized indicators
        print("Engineering features...")
        df = engineer_features(df)
        
        # Check if we have enough data after feature engineering
        if df is None or df.empty or len(df) < 50:
            print("Error: Not enough data points after feature engineering (minimum 50 required)")
            return
        
        print("Training model...")
        # Split data
        train, test = train_test_split(df, 0.7)
        
        if len(train) == 0 or len(test) == 0:
            print("Error: Invalid train/test split")
            return
            
        try:
            # Initialize model once
            model = XGBRegressor(
                objective='reg:squarederror',
                n_estimators=500,
                learning_rate=0.1,
                colsample_bytree=0.7,
                max_depth=3,
                gamma=1,
                n_jobs=-1
            )
            
            # Train model once
            X_train = train[:,:-1]
            y_train = train[:,-1]
            
            if len(X_train) == 0 or len(y_train) == 0:
                print("Error: Empty training data")
                return
                
            model.fit(X_train, y_train)
            
            print("Validating model...")
            # Validate and get predictions
            rmse, MAPE, y, pred = validate(df, 0.7)
            if len(pred) == 0:
                print("Error: No predictions generated")
                return
                
            pred = np.array(pred, dtype='float32')
            test_pred = np.c_[test, pred]
            
            # Create visualization DataFrame efficiently
            columns = list(df.columns) + ['Pred']
            df_TP = pd.DataFrame(test_pred, columns=columns[:test_pred.shape[1]])
            
            # Prepare data for plotting
            df = df.reset_index(names='Date')
            df_dates = df[['Date', 'Target', 'Close']]
            df_TP = pd.merge(df_TP, df_dates, on='Target', how='left')
            df_TP = df_TP.sort_values(by='Date').reset_index(drop=True)
            
            print("Generating future predictions...")
            # Get future predictions (reduced to 10 days)
            df_indexed = df.set_index('Date')
            future_predictions, future_dates = predict_future(df_indexed, days=10)
            
            if len(future_predictions) == 0:
                print("Error: No future predictions generated")
                return
                
            # Display predictions
            current_price = df['Close'].iloc[-1]
            display_future_predictions(future_predictions, future_dates, current_price)
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
            
            # Plot 1: Historical Prices
            ax1.plot(df['Date'], df['Close'], label='Historical Close Price', color='blue')
            ax1.set_title('AAPL Historical Prices', fontsize=18)
            ax1.set_xlabel('Date', fontsize=14)
            ax1.set_ylabel('Price USD', fontsize=14)
            ax1.legend(loc='upper left')
            ax1.grid(True)
            
            # Plot 2: Future Predictions
            days_into_future = range(1, len(future_predictions) + 1)
            ax2.plot(days_into_future, future_predictions, label='Predicted Price', color='red', marker='o')
            ax2.set_title('AAPL 10-Day Price Predictions', fontsize=18)
            ax2.set_xlabel('Days into Future', fontsize=14)
            ax2.set_ylabel('Predicted Price (USD)', fontsize=14)
            ax2.legend(loc='upper left')
            ax2.grid(True)
            
            # Add horizontal line for current price
            ax2.axhline(y=current_price, color='blue', linestyle='--', label=f'Current Price (${current_price:.2f})')
            
            # Add price annotations
            ax2.annotate(f'${future_predictions[0]:.2f}', 
                        xy=(1, future_predictions[0]),
                        xytext=(1, future_predictions[0] + 2),
                        ha='center')
            ax2.annotate(f'${future_predictions[-1]:.2f}', 
                        xy=(10, future_predictions[-1]),
                        xytext=(10, future_predictions[-1] - 2),
                        ha='center')
            
            plt.tight_layout()
            
            # Print performance metrics
            print(f"\nModel Performance Metrics:")
            print(f"RMSE: ${rmse:.2f}")
            print(f"MAPE: {MAPE:.2f}%")
            
            # Calculate risk metrics
            print("\nCalculating risk metrics...")
            df = calculate_risk_metrics(df)
            risk_summary = get_risk_summary(df)
            
            # Print risk metrics
            print("\nRisk Assessment Metrics:")
            print(f"20-day Volatility: {risk_summary['Current_Volatility_20d']:.2%}")
            print(f"60-day Volatility: {risk_summary['Current_Volatility_60d']:.2%}")
            print(f"Sharpe Ratio: {risk_summary['Sharpe_Ratio']:.2f}")
            print(f"Sortino Ratio: {risk_summary['Sortino_Ratio']:.2f}")
            print(f"95% VaR: {risk_summary['VaR_95']:.2%}")
            print(f"Maximum Drawdown: {risk_summary['Max_Drawdown']:.2%}")
            
            # Generate and print recommendations
            print("\nGenerating trading recommendations...")
            recommendation = generate_trading_recommendations(
                df, 
                risk_summary, 
                pred,
                risk_tolerance='moderate',
                owns_stock=False
            )
            print_recommendation(recommendation)
            
            plt.show()
            
        except Exception as e:
            print(f"Error in model training/prediction: {str(e)}")
            
    except Exception as e:
        print(f"Error in main function: {str(e)}")
    finally:
        # Clear memory
        try:
            del df, df_TP, train, test, model
            import gc
            gc.collect()
        except:
            pass

if __name__ == '__main__':
    main()