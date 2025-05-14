import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import yfinance as yf
from finta import TA    
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import time



def fetch_daily_data(symbol):
    """
    Fetches daily stock data for a given symbol over the last 2.5 years
    @return: pandas DF containing daily stock data
    """
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        two_years_ago = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=2.5*365)).strftime("%Y-%m-%d")
        
        # Fetch data with retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                df = yf.download(symbol, start=two_years_ago, end=today)
                if not df.empty:
                    print(f"Successfully fetched data for {symbol}")
                    return df
                else:
                    print(f"Attempt {attempt + 1}: No data received for {symbol}")
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2)  # Wait before retrying
        
        raise ValueError(f"No data available for symbol {symbol}")
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        # Return a sample DataFrame with the required columns
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Enhanced feature engineering with additional technical indicators
    @param df: pandas DF containing stock data
    @return: pandas DF containing engineered features
    '''
    # Ensure we're working with a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Use vectorized operations for basic indicators
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    df['EMA50'] = df['Close'].ewm(span=50).mean()
    
    # Add more sophisticated indicators
    df['Price_Change'] = df['Close'].pct_change()
    df['Volume_Change'] = df['Volume'].pct_change()
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_Open_Ratio'] = df['Close'] / df['Open']
    
    # Add Bollinger Bands with proper Series handling
    close_series = df['Close']
    bb_middle = close_series.rolling(window=20).mean()
    bb_std = close_series.rolling(window=20).std()
    
    df['BB_Middle'] = bb_middle
    df['BB_Upper'] = bb_middle + (2 * bb_std)
    df['BB_Lower'] = bb_middle - (2 * bb_std)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    # Calculate essential indicators
    try:
        # Ensure column names are in the correct case for finta
        df.columns = [col.lower() for col in df.columns]
        
        # Calculate RSI
        rsi = TA.RSI(df)
        if isinstance(rsi, tuple):
            df['RSI'] = rsi[0]
        else:
            df['RSI'] = rsi
            
        # Calculate MACD
        macd = TA.MACD(df)
        if isinstance(macd, tuple):
            df['MACD'] = macd[0]['MACD']
            df['MACD_SIGNAL'] = macd[0]['SIGNAL']
            df['MACD_HIST'] = macd[0]['HIST']
        else:
            df['MACD'] = macd['MACD']
            df['MACD_SIGNAL'] = macd['SIGNAL']
            df['MACD_HIST'] = macd['HIST']
            
        # Add Stochastic Oscillator
        stoch = TA.STOCH(df)
        df['STOCH_K'] = stoch['STOCH_K']
        df['STOCH_D'] = stoch['STOCH_D']
        
        # Add ATR (Average True Range)
        df['ATR'] = TA.ATR(df)
        
    except Exception as e:
        print(f"Error calculating technical indicators: {str(e)}")
        # Set default values if calculation fails
        df['RSI'] = 50
        df['MACD'] = 0
        df['MACD_SIGNAL'] = 0
        df['MACD_HIST'] = 0
        df['STOCH_K'] = 50
        df['STOCH_D'] = 50
        df['ATR'] = 0
    
    # Add lagged features
    for lag in [1, 2, 3, 5]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
    
    # Add rolling statistics
    for window in [5, 10, 20]:
        df[f'Close_MA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'Volume_MA_{window}'] = df['Volume'].rolling(window=window).mean()
        df[f'Close_Std_{window}'] = df['Close'].rolling(window=window).std()
    
    # Drop rows with NaN values
    df = df.iloc[200:, :]
    df['Target'] = df.Close.shift(-1)
    df.dropna(inplace=True)
    
    return df

def train_ensemble_model(train_data: pd.DataFrame) -> tuple:
    '''
    Train an ensemble of models for more robust predictions
    @param train_data: DataFrame containing training data
    @return: Tuple of trained models and scaler
    '''
    # Prepare data
    X = train_data.iloc[:, :-1]
    y = train_data.iloc[:, -1]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize models with optimized hyperparameters
    models = {
        'xgb': XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=5,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1,
            n_jobs=-1
        ),
        'rf': RandomForestRegressor(
            n_estimators=500,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1
        ),
        'gb': GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2
        )
    }
    
    # Train models using time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = {name: [] for name in models.keys()}
    
    for train_idx, val_idx in tscv.split(X_scaled):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            score = mean_absolute_percentage_error(y_val, pred)
            cv_scores[name].append(score)
    
    # Calculate model weights based on CV performance
    weights = {}
    for name, scores in cv_scores.items():
        weights[name] = 1 / (np.mean(scores) + 1e-10)
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {name: w/total_weight for name, w in weights.items()}
    
    # Train final models on full dataset
    for model in models.values():
        model.fit(X_scaled, y)
    
    return models, scaler, weights

def predict_future(df: pd.DataFrame, days: int = 10) -> tuple[list, list]:
    '''
    Enhanced future predictions using ensemble methods
    @param df: DataFrame containing historical data and features
    @param days: Number of days to predict into the future
    @return: Tuple of (predictions list, dates list)
    '''
    if df.empty or len(df) < 50:
        print("Insufficient data for predictions")
        return [], []
        
    try:
        # Train ensemble model
        models, scaler, weights = train_ensemble_model(df)
        
        predictions = []
        dates = []
        current_data = df.iloc[-1:].copy()
        last_date = df.index[-1]
        
        # Process predictions in batches
        for i in range(0, days, 5):
            batch_days = min(5, days - i)
            batch_predictions = []
            
            for j in range(batch_days):
                next_date = last_date + timedelta(days=i+j+1)
                dates.append(next_date)
                
                # Get features and scale them
                features = current_data.iloc[:, :-1]
                features_scaled = scaler.transform(features)
                
                # Get predictions from each model
                model_predictions = []
                for name, model in models.items():
                    pred = model.predict(features_scaled)[0]
                    model_predictions.append(pred * weights[name])
                
                # Weighted ensemble prediction
                pred = sum(model_predictions)
                batch_predictions.append(pred)
                
                # Update current data
                current_data['Close'] = pred
                current_data['Target'] = pred
                
                # Update technical indicators
                current_data['SMA200'] = (df['Close'].iloc[-199:].sum() + pred) / 200
                current_data['SMA50'] = (df['Close'].iloc[-49:].sum() + pred) / 50
                current_data['EMA20'] = pred * 0.1 + current_data['EMA20'].iloc[-1] * 0.9
                current_data['EMA50'] = pred * 0.02 + current_data['EMA50'].iloc[-1] * 0.98
                
                # Update other indicators
                current_data['Price_Change'] = (pred - current_data['Close'].iloc[-1]) / current_data['Close'].iloc[-1]
                current_data['High_Low_Ratio'] = pred / (pred * 0.99)  # Simplified
                current_data['Close_Open_Ratio'] = pred / (pred * 0.995)  # Simplified
                
                # Fill NaN values with 0
                current_data = current_data.fillna(0)
            
            predictions.extend(batch_predictions)
        
        return predictions, dates
        
    except Exception as e:
        print(f"Error in predict_future function: {str(e)}")
        return [], []

def validate(data: pd.DataFrame, perc: float) -> tuple:
    '''
    Optimized validation function that reuses the trained model
    @param data: df containing stock data
    @param perc: percentage of data for test set
    @return: tuple of (error, MAPE, actual values, predictions)
    '''
    # Check if we have enough data
    if len(data) < 50:  # Minimum required data points
        print("Not enough data for validation")
        return 0.0, 0.0, [], []
    
    try:
        # Convert DataFrame to numpy array
        data_array = data.values
        
        # Calculate split index
        n = int(len(data_array) * (1 - perc))
        train = data_array[:n]
        test = data_array[n:]
        
        # Check if test set is empty
        if len(test) == 0:
            print("Test set is empty")
            return 0.0, 0.0, [], []
        
        predictions = []
        history = train.copy()
        model = None
        
        for i in range(len(test)):
            X_test = test[i,:-1].reshape(1, -1)
            y_test = test[i,-1]
            
            # Train model if not exists
            if model is None:
                X_train = history[:,:-1]
                y_train = history[:,-1]
                
                model = XGBRegressor(
                    objective='reg:squarederror',
                    n_estimators=500,
                    learning_rate=0.1,
                    colsample_bytree=0.7,
                    max_depth=3,
                    gamma=1,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
            
            # Make prediction
            pred = model.predict(X_test)[0]
            predictions.append(pred)
            
            # Update history
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
        MAPE = np.mean(np.abs((actual - predictions) / actual)) * 100
        
        return error, MAPE, actual, predictions
        
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
                                  risk_tolerance: str = 'moderate') -> dict:
    # Get latest data
    latest = df.iloc[-1]
    
    # Define even more aggressive risk tolerance thresholds
    risk_thresholds = {
        'conservative': {
            'max_volatility': 0.50,  # Increased from 0.40
            'min_sharpe': 0.2,      # Decreased from 0.4
            'max_var': 0.07,        # Increased from 0.05
            'min_return': 0.02,     # Decreased from 0.03
            'stop_loss': -0.10      # Increased from -0.08
        },
        'moderate': {
            'max_volatility': 0.65,  # Increased from 0.50
            'min_sharpe': 0.0,      # Decreased from 0.2
            'max_var': 0.08,        # Increased from 0.06
            'min_return': 0.01,     # Decreased from 0.02
            'stop_loss': -0.12      # Increased from -0.10
        },
        'aggressive': {
            'max_volatility': 0.85,  # Increased from 0.70
            'min_sharpe': -0.2,     # Decreased from 0.0
            'max_var': 0.10,        # Increased from 0.08
            'min_return': 0.005,    # Decreased from 0.01
            'stop_loss': -0.18      # Increased from -0.15
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
    
    # 1. Risk Assessment with even more lenient scoring
    risk_score = 0
    
    # Volatility scoring with more lenient trend consideration
    vol_trend = df['Volatility_20d'].iloc[-5:].mean() - df['Volatility_20d'].iloc[-10:-5].mean()
    if risk_summary['Current_Volatility_20d'] < thresholds['max_volatility']:
        risk_score += 1.5  # Increased base score
        recommendation['risk_assessment'].append("Volatility within acceptable range")
        if vol_trend < 0:  # Bonus for decreasing volatility
            risk_score += 0.5
            recommendation['risk_assessment'].append("Volatility is decreasing")
    else:
        if vol_trend < 0:  # Volatility is decreasing
            risk_score -= 0.2  # Reduced penalty
            recommendation['risk_assessment'].append("Elevated but decreasing volatility")
        else:
            risk_score -= 0.3  # Reduced penalty
            recommendation['risk_assessment'].append("Elevated volatility - monitor closely")
    
    # Sharpe Ratio with more lenient trend consideration
    sharpe_trend = df['Sharpe_Ratio'].iloc[-5:].mean() - df['Sharpe_Ratio'].iloc[-10:-5].mean()
    if risk_summary['Sharpe_Ratio'] > thresholds['min_sharpe']:
        risk_score += 1.5  # Increased base score
        recommendation['risk_assessment'].append("Good risk-adjusted returns")
        if sharpe_trend > 0:  # Bonus for improving Sharpe ratio
            risk_score += 0.5
            recommendation['risk_assessment'].append("Risk-adjusted returns improving")
    else:
        if sharpe_trend > 0:  # Sharpe ratio is improving
            risk_score -= 0.2  # Reduced penalty
            recommendation['risk_assessment'].append("Risk-adjusted returns improving")
        else:
            risk_score -= 0.3  # Reduced penalty
            recommendation['risk_assessment'].append("Below target risk-adjusted returns")
    
    # VaR with more lenient market context
    if abs(risk_summary['VaR_95']) < thresholds['max_var']:
        risk_score += 1.5  # Increased base score
        recommendation['risk_assessment'].append("Value at Risk within acceptable range")
        if abs(risk_summary['VaR_95']) < abs(df['VaR_95'].mean()):  # Bonus for better than average
            risk_score += 0.5
            recommendation['risk_assessment'].append("VaR better than average")
    else:
        if abs(risk_summary['VaR_95']) < abs(df['VaR_95'].mean()):  # Better than average
            risk_score -= 0.2  # Reduced penalty
            recommendation['risk_assessment'].append("VaR elevated but better than average")
        else:
            risk_score -= 0.3  # Reduced penalty
            recommendation['risk_assessment'].append("VaR above acceptable range")
    
    # 2. Technical Analysis with more lenient conditions
    tech_score = 0
    
    # RSI with more lenient trend consideration
    rsi_trend = df['RSI'].iloc[-5:].mean() - df['RSI'].iloc[-10:-5].mean()
    rsi_value = float(latest['RSI'].iloc[0])
    
    if rsi_value < 40:  # More lenient oversold threshold
        tech_score += 2.0 if rsi_trend > 0 else 1.5  # Increased scores
        recommendation['technical_signals'].append("Oversold condition (RSI)")
    elif rsi_value > 60:  # More lenient overbought threshold
        tech_score -= 1.5 if rsi_trend < 0 else 1
        recommendation['technical_signals'].append("Overbought condition (RSI)")
    elif 25 <= rsi_value <= 75:  # More lenient neutral zone
        tech_score += 1.0  # Increased score
        recommendation['technical_signals'].append("RSI in neutral zone")
    
    # MACD with more lenient trend strength
    macd_trend = df['MACD'].iloc[-5:].mean() - df['MACD'].iloc[-10:-5].mean()
    macd_value = float(latest['MACD'].iloc[0])
    macd_signal_value = float(latest['MACD_SIGNAL'].iloc[0])
    
    if macd_value > macd_signal_value:
        tech_score += 2.0 if macd_trend > 0 else 1.5  # Increased scores
        recommendation['technical_signals'].append("Positive MACD crossover")
    else:
        tech_score -= 0.8  # Reduced penalty
        recommendation['technical_signals'].append("Negative MACD crossover")
    
    # Moving Average analysis with more lenient conditions
    ma_score = 0
    close_value = float(latest['Close'].iloc[0])
    sma200_value = float(latest['SMA200'].iloc[0])
    sma50_value = float(latest['SMA50'].iloc[0])
    
    if close_value > sma200_value:
        ma_score += 1.5  # Increased score
        recommendation['technical_signals'].append("Price above 200-day SMA")
    if close_value > sma50_value:
        ma_score += 1.0  # Increased score
        recommendation['technical_signals'].append("Price above 50-day SMA")
    tech_score += ma_score
    
    # 3. Prediction Analysis with more lenient conditions
    pred_score = 0
    current_price = close_value
    
    # Calculate average predicted return with more lenient trend consideration
    avg_prediction = np.mean(predictions[-5:])
    predicted_return = (avg_prediction - current_price) / current_price
    
    # Consider prediction stability with more lenient requirements
    pred_volatility = np.std(predictions[-5:]) / np.mean(predictions[-5:])
    pred_trend = predictions[-1] - predictions[-5]
    
    if predicted_return > thresholds['min_return']:
        pred_score += 1.5  # Increased base score
        if pred_volatility < 0.04:  # More lenient volatility threshold
            pred_score += 1.0  # Increased bonus
            recommendation['prediction_signals'].append(
                f"Stable positive prediction: {predicted_return:.1%} potential return"
            )
        else:
            recommendation['prediction_signals'].append(
                f"Short-term prediction shows {predicted_return:.1%} potential return"
            )
    
    # Trend analysis with more lenient momentum requirements
    if predictions[-1] > predictions[-2]:
        pred_score += 1.5  # Increased base score
        if pred_trend > 0:  # More lenient trend requirement
            pred_score += 1.0  # Increased bonus
            recommendation['prediction_signals'].append("Strong upward prediction trend")
        else:
            recommendation['prediction_signals'].append("Upward prediction trend")
    else:
        pred_score -= 0.8  # Reduced penalty
        recommendation['prediction_signals'].append("Downward prediction trend")
    
    # 4. Position Analysis
    if predicted_return > thresholds['min_return'] or tech_score > 0:  # Either condition is sufficient
        if pred_volatility < 0.04 or macd_trend > 0 or rsi_value < 45:  # More lenient requirements
            recommendation['position_analysis'].append(
                "Good entry point: positive technical signals and predicted returns"
            )
        else:
            recommendation['position_analysis'].append(
                "Potential entry point: positive technical signals"
            )
    else:
        recommendation['position_analysis'].append(
            "Wait for better entry point"
        )
    
    # 5. Generate Final Recommendation with more lenient requirements
    total_score = (risk_score * 1.2) + (tech_score * 1.0) + (pred_score * 0.8)
    max_possible_score = (3 * 1.2) + (3 * 1.0) + (2 * 0.8)
    confidence = (total_score / max_possible_score) * 100
    
    # Modified action logic with slightly more conservative requirements
    # Require higher confidence and at least one positive technical signal for relaxed BUY
    positive_technical = (
        (rsi_value < 40) or
        (macd_value > macd_signal_value) or
        (close_value > sma50_value) or
        (close_value > sma200_value)
    )
    if confidence < -40 and pred_trend < 0:  # More lenient sell threshold
        action = "SELL"
        recommendation['reasoning'].append("Negative indicators and trend suggest selling")
    elif (confidence >= 65 and 
          risk_summary['Current_Volatility_20d'] < thresholds['max_volatility'] and
          risk_summary['Sharpe_Ratio'] > thresholds['min_sharpe'] and
          abs(risk_summary['VaR_95']) < thresholds['max_var'] and
          positive_technical):
        action = "BUY"
        recommendation['reasoning'].append("High confidence, favorable risk metrics, and at least one positive technical signal suggest buying")
    elif confidence >= -10 and pred_trend > 0:  # More lenient buy threshold
        action = "BUY"
        recommendation['reasoning'].append("Positive indicators and trend suggest buying")
    else:
        action = "NO ACTION"
        recommendation['reasoning'].append("Current indicators suggest waiting for better conditions")
    
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
    elif recommendation['action'] == "NO ACTION":
        print("\nThe overall analysis suggests waiting for better conditions:")
        print("• Current market conditions are unclear")
        print("• It's better to wait for more positive signals")
    
    print("="*80 + "\n")

def display_future_predictions(predictions: list, dates: list, current_price: float):
    '''
    Enhanced display of future predictions with confidence intervals
    @param predictions: List of predicted prices
    @param dates: List of dates for predictions
    @param current_price: Current stock price
    '''
    print("\n" + "="*80)
    print("10-DAY PRICE PREDICTION FORECAST")
    print("="*80)
    
    # Calculate confidence intervals
    pred_std = np.std(predictions)
    confidence_intervals = {
        '68%': 1.0,  # 1 standard deviation
        '95%': 1.96,  # 2 standard deviations
        '99%': 2.58   # 3 standard deviations
    }
    
    # Print header
    print(f"\nCurrent Price: ${current_price:.2f}")
    print("\nDaily Predictions with Confidence Intervals:")
    print("-"*100)
    print(f"{'Day':<8} {'Predicted Price':<15} {'Change':<10} {'% Change':<10} {'Confidence Range':<30}")
    print("-"*100)
    
    # Print daily predictions with confidence intervals
    for i, (date, price) in enumerate(zip(dates, predictions)):
        change = price - current_price
        pct_change = (change / current_price) * 100
        
        # Calculate confidence ranges
        ranges = []
        for conf, multiplier in confidence_intervals.items():
            lower = price - (pred_std * multiplier)
            upper = price + (pred_std * multiplier)
            ranges.append(f"{conf}: ${lower:.2f}-${upper:.2f}")
        
        print(f"Day {i+1:<3} ${price:<14.2f} {change:>+8.2f} {pct_change:>+8.2f}% {' | '.join(ranges)}")
    
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
    
    # Calculate prediction metrics
    pred_volatility = np.std(predictions) / np.mean(predictions) * 100
    pred_range = (max(predictions) - min(predictions)) / current_price * 100
    
    print(f"Predicted Volatility: {pred_volatility:.2f}%")
    print(f"Predicted Price Range: {pred_range:.2f}%")
    
    # Add prediction confidence assessment
    confidence_level = 100 - (pred_volatility * 2)  # Simple confidence metric
    confidence_level = max(0, min(100, confidence_level))  # Clamp between 0 and 100
    
    print(f"\nPrediction Confidence: {confidence_level:.1f}%")
    if confidence_level >= 80:
        print("→ High confidence in predictions")
    elif confidence_level >= 60:
        print("→ Moderate confidence in predictions")
    else:
        print("→ Low confidence in predictions - consider waiting for more data")
    
    print("="*80 + "\n")

def main():
    try:
        # Fetch data with reduced time range
        df = fetch_daily_data('AAPL')
        
        # Check if we have valid data
        if df.empty:
            print("No data available for analysis")
            return
            
        df.columns = df.columns.droplevel(1)
        
        # Convert to float32 for memory efficiency
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        
        # Engineer features with optimized indicators
        df = engineer_features(df)
        
        # Check if we have enough data after feature engineering
        if len(df) < 50:
            print("Not enough data points after feature engineering")
            return
        
        # Split data
        train, test = train_test_split(df, 0.7)
        
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
        model.fit(X_train, y_train)
        
        # Validate and get predictions
        rmse, MAPE, y, pred = validate(df, 0.7)
        if len(pred) == 0:
            print("No predictions generated")
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
        
        # Get future predictions (reduced to 10 days)
        df_indexed = df.set_index('Date')
        future_predictions, future_dates = predict_future(df_indexed, days=10)
        
        if len(future_predictions) == 0:
            print("No future predictions generated")
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
        
        # Calculate risk metrics
        df = calculate_risk_metrics(df)
        risk_summary = get_risk_summary(df)
        
        # Generate and print recommendations
        recommendation = generate_trading_recommendations(
            df, 
            risk_summary, 
            pred,
            risk_tolerance='moderate'
        )
        print_recommendation(recommendation)
        
        plt.show()
        
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