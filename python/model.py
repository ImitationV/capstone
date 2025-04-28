import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import yfinance as yf
from finta import TA    
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error



def fetch_daily_data(symbol):
    """
    Fetches daily stock data for a given symbol over the last 5 years
    @return: pandas DF containing daily stock data
    """
    today = datetime.now().strftime("%Y-%m-%d")
    # this is the date 2.5 years ago 
    two_years_ago = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=2.5*365)).strftime("%Y-%m-%d")
    # yf returns a dataframe with OHLCV (Open, High, Low, Close, Volume) data
    df = yf.download(symbol, start=two_years_ago, end=today)


    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Enhanced feature engineering with additional technical indicators
    @param df: pandas DF containing stock data
    @return: pandas DF containing engineered features
    '''

    

    # Trend Indicators
    df['SMA200'] = TA.SMA(df, 200)
    df['SMA50'] = TA.SMA(df, 50)
    df['EMA20'] = TA.EMA(df, 20)
    df['MACD'] = TA.MACD(df)['MACD']
    df['MACD_SIGNAL'] = TA.MACD(df)['SIGNAL']    
    
    # Momentum indicators
    df['Williams'] = TA.WILLIAMS(df)
    df['STOCH'] = TA.STOCH(df)
    df['RSI'] = TA.RSI(df)
    df['STOCH_SIGNAL'] = TA.STOCHD(df)
    df['ROC'] = TA.ROC(df)  # Rate of Change
    
    # Volume-based indicators
    df['OBV'] = TA.OBV(df)  # On Balance Volume
    df['ADL'] = TA.ADL(df)  # Accumulation/Distribution Line
    
    # Volatility indicators
    df['ATR'] = TA.ATR(df)
    df['ATR_PERCENT'] = df['ATR'] / df['Close'] * 100
    df['BBWIDTH'] = TA.BBWIDTH(df)

    # Price patterns
    df['Higher_Highs'] = df['High'] > df['High'].shift(1)
    df['Lower_Lows'] = df['Low'] < df['Low'].shift(1)

    
    # Drop rows with NaN values created by indicators
    df = df.iloc[200:, :]  # Keep the 200-day requirement for SMA
    df['Target'] = df.Close.shift(-1)
    df.dropna(inplace=True)
    
    return df

def train_test_split(data: pd.DataFrame,perc: float) -> tuple[np.ndarray, np.ndarray]:
    '''
    Splits data into train and test sets
    @param data: pandas DF containing stock data
    @param perc: percentage of data for test set
    @return: train set, test set
    '''
    # the data is converted to a numpy array
    ret = data.values
    # the number of rows in the data is calculated 
    n = int(len(data) *  (1-perc)) 

    # the data is split into training and test sets
    return ret[:n], ret[n:]

def xgb_predict(train: pd.DataFrame, val: pd.DataFrame) -> float:
    '''
    Predicts closing price of stock using XGBoost Regressor 
    @param train: pandas DF containing training data 
    @param val: pandas DF as a validation set 
    @return: float prediction

    ---
    - First the training data is converted to a numpy array 
    - Then the features and target values are separated 
    - Then the model is initialized and trained on the training data 
    - Then the validation set is converted to a numpy array and reshaped into a 2D array with 1 row
    - Then the model is used to make a prediction on the validation set 
    - The prediction is then returned as a float
    '''
    # Convert to numpy arrays
    train = np.array(train)
    val = np.array(val)
    
    # Separate features and target
    X = train[:,:-1]  # All columns except last are features
    y = train[:,-1]   # Last column is target
    
    # Initialize and train model
    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=750,
        learning_rate=0.05,
        colsample_bytree=0.7,
        max_depth=3,
        gamma=1
    )
    model.fit(X, y)
    
    # Make prediction
    val = val.reshape(1, -1)
    pred = model.predict(val)
    return pred[0]

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
    Validates the model by predicting the closing price of the stock for each day in the test set 
    @param data: df containing stock data 
    @param perc: percentage of data for test set 
    @return: float error, float MAPE, array of actual target values, array of predicted values
    '''
    predictions = []
    train, test = train_test_split(data, perc)
    history = train.copy()  # Use copy to avoid modifying original
    
    # Iterate through test set
    for i in range(len(test)):
        # Get features and target for current timestep
        X_test = test[i,:-1].reshape(1, -1)  # Reshape for single prediction
        y_test = test[i,-1]
        
        # Make prediction using all features
        pred = xgb_predict(history, X_test)
        predictions.append(pred)
        
        # Add current observation to history
        history = np.vstack([history, test[i]])
    
    # Calculate performance metrics
    error = root_mean_squared_error(test[:,-1], predictions)
    MAPE = mape(test[:,-1], predictions)
    
    return error, MAPE, test[:,-1], predictions

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
    
    # Define risk tolerance thresholds
    risk_thresholds = {
        'conservative': {
            'max_volatility': 0.20,
            'min_sharpe': 1.0,
            'max_var': 0.02,
            'min_return': 0.05,
            'stop_loss': -0.05  # 5% stop loss
        },
        'moderate': {
            'max_volatility': 0.30,
            'min_sharpe': 0.5,
            'max_var': 0.03,
            'min_return': 0.03,
            'stop_loss': -0.08  # 8% stop loss
        },
        'aggressive': {
            'max_volatility': 0.40,
            'min_sharpe': 0.0,
            'max_var': 0.04,
            'min_return': 0.02,
            'stop_loss': -0.12  # 12% stop loss
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
        'position_analysis': []  # New section for position-specific analysis
    }
    
    # 1. Risk Assessment
    risk_score = 0
    if risk_summary['Current_Volatility_20d'] < thresholds['max_volatility']:
        risk_score += 1
        recommendation['risk_assessment'].append("Volatility within acceptable range")
    else:
        risk_score -= 1
        recommendation['risk_assessment'].append("High volatility - exercise caution")
    
    if risk_summary['Sharpe_Ratio'] > thresholds['min_sharpe']:
        risk_score += 1
        recommendation['risk_assessment'].append("Good risk-adjusted returns")
    
    if abs(risk_summary['VaR_95']) < thresholds['max_var']:
        risk_score += 1
        recommendation['risk_assessment'].append("Value at Risk within acceptable range")
    
    # 2. Technical Analysis Signals
    tech_score = 0
    
    # RSI signals
    if latest['RSI'] < 30:
        tech_score += 1
        recommendation['technical_signals'].append("Oversold condition (RSI)")
    elif latest['RSI'] > 70:
        tech_score -= 1
        recommendation['technical_signals'].append("Overbought condition (RSI)")
    
    # MACD signals
    if latest['MACD'] > latest['MACD_SIGNAL']:
        tech_score += 1
        recommendation['technical_signals'].append("Positive MACD crossover")
    else:
        tech_score -= 1
        recommendation['technical_signals'].append("Negative MACD crossover")
    
    # Moving Average signals
    if latest['Close'] > latest['SMA200']:
        tech_score += 1
        recommendation['technical_signals'].append("Price above 200-day SMA")
    
    # 3. Prediction Analysis
    pred_score = 0
    current_price = latest['Close']
    
    # Calculate average predicted return
    avg_prediction = np.mean(predictions[-5:])
    predicted_return = (avg_prediction - current_price) / current_price
    
    if predicted_return > thresholds['min_return']:
        pred_score += 1
        recommendation['prediction_signals'].append(
            f"Short-term prediction shows {predicted_return:.1%} potential return"
        )
    
    # Add trend analysis
    if predictions[-1] > predictions[-2]:
        pred_score += 1
        recommendation['prediction_signals'].append("Upward prediction trend")
    else:
        pred_score -= 1
        recommendation['prediction_signals'].append("Downward prediction trend")
    
    # 4. Position Analysis
    if owns_stock:
        # Check stop loss
        recent_return = (current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]
        if recent_return < thresholds['stop_loss']:
            recommendation['position_analysis'].append(
                f"Stop loss triggered: Current loss ({recent_return:.1%}) exceeds threshold ({thresholds['stop_loss']:.1%})"
            )
            pred_score -= 2
        
        # Check if holding is still profitable
        if predicted_return < 0:
            recommendation['position_analysis'].append(
                "Negative future return predicted, consider taking profits"
            )
            pred_score -= 1
        else:
            recommendation['position_analysis'].append(
                "Positive future return predicted, consider holding position"
            )
    else:
        # Entry analysis for non-holders
        if predicted_return > thresholds['min_return'] and tech_score > 0:
            recommendation['position_analysis'].append(
                "Good entry point: positive technical signals and predicted returns"
            )
        else:
            recommendation['position_analysis'].append(
                "Wait for better entry point"
            )
    
    # 5. Generate Final Recommendation
    total_score = risk_score + tech_score + pred_score
    max_possible_score = 3 + 3 + 2  # risk + tech + predictions
    confidence = (total_score / max_possible_score) * 100
    
    # Simplified action logic based on ownership status
    if owns_stock:
        if confidence < -30:  # More negative confidence needed to trigger sell
            action = "SELL"
            recommendation['reasoning'].append("Negative indicators suggest exiting position")
        else:
            action = "HOLD"
            recommendation['reasoning'].append("Current indicators support maintaining position")
    else:
        if confidence > 40:  # Strong positive confidence needed for buy
            action = "BUY"
            recommendation['reasoning'].append("Strong positive indicators suggest entering position")
        else:
            action = "NO ACTION"  # Changed from HOLD to NO ACTION
            recommendation['reasoning'].append("Current indicators suggest waiting for better entry")
    
    recommendation['action'] = action
    recommendation['confidence'] = confidence
    
    return recommendation

def print_recommendation(recommendation: dict):
    '''
    Prints the trading recommendation in a formatted way
    @param recommendation: Dictionary containing recommendation details
    '''
    print("\n=== Trading Recommendation ===")
    print(f"Action: {recommendation['action']}")
    print(f"Confidence: {recommendation['confidence']:.1f}%")
    
    print("\nRisk Assessment:")
    for item in recommendation['risk_assessment']:
        print(f"• {item}")
    
    print("\nTechnical Signals:")
    for item in recommendation['technical_signals']:
        print(f"• {item}")
    
    print("\nPrediction Signals:")
    for item in recommendation['prediction_signals']:
        print(f"• {item}")

def main():
    # Fetch data
    df = fetch_daily_data('AAPL')
    df.columns = df.columns.droplevel(1)
    
    # Engineer features with enhanced indicators
    df = engineer_features(df)
  
    # data split
    train, test = train_test_split(df,.7)
    
    # Initialize the model
    xgb_predict(train, test[0, :-1])
    
    # Validate and get predictions
    rmse, MAPE, y, pred = validate(df, 0.7)
    pred = np.array(pred)
    test_pred = np.c_[test, pred]
    
    # Create visualization DataFrame with correct column names
    columns = list(df.columns) + ['Pred']  # Use actual column names from df
    df_TP = pd.DataFrame(test_pred, columns=columns[:test_pred.shape[1]])
    
    # Prepare data for plotting
    df = df.reset_index(names='Date')
    df_dates = df[['Date', 'Target', 'Close']]
    df_TP = pd.merge(df_TP, df_dates, on='Target', how='left')
    df_TP = df_TP.sort_values(by='Date').reset_index(drop=True)

    print(df_TP['Date'].head().dtype)
    
    # Create visualization
    plt.figure(figsize=(15, 6))
    plt.title("AAPL Price Predictions", fontsize=18)
    plt.plot(df_TP['Date'], df_TP['Target'], label='Next Day Actual Close Price', color='cyan')
    plt.plot(df_TP['Date'], df_TP['Pred'], label='Predicted Price', color='green', alpha=1)
    plt.xlabel('Date', fontsize=18)
    plt.legend(loc='upper left')
    plt.ylabel('Price USD', fontsize=18)
    
    # Print performance metrics
    print(f"\nModel Performance Metrics:")
    print(f"RMSE: ${rmse:.2f}")
    print(f"MAPE: {MAPE:.2f}%")

    # Calculate risk metrics
    df = calculate_risk_metrics(df)
    
    # Get risk summary
    risk_summary = get_risk_summary(df)
    
    # Print risk metrics
    print("\nRisk Assessment Metrics:")
    print(f"20-day Volatility: {risk_summary['Current_Volatility_20d']:.2%}")
    print(f"60-day Volatility: {risk_summary['Current_Volatility_60d']:.2%}")
    print(f"Sharpe Ratio: {risk_summary['Sharpe_Ratio']:.2f}")
    print(f"Sortino Ratio: {risk_summary['Sortino_Ratio']:.2f}")
    print(f"95% VaR: {risk_summary['VaR_95']:.2%}")
    print(f"Maximum Drawdown: {risk_summary['Max_Drawdown']:.2%}")
    
    # Add risk visualization
    plot_risk_metrics(df)

    
    # Generate trading recommendations
    recommendation = generate_trading_recommendations(
        df, 
        risk_summary, 
        pred,
        risk_tolerance='moderate',
        owns_stock=False  # Change this based on whether you own the stock
    )
    
    # Print recommendations
    print_recommendation(recommendation)
    plt.show()

if __name__ == '__main__':
    main()