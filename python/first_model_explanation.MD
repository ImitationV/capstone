### The notebook implements a stock price prediction model using XGBoost Regression

---

### Data Processing:
- The engineer_features function calculates several technical indicators:
- Simple Moving Average (SMA200)
    - This is a 200-day moving average of the closing prices
    - Why it's useful:
    - Helps identify long-term trends
    - Acts as a support/resistance level
    - When price crosses above/below SMA200, it often signals a major trend change
    - Smoothes out short-term price fluctuations to show the underlying trend
- Relative Strength Index (RSI)
    - A momentum oscillator that measures the speed and change of price movements
    - Why it's useful:
    - Identifies overbought (typically >70) and oversold (typically <30) conditions
    - Helps spot potential trend reversals
    - Can indicate when a stock might be due for a correction
    - Useful for identifying divergence between price and momentum
- Average True Range (ATR)
    - Measures market volatility by calculating the average range between high and low prices
    - Why it's useful:
    - Helps determine the volatility of a stock
    - Useful for setting stop-loss levels
    - Can indicate when market volatility is increasing or decreasing
    - Helps in risk management and position sizing
- Bollinger Bands Width (BBWIDTH)
    - Measures the width of Bollinger Bands (volatility bands placed above and below a moving average)
    - Why it's useful:
    - Indicates market volatility
    - Narrow bands suggest low volatility (potential breakout coming)
    - Wide bands suggest high volatility
    - Helps identify potential trend reversals when bands contract after expansion
- Williams %R   
    - A momentum indicator that shows where the current price is relative to the highest high for a specific period
    - Why it's useful:
    - Identifies overbought and oversold conditions
    - Helps spot potential trend reversals
    - Works well in ranging markets
    - Can be used to confirm other technical signals
- The data is split into training and testing sets using a 50% split

--- 
### Model Training

params = {
    'max_depth': [3,6], - determines the max depth of each tree in the model 
    'learning_rate': [0.05], - controls the step size in each boosting iteration which prevents overfitting
    'n_estimators': [700,1000], - number of trees in the model
    'colsample_bytree': [0.3, 0.7], - determines the fraction of features to randomly sample for each tree, helps prevent overfitting
}

    - The learning rate is a key hyperparameter that determines how much each new tree added to the model contributes to the overall prediction
    -  It acts as a scaling factor for the corrections made by each successive tree in the boosting process

##### How it works:
- Sequential Learning: trees are added one at a time each aiming to correct the errors (residuals) of the existing ensemble
- The learning rate multiplies the output of each new tree before adding it to the ensemble 
- A learning rate of 1.0 means each tree's corrections are fully applied 
- Values less than 1 reduce the impact of each tree - making the model learn more slowly and cautiously 
- Lower Learning Rate -- model learns slower, requires more trees, prevents overfitting, training takes longer 


#### XGBoost Regressor is a type of gradient boosting model that build trees one at a time:
- each tree is trained to correct the errors of the previous tree by focusing on the examples that were misclassified 
- the model then combines the predictions of all trees to make a final prediction
- gradient boosting is a technique that build an ensemble of weak models to create a strong predictive model


#### GridSearchCV is used to find the best parameters for the model by searching through a grid of parameters
- the scoring is neg_mean_squared_error which is the average of the squared differences between the predicted and actual values
- this metric penalizes large errors more heavily which helps the model focus on minimizing large errors

---
# Model Evaluation:
- The model achieved:
- RMSE (Root Mean Square Error): 6.425
- MAPE (Mean Absolute Percentage Error): 1.31%
- These metrics indicate relatively good performance, with the MAPE being particularly low at 1.31%

--- 
# Results Visualization
- The xgb_predict function handles individual predictions
- The validate function performs a walk-forward validation on the test set
- Predictions are stored alongside actual values for comparison

