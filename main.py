import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import gurobipy as gp
import ta
import quantstats as qs
from sklearn.svm import SVR
import warnings
from datetime import datetime



# Suppress specific FutureWarnings and other non-critical messages
warnings.filterwarnings("ignore", category=FutureWarning, module="numpy")
warnings.filterwarnings("ignore", category=FutureWarning, module="quantstats")
warnings.filterwarnings("ignore", message="dropping on a non-lexsorted multi-index")
pd.set_option('future.no_silent_downcasting', True)

# Define benchmark symbols and historical components for the Dow Jones Industrial Average
benchmarks = ['^DJI', 'SPY']
dow_jones_components = {
    # Historical components of the Dow Jones Industrial Average: https://en.wikipedia.org/wiki/Historical_components_of_the_Dow_Jones_Industrial_Average
    "1991-05-06": ["AA", "AXP", "BA", "CAT", "CVX", "DD", "DIS", "FL", "GE", "GT", "HON", "IBM", "IP", "JPM", "KO", "MCD", "MMM", "MO", "MRK", "PG", "T", "XOM"],
    #add "HPQ", "JNJ", "TRV", "WMT"; remove "FL"
    "1997-03-17": ["AA", "AXP", "BA", "CAT", "CVX", "DD", "DIS", "GE", "GT", "HON", "HPQ", "IBM", "IP", "JNJ", "JPM", "KO", "MCD", "MMM", "MO", "MRK", "PG", "T", "TRV", "WMT", "XOM"],
    #add "C", "HD", "INTC", "MSFT"; remove "CVX", "GT", "TRV"
    "1999-11-01": ["AA", "AXP", "BA", "C", "CAT", "DD", "DIS", "GE", "HD", "HON", "HPQ", "IBM", "INTC", "IP", "JNJ", "JPM", "KO", "MCD", "MMM", "MO", "MRK", "MSFT", "PG", "T", "WMT", "XOM"],
    #add "AIG", "PFE", "VZ"; remove "IP"
    "2004-04-08": ["AA", "AIG", "AXP", "BA", "C", "CAT", "DD", "DIS", "GE", "HD", "HON", 'HPQ', "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MO", "MRK", "MSFT", "PFE", "PG", "T", "VZ", "WMT", "XOM"],
    #add "BAC", "CVX"; remove "HON", "MO"
    "2008-02-19": ["AA", "AIG", "AXP", "BA", "BAC", "C", "CAT", "CVX", "DD", "DIS", "GE", "HD", "HPQ", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "PFE", "PG", "T", "VZ", "WMT", "XOM"],
    #remove "AIG"
    "2008-09-22": ["AA", "AXP", "BA", "BAC", "C", "CAT", "CVX", "DD", "DIS", "GE", "HD", "HPQ", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "PFE", "PG", "T", "VZ", "WMT", "XOM"],
    #add "TRV", "CSCO"; remove "C"
    "2009-06-08": ["AA", "AXP", "BA", "BAC", "CAT", "CSCO", "CVX", "DD", "DIS", "GE", "HD", "HPQ", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "PFE", "PG", "T", "TRV", "VZ", "WMT", "XOM"],
    #add "UNH"
    "2012-09-24": ["AA", "AXP", "BA", "BAC", "CAT", "CSCO", "CVX", "DD", "DIS", "GE", "HD", "HPQ", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "PFE", "PG", "T", "TRV", "UNH", "VZ", "WMT", "XOM"],
    #add "GS", "NKE"; remove "AA", "BAC", "HPQ"
    "2013-09-23": ["AXP", "BA", "CAT", "CSCO", "CVX", "DD", "DIS", "GE", "GS", "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "PFE", "PG", "T", "TRV", "UNH", "VZ", "WMT", "XOM"],
    #add "AAPL"; remove "T"
    "2015-03-19": ["AXP", "AAPL", "BA", "CAT", "CSCO", "CVX", "DD", "DIS", "GE", "GS", "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "PFE", "PG", "TRV", "UNH", "VZ", "WMT", "XOM"],
    #add "WBA"; remove "GE"
    "2018-06-26": ["AXP", "AAPL", "BA", "CAT", "CSCO", "CVX", "DD", "DIS", "GS", "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "PFE", "PG", "TRV", "UNH", "VZ", "WBA", "WMT", "XOM"],
    #remove "DD"
    "2019-04-02": ["AXP", "AAPL", "BA", "CAT", "CSCO", "CVX", "DIS", "GS", "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "PFE", "PG", "TRV", "UNH", "VZ", "WBA", "WMT", "XOM"],
    #add "AMGN", "HON"; remove "XOM", "PFE"
    "2020-08-31": ["AXP", "AAPL", "AMGN", "BA", "CAT", "CSCO", "CVX", "DIS", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "VZ", "WBA", "WMT"],
    
    "Union": ['AA', 'AIG', 'AAPL', 'AMGN', 'AXP', 'BA', 'BAC', 'C', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', "FL", 'GE', 'GS', "GT", 'HD', 'HON', 'HPQ', 'IBM', 'INTC', 'IP',
              'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MO', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'T', 'TRV', 'UNH', 'VZ', 'WBA', 'WMT', 'XOM'],
    #Non-existing or Inaccessible stock data
    "Removed" : ["BS", 'CRM', 'DOW', 'DWDP', "EK", 'GM', "KHC", 'S', 'SBC', "TX", 'UK', 'UTX', 'V', "WX"],
}
spdr_components = {
    "1998-12-22": ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY'],
    #add "XLRE"
    "2015-10-08": ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XLRE'],
    #add "XLC"
    "2018-06-19": ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XLRE'],
    
    # Add future changes here...
    "Union": ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XLC', 'XLRE'],
    "Removed" : [],
}


# Select (1).DJIA or (2).SPDR components for the dataset
components = dow_jones_components
update_dates = list(components.keys())[:-2]  # Exclude "Union" and "Removed"
stock_list = sorted(components["Union"])

# Define date ranges
data_start_date = '1991-01-01'
decision_start_date = '1994-01-01'
end_date = '2024-02-25'

# Download complete price data for stocks
global_price_data = yf.download(stock_list, start=data_start_date, end=end_date, auto_adjust=False)

# Download decision price data and concatenate for each stock
raw_data = pd.DataFrame()
for stock in stock_list:
    stock_data = yf.download(stock, start=decision_start_date, end=end_date, auto_adjust=False)
    print(f'{stock} start from {stock_data.index[0]}')
    stock_data['Symbol'] = stock
    raw_data = pd.concat([raw_data, stock_data], axis=0)

# Pivot the data to have dates as index and stock symbols as columns (using Adjusted Close)
price_data = raw_data.pivot_table(index='Date', columns='Symbol', values='Adj Close').loc[decision_start_date:]
price_data.columns = price_data.columns.droplevel()  # Remove multi-level index
returns_data = price_data.pct_change().iloc[1:]
price_data = price_data.loc[returns_data.index]

# Download benchmark data similarly
benchmark_raw = pd.DataFrame()
for benchmark_symbol in benchmarks:
    bench_data = yf.download(benchmark_symbol, start=price_data.index[0], end=price_data.index[-1], auto_adjust=False)
    bench_data['Symbol'] = benchmark_symbol
    benchmark_raw = pd.concat([benchmark_raw, bench_data], axis=0)
benchmark_price_data = benchmark_raw.pivot_table(index='Date', columns='Symbol', values='Adj Close').loc[decision_start_date:]
benchmark_price_data.columns = benchmark_price_data.columns.droplevel()
benchmark_data = benchmark_price_data.copy()
benchmark_returns_data = benchmark_data.pct_change().iloc[1:]
benchmark_data = benchmark_data.loc[benchmark_returns_data.index]

# Plot cumulative returns for stocks and benchmarks
fig, ax = plt.subplots(figsize=(20, 8))
(1 + returns_data).cumprod().plot(ax=ax)
(1 + benchmark_returns_data).cumprod().plot(ax=ax)
legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, title='Assets')
plt.setp(legend.get_title(), fontsize=12)
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Cumulative Returns', fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.tight_layout(rect=[0, 0, 0.75, 1])
plt.show()

def equal_weighting_portfolio(returns_df, exclude_asset):
    """
    Compute equal-weighted portfolio returns excluding a specified asset.
    """
    assets = returns_df.columns[returns_df.columns != exclude_asset]
    weights = np.array([1/len(assets)] * len(assets))
    strategy_returns = returns_df.copy()
    strategy_returns['portfolio'] = returns_df[assets].mul(weights, axis=1).sum(axis=1)
    return strategy_returns

# Example equal-weighted strategy (exclude_asset is None, so all assets are included)
equal_weighting_strategy = [equal_weighting_portfolio(returns_data.copy(), None)]

# Define the number of technical indicators used
NUM_INDICATORS = 9

def compute_technical_indicators(stock_df, stock_label=''):
    """
    Compute a set of technical indicators for a given stock's price data.
    Returns a DataFrame with the computed indicators.
    """
    stock_df = stock_df.dropna()
    # Standard technical indicators
    atr = ta.volatility.AverageTrueRange(high=stock_df['High'],
                                         low=stock_df['Low'],
                                         close=stock_df['Close']).average_true_range()
    adx = ta.trend.ADXIndicator(high=stock_df['High'],
                                low=stock_df['Low'],
                                close=stock_df['Close']).adx()
    ema = ta.trend.EMAIndicator(close=stock_df['Adj Close']).ema_indicator()
    macd = ta.trend.MACD(close=stock_df['Adj Close']).macd()
    sma = ta.trend.SMAIndicator(close=stock_df['Adj Close'], window=20).sma_indicator()
    rsi = ta.momentum.RSIIndicator(close=stock_df['Adj Close']).rsi()

    # Additional indicators
    bollinger = ta.volatility.BollingerBands(close=stock_df['Close'])
    obv = ta.volume.OnBalanceVolumeIndicator(close=stock_df['Close'], volume=stock_df['Volume']).on_balance_volume()
    obv_normalized = obv / (obv.max() - obv.min())

    label_suffix = f"_{stock_label}" if stock_label else ""
    indicators = pd.DataFrame({
        f'ATR{label_suffix}': atr,
        f'ADX{label_suffix}': adx,
        f'EMA{label_suffix}': ema,
        f'MACD{label_suffix}': macd,
        f'SMA{label_suffix}': sma,
        f'RSI{label_suffix}': rsi,
        f'BB_Upper{label_suffix}': bollinger.bollinger_hband(),
        f'BB_Lower{label_suffix}': bollinger.bollinger_lband(),
        f'OBV_Norm{label_suffix}': obv_normalized
    })

    indicators = indicators.dropna()
    indicators = indicators.loc[(indicators != 0).all(axis=1)]
    return indicators

# Compute technical indicators for each stock and combine them into a global indicator DataFrame
indicator_samples = []
for stock in stock_list:
    stock_indicators = compute_technical_indicators(global_price_data.xs(stock, axis=1, level=1), stock_label=stock)
    indicator_samples.append(stock_indicators)
global_indicators = pd.concat(indicator_samples, axis=1, keys=stock_list)

def estimate_svr_parameters(price_data, features, stock_name):
    """
    Estimate regression parameters for one stock using Support Vector Regression.
    Returns the model weights, intercept, and variance of the training features.
    """
    X_train_raw = features[stock_name]
    Y_train_raw = price_data['Adj Close'][stock_name]
    X_train = X_train_raw.dropna()
    Y_train = Y_train_raw.loc[X_train.index]
    
    svr_model = SVR(kernel='linear')
    svr_model.fit(X_train, Y_train)
    
    weights = np.vstack(svr_model.coef_)
    intercepts = svr_model.intercept_
    variance = np.var(X_train, axis=0).values
    return weights, intercepts, variance

def perform_rolling_training(returns_df, features, train_start, train_end):
    """
    Perform rolling training to estimate regression parameters and uncertainty.
    Returns the uncertainty matrix, regression weights matrix, and intercept vector.
    """
    n = len(returns_df.index)
    d = len(returns_df.columns)
    regression_weights = np.zeros((d, d * NUM_INDICATORS))
    intercept_vector = np.zeros(d)
    
    h_factor = (4 / (d * NUM_INDICATORS + 2))**(2 / (d * NUM_INDICATORS + 4)) * n**(-2 / (d * NUM_INDICATORS + 4))
    h_values = np.array([])
    price_data_window = global_price_data.loc[train_start:train_end, (slice(None), list(returns_df.columns))]
    
    for j, stock in enumerate(returns_df.columns):
        weights, intercepts, variance = estimate_svr_parameters(price_data_window, features, stock)
        regression_weights[j, NUM_INDICATORS * j: NUM_INDICATORS * (j + 1)] = weights.flatten()
        intercept_vector[j] = intercepts[0]
        h_values = np.append(h_values, h_factor * variance)
        
    H_tilde = np.diag(h_values)
    uncertainty_matrix = regression_weights @ H_tilde @ regression_weights.T
    return uncertainty_matrix, regression_weights, intercept_vector

def compute_black_litterman_estimates(returns_df):
    """
    Compute the Black-Litterman adjusted mean and covariance estimates.
    """
    tau = 1
    prior_mean = returns_df.mean().values
    sample_covariance = returns_df.cov().values
    d = len(returns_df.columns)
    identity_matrix = np.eye(d)
    features = global_indicators[returns_df.columns].loc[returns_df.index]
    
    Lambda, B, intercepts = perform_rolling_training(returns_df, features, returns_df.index[0], returns_df.index[-1])
    adjusted_covariance = sample_covariance + np.linalg.pinv(np.linalg.pinv(sample_covariance * tau) + identity_matrix.T @ np.linalg.pinv(Lambda) @ identity_matrix)
    adjusted_mean = adjusted_covariance @ (np.linalg.pinv(sample_covariance * tau) @ prior_mean + identity_matrix.T @ np.linalg.pinv(Lambda) @ (intercepts + B @ features.mean()))
    return adjusted_mean, adjusted_covariance

def optimize_mean_variance_portfolio(returns_df):
    """
    Solve the mean–variance (MV) portfolio optimization problem using Gurobi.
    Returns the normalized portfolio weights.
    """
    covariance_matrix = returns_df.cov().values
    expected_returns = returns_df.mean().values
    d = len(returns_df.columns)
    
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.setParam('DualReductions', 0)
        env.start()
        with gp.Model(env=env, name="mv_portfolio") as model:
            weights_var = model.addMVar(d, name="weights", lb=0, ub=gp.GRB.INFINITY)
            portfolio_return = weights_var @ expected_returns
            portfolio_variance = weights_var @ covariance_matrix @ weights_var
            
            if np.all(expected_returns < 0):
                return None
            else:
                model.setObjective(portfolio_variance, gp.GRB.MINIMIZE)
                model.addConstr(portfolio_return == 1)
    
            model.optimize()
            if model.status == gp.GRB.OPTIMAL:
                optimal_weights = weights_var.X
                normalized_weights = optimal_weights / optimal_weights.sum()
            elif model.status == gp.GRB.INF_OR_UNBD:
                print("Model is infeasible or unbounded.")
                normalized_weights = None
            else:
                print(f"Optimization ended with status: {model.status}")
                normalized_weights = None
            return normalized_weights

def optimize_black_litterman_portfolio(returns_df):
    """
    Solve the Black-Litterman (BL) portfolio optimization problem using Gurobi.
    Returns the normalized portfolio weights.
    """
    d = len(returns_df.columns)
    bl_mean, bl_covariance = compute_black_litterman_estimates(returns_df)
    
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.setParam('DualReductions', 0)
        env.setParam('TimeLimit', 10)
        env.start()
        with gp.Model(env=env, name="bl_portfolio") as model:
            weights_var = model.addMVar(d, name="weights", lb=0, ub=gp.GRB.INFINITY)
            portfolio_return = weights_var @ bl_mean
            portfolio_variance = weights_var @ bl_covariance @ weights_var
            
            if np.all(bl_mean < 0):
                return None
            else:
                model.setObjective(portfolio_variance, gp.GRB.MINIMIZE)
                model.addConstr(portfolio_return == 1)
            model.optimize()
            if model.status == gp.GRB.OPTIMAL:
                optimal_weights = weights_var.X
                normalized_weights = optimal_weights / optimal_weights.sum()
            elif model.status == gp.GRB.INF_OR_UNBD:
                print("Model is infeasible or unbounded.")
                normalized_weights = None
            else:
                print(f"Optimization ended with status: {model.status}")
                normalized_weights = None
            return normalized_weights

def execute_portfolio_strategy(price_df, returns_df, lookback_period):
    """
    Execute portfolio optimization over a moving window defined by the lookback period.
    Returns the strategy performance and computed portfolio weights for MV and BL models.
    """
    print(f'Running portfolio strategy with a lookback window of {lookback_period} days.')
    monthly_rebalance_index = price_df.groupby(price_df.index.strftime('%Y-%m')).head(1).index
    daily_dates = price_df.index
    mv_weights = pd.DataFrame(index=price_df.index, columns=price_df.columns)
    bl_weights = pd.DataFrame(index=price_df.index, columns=price_df.columns)
    
    strategy_returns = returns_df.copy()
    strategy_returns['portfolio_mv'] = 0.0
    strategy_returns['portfolio_bl'] = 0.0
    
    rebalance_info = []
    # Identify rebalance dates and associated stock components
    for current_date in daily_dates:
        current_index = price_df.index.get_loc(current_date)
        if current_index > lookback_period:
            if current_date in monthly_rebalance_index:
                current_date_parsed = current_date.to_pydatetime()
                for update_date in sorted(update_dates, reverse=True):
                    if datetime.strptime(update_date, "%Y-%m-%d") <= current_date_parsed:
                        current_components = sorted(components[update_date])
                        break
                rebalance_info.append((current_date, current_index, current_components))
                
    window_returns_list = []
    for info in rebalance_info:
        window_returns = returns_df[info[2]].iloc[info[1] - lookback_period:info[1]].dropna(axis=1)
        window_returns_list.append(window_returns)
    
    for window_returns in window_returns_list:
        decision_date = price_df.index[price_df.index.get_loc(window_returns.index[-1])]
        print(f'Lookback = {lookback_period}. Decision made on {decision_date}:')
        
        mv_weights.loc[decision_date, window_returns.columns] = optimize_mean_variance_portfolio(window_returns)
        mv_weights.loc[decision_date] = mv_weights.loc[decision_date].fillna(0)
        
        bl_weights.loc[decision_date, window_returns.columns] = optimize_black_litterman_portfolio(window_returns)
        bl_weights.loc[decision_date] = bl_weights.loc[decision_date].fillna(0)
        
    mv_weights = mv_weights.ffill().fillna(0)
    bl_weights = bl_weights.ffill().fillna(0)
    
    for idx in range(len(price_df) - 1):
        current_date = price_df.index[idx]
        next_date = price_df.index[idx + 1]
        strategy_returns.loc[next_date, 'portfolio_mv'] = np.sum(mv_weights.loc[current_date] * returns_df.loc[next_date])
        strategy_returns.loc[next_date, 'portfolio_bl'] = np.sum(bl_weights.loc[current_date] * returns_df.loc[next_date])
    
    return strategy_returns, mv_weights, bl_weights

if __name__ == '__main__':
    print('Starting portfolio optimization strategies.')
    
    # Execute portfolio strategies with different lookback periods
    portfolio_allocations = [
        execute_portfolio_strategy(price_data.copy(), returns_data.copy(), lookback_period=50),
        execute_portfolio_strategy(price_data.copy(), returns_data.copy(), lookback_period=80),
        # execute_portfolio_strategy(price_data.copy(), returns_data.copy(), lookback_period=100),
        # execute_portfolio_strategy(price_data.copy(), returns_data.copy(), lookback_period=120),
        # execute_portfolio_strategy(price_data.copy(), returns_data.copy(), lookback_period=150),
    ]

    # Define the portfolio variant and create the DataFrame for cumulative returns
    portfolio = 'portfolio bl'
    variant = ['portfolio mv']

    # Initialize an empty DataFrame for the benchmark and portfolio performance
    df_bl = pd.DataFrame()

    # Add the equal-weighted portfolio returns
    df_bl['EQW'] = equal_weighting_portfolio(returns_data.copy(), None)['portfolio']
    
    # Add benchmark returns
    for _benchmark in benchmarks:
        df_bl[_benchmark] = benchmark_returns_data[_benchmark]
    
    # Add the returns for each portfolio optimization (MV and BL models)
    for i, allocation in enumerate(portfolio_allocations):
        df_bl[f'MV {i+1}'] = allocation[0]['portfolio_mv']
        df_bl[f'BL {i+1}'] = allocation[0]['portfolio_bl']
    
    # Generate performance metrics report using QuantStats
    qs.reports.metrics(df_bl, mode="full", display=True)
    
    
    ### Academic Log-Scale Cumulative Plot
    
    def plot_cumulative_returns(df_bl):
        """
        Plot the cumulative returns of portfolio models and benchmark indices on a log scale.
        """
        # Clean the dataframe
        df_clean = df_bl.drop(columns=[]).fillna(0)
        log_cumulative_returns = np.log1p(df_clean).cumsum()

        # Identify Columns by Group
        bl_columns = [col for col in log_cumulative_returns.columns if "BL" in col]
        mv_columns = [col for col in log_cumulative_returns.columns if "MV" in col]
        other_columns = [
            col for col in log_cumulative_returns.columns 
            if col not in bl_columns + mv_columns
        ]

        # Define Colors and Line Styles
        bl_color_base = (0.7, 0.0, 0.0)  # Dark Red (RGB)
        mv_color_base = (0.0, 0.7, 0.0)  # Dark Green (RGB)
        other_color = 'gray'  # Constant Gray for 'Other' models

        # Set Style
        plt.style.use('seaborn-v0_8-whitegrid')  # or any style from mpl.style.available
        fig, ax = plt.subplots(figsize=(22, 15))

        # Map column names to more descriptive names for legends
        bl_legend_labels = ['BL (50d)', 'BL (80d)', 'BL (100d)', 'BL (120d)', 'BL (150d)']
        mv_legend_labels = ['MV (50d)', 'MV (80d)', 'MV (100d)', 'MV (120d)', 'MV (150d)']

        # Custom labels for "Other" columns (this can be edited)
        other_legend_labels = ['EQW: Eqaul-Weighted', 'DJIA: Dow Jones', 'SPY: S&P500']  # Customize these labels as needed

        def plot_group(columns, base_color, legend_labels=None):
            for i, col in enumerate(columns):
                if isinstance(base_color, str):  # If the color is a string like 'gray', no intensity adjustment
                    color_with_intensity = base_color
                else:  # If the color is a tuple (RGB), adjust intensity
                    # Adjust intensity by decreasing the RGB values
                    intensity = 1 - (i * 0.16)  # Decrease intensity for each subsequent line
                    color_with_intensity = tuple([c * intensity for c in base_color])  # Adjust the color intensity
                label = legend_labels[i] if legend_labels else col
                ax.plot(log_cumulative_returns.index, 
                        log_cumulative_returns[col], 
                        label=label, 
                        color=color_with_intensity,  # Use the adjusted color
                        linestyle='-', 
                        linewidth=1.5)

        plot_group(other_columns, other_color, legend_labels=other_legend_labels)  # Custom labels for 'Other' models
        plot_group(mv_columns, mv_color_base, legend_labels=mv_legend_labels)  # Dark Green to Light Green
        plot_group(bl_columns, bl_color_base, legend_labels=bl_legend_labels)  # Dark Red to Light Red

        # Customize Plot Aesthetics
        ax.set_title('Cumulative Returns (Log Scale) of Models and Benchmark Indices', fontsize=40)
        ax.set_xlabel('Date', fontsize=35)
        ax.set_ylabel('Cumulative Return (%)', fontsize=35)  # Increased font size for y-axis label
        ax.legend(fontsize=30, loc='upper left')

        ticks = ax.get_yticks()
        ax.set_yticklabels([f'{(np.exp(tick)-1)*100:.1f}%' for tick in ticks])

        ax.tick_params(axis='x', labelsize=28)  # Make y-axis tick labels larger
        ax.tick_params(axis='y', labelsize=28)  # Make y-axis tick labels larger

        ax.minorticks_on()  # Enable minor ticks
        ax.grid(which='major', color='#CCCCCC', linestyle='--')  # Major grid
        ax.grid(which='minor', color='#CCCCCC', linestyle=':')  # Minor grid
        fig.tight_layout()

        # Save and display the plot
        plt.savefig('cumulative_returns_plot.eps', format='eps')
        plt.show()

    # Call the function to generate the log-scale cumulative plot
    plot_cumulative_returns(df_bl)


    ### Asset Allocation Plot

    def plot_allocation(df_weights):
        """
        Plot the asset allocation over time with area charts.
        """
        # Fill missing values and remove negative allocations
        df_weights = df_weights.fillna(0).ffill()
        df_weights[df_weights < 0] = 0

        # Use the stock list for distinct color mapping
        d = len(stock_list)
        colormap = matplotlib.colormaps['tab20c']
        colors = [colormap(i / d) for i in range(d)[::-1]]

        # Create an area plot of the allocations
        fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
        df_weights.plot.area(ax=ax, color=colors)

        ax.set_xlabel('Date')
        ax.set_ylabel('Allocation')
        ax.set_title('Asset Allocation Over Time')

        # Place the legend outside the plot area
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels), title='Assets', bbox_to_anchor=(1, 1), loc='upper left', fontsize='small')

        plt.tight_layout()
        plt.show()
        return None

    # Plot the allocations for both MV and BL models for each lookback period
    for allocation in portfolio_allocations:
        plot_allocation(allocation[1])  # Mean-Variance allocation
        plot_allocation(allocation[2])  # Black-Litterman allocation
