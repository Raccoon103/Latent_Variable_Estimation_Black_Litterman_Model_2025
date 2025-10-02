
import pandas as pd
import ta

def compute_technical_indicators(stock_df: pd.DataFrame, stock_label: str = "") -> pd.DataFrame:
    """Compute a set of technical indicators for a given stock's OHLCV data."""
    stock_df = stock_df.dropna()
    if stock_df.empty:
        return pd.DataFrame()

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

    boll = ta.volatility.BollingerBands(close=stock_df['Close'])
    obv = ta.volume.OnBalanceVolumeIndicator(close=stock_df['Close'], volume=stock_df['Volume']).on_balance_volume()
    obv_norm = (obv - obv.min()) / (obv.max() - obv.min()) if obv.max() != obv.min() else obv*0

    suffix = f"_{stock_label}" if stock_label else ""
    indicators = pd.DataFrame({
        f'ATR{suffix}': atr,
        f'ADX{suffix}': adx,
        f'EMA{suffix}': ema,
        f'MACD{suffix}': macd,
        f'SMA{suffix}': sma,
        f'RSI{suffix}': rsi,
        f'BB_Upper{suffix}': boll.bollinger_hband(),
        f'BB_Lower{suffix}': boll.bollinger_lband(),
        f'OBV_Norm{suffix}': obv_norm
    })
    indicators = indicators.dropna()
    indicators = indicators.loc[(indicators != 0).all(axis=1)]
    return indicators
