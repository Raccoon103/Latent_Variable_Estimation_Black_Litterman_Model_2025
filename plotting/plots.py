# plotting/plot.py  ← 若你的檔名是 plots.py 就相同覆蓋
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from typing import Iterable, Optional

# ---- 共用樣式 ----
def _apply_style():
    plt.style.use('seaborn-v0_8-whitegrid')

# =============== 新增：線性尺度 累積報酬（股票 + 基準） ===============
def plot_linear_cumulative_assets(
    asset_returns: pd.DataFrame,
    benchmark_returns: Optional[pd.DataFrame] = None,
    save_path: str = "stocks_benchmarks_cumulative.png",
    legend_title: str = "Assets",
    subset_assets: Optional[Iterable[str]] = None,
    figsize=(20, 8),
):
    """
    等價於舊版這段：
        (1 + returns_data).cumprod().plot(ax=ax)
        (1 + benchmark_returns_data).cumprod().plot(ax=ax)

    會輸出單張圖（線性尺度），把股票與基準一起畫在同一張圖上。
    - asset_returns: 各股票的日報酬 DataFrame（columns=資產）
    - benchmark_returns: 基準的日報酬 DataFrame（columns=^DJI, SPY 等）
    - subset_assets: 若提供，只畫其中的資產（避免太多線）
    """
    if asset_returns is None or asset_returns.empty:
        # 沒有資產報酬就不畫
        return

    # 可選擇只畫部分資產，避免線太多
    assets = asset_returns.copy()
    if subset_assets:
        subset_assets = [c for c in subset_assets if c in assets.columns]
        if subset_assets:
            assets = assets[subset_assets]

    # 線性尺度累積報酬（成長指數）
    cum_assets = (1 + assets).cumprod()
    cum_bench = None
    if benchmark_returns is not None and not benchmark_returns.empty:
        cum_bench = (1 + benchmark_returns).cumprod()

    _apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    # 畫資產
    cum_assets.plot(ax=ax, linewidth=1.2)

    # 疊加基準
    if cum_bench is not None:
        cum_bench.plot(ax=ax, linewidth=1.8)

    # 舊版擺法：圖例放外面
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, title=legend_title)
    if legend and legend.get_title():
        plt.setp(legend.get_title(), fontsize=12)

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Cumulative Return (×)', fontsize=12)  # 直接用成長倍數，與舊版一致
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout(rect=[0, 0, 0.75, 1])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# =============== 你原本的「學術風（對數尺度）」 ===============
def plot_cumulative_returns(df_ret: pd.DataFrame, save_path: str = 'cumulative_returns_plot.png'):
    """
    把多個策略/基準的日報酬畫成『對數尺度累積報酬』。
    df_ret 需包含：EQW、基準（如 ^DJI, SPY）、以及 MV/BL 欄位。
    """
    if df_ret is None or df_ret.empty:
        return

    df_clean = df_ret.fillna(0)
    log_cum = np.log1p(df_clean).cumsum()

    bl_cols = [c for c in log_cum.columns if 'BL' in c]
    mv_cols = [c for c in log_cum.columns if 'MV' in c]
    other_cols = [c for c in log_cum.columns if c not in bl_cols + mv_cols]

    _apply_style()
    fig, ax = plt.subplots(figsize=(22, 15))

    def plot_group(cols, color, labels=None):
        for i, c in enumerate(cols):
            col = color if isinstance(color, str) else tuple([cc * (1 - i * 0.16) for cc in color])
            lab = (labels[i] if labels and i < len(labels) else c)
            ax.plot(log_cum.index, log_cum[c], label=lab, color=col, linestyle='-', linewidth=1.5)

    # 依舊版預設的順序與標籤，但 EQW 使用獨特顏色
    eqw_cols = [c for c in other_cols if 'EQW' in c]
    benchmark_cols = [c for c in other_cols if c not in eqw_cols]
    
    plot_group(eqw_cols, 'blue', labels=['EQW'])
    plot_group(benchmark_cols, 'gray', labels=['DJIA', 'SPY'])
    plot_group(mv_cols, (0.0, 0.7, 0.0), labels=['MV (50d)', 'MV (80d)', 'MV (100d)', 'MV (120d)', 'MV (150d)'])
    plot_group(bl_cols, (0.7, 0.0, 0.0), labels=['BL (50d)', 'BL (80d)', 'BL (100d)', 'BL (120d)', 'BL (150d)'])

    ax.set_title('Cumulative Returns (Log Scale)', fontsize=40)
    ax.set_xlabel('Date', fontsize=35)
    ax.set_ylabel('Cumulative Return (%)', fontsize=35)
    ax.legend(fontsize=30, loc='upper left')

    ticks = ax.get_yticks()
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{(np.exp(t)-1)*100:.1f}%" for t in ticks])

    ax.tick_params(axis='x', labelsize=28)
    ax.tick_params(axis='y', labelsize=28)
    ax.minorticks_on()
    ax.grid(which='major', color='#CCCCCC', linestyle='--')
    ax.grid(which='minor', color='#CCCCCC', linestyle=':')
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close(fig)

# =============== 你原本的「資產配置面積圖」 ===============
def plot_allocation(df_w: pd.DataFrame, save_path: str = 'allocation_plot.png'):
    """
    畫資產配置（面積圖）。預設會把負權重截成 0，且自動 forward-fill。
    """
    if df_w is None or df_w.empty:
        return

    df_w = df_w.fillna(0).ffill()
    df_w[df_w < 0] = 0

    d = len(df_w.columns)
    colormap = matplotlib.colormaps['tab20c']
    colors = [colormap(i / max(d, 1)) for i in range(d)[::-1]]

    fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
    df_w.plot.area(ax=ax, color=colors)

    ax.set_xlabel('Date')
    ax.set_ylabel('Allocation')
    ax.set_title('Asset Allocation Over Time')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(list(reversed(handles)), list(reversed(labels)),
              title='Assets', bbox_to_anchor=(1, 1), loc='upper left', fontsize='small')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# =============== 新增：資產配置流轉率（Turnover Rate） ===============
def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    """
    計算投資組合的流轉率（turnover rate）。
    流轉率 = sum(|w_t - w_{t-1}|) / 2
    
    Args:
        weights: 資產配置權重的 DataFrame（index=日期, columns=資產）
    
    Returns:
        pd.Series: 每日的流轉率
    """
    return weights.diff().abs().sum(axis=1) / 2

def plot_turnover_rate(
    weights: pd.DataFrame, 
    save_path: str = 'turnover_rate_plot.png',
    figsize=(24, 12)
):
    """
    畫出投資組合的流轉率圖表，包含平均流轉率線。
    
    Args:
        weights: 資產配置權重的 DataFrame（index=日期, columns=資產）
        save_path: 儲存路徑
        figsize: 圖表大小
    """
    if weights is None or weights.empty:
        print("Warning: No weights data provided for turnover rate plot")
        return
    
    # 計算流轉率
    turnover = compute_turnover(weights)
    
    # 計算平均流轉率（排除 0 值，因為第一天沒有前一天可比較）
    avg_turnover = turnover[turnover != 0].mean()
    
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # 畫流轉率線
    ax.plot(turnover.index, turnover, label='Turnover Rate', color='royalblue', linewidth=1.5)
    
    # 加上平均流轉率線
    ax.axhline(avg_turnover, color='firebrick', linestyle='--', linewidth=2, 
               label=f'Average Turnover: {avg_turnover:.4f}')
    
    # 設定標題與標籤
    ax.set_title('Portfolio Turnover Rate Over Time', fontsize=40)
    ax.set_xlabel('Date', fontsize=35)
    ax.set_ylabel('Turnover Rate', fontsize=35)
    ax.legend(fontsize=30, loc='upper left')
    
    # Y軸格式化為百分比
    ticks = ax.get_yticks()
    ax.set_yticklabels([f'{tick*100:.1f}%' for tick in ticks])
    ax.tick_params(axis='x', labelsize=28)
    ax.tick_params(axis='y', labelsize=28)
    
    # 加上格線
    ax.minorticks_on()
    ax.grid(which='major', color='#CCCCCC', linestyle='--')
    ax.grid(which='minor', color='#CCCCCC', linestyle=':')
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# =============== 類別式包裝（相容你的原始設計） ===============
class PortfolioTurnover:
    """
    投資組合流轉率分析類別，相容於你原始的設計。
    """
    def __init__(self, portfolio_data: pd.DataFrame):
        self.portfolio_data = portfolio_data
        self.turnover = self.compute_turnover(portfolio_data)
        self.avg_turnover = self.turnover[self.turnover != 0].mean()

    def compute_turnover(self, weights: pd.DataFrame) -> pd.Series:
        """計算投資組合的流轉率。"""
        return compute_turnover(weights)

    def plot_turnover_rate(self, filename='turnover_rate_plot.eps'):
        """畫出流轉率圖表並儲存。"""
        plot_turnover_rate(self.portfolio_data, save_path=filename)
