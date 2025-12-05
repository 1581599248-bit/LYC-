import streamlit as st
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import akshare as ak
from datetime import datetime, timedelta
import os
import glob

# ==========================================
# 1. 全局配置
# ==========================================
st.set_page_config(
    page_title="QuantLab Pro Ultimate",
    layout="wide",
    initial_sidebar_state="expanded",
)

STRATEGY_DIR = "my_strategies"
if not os.path.exists(STRATEGY_DIR):
    os.makedirs(STRATEGY_DIR)

# ---------- 全局样式 ----------
st.markdown(
    """
<style>
    .stApp { background-color: #0e1117; }
    section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    
    /* AI 战情室 Banner */
    .ai-war-room {
        background: linear-gradient(135deg, #1e222d 0%, #1a2333 100%);
        border: 1px solid #30363d;
        border-left: 5px solid #2962ff;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .ai-title { color: #8b949e; font-size: 12px; letter-spacing: 1px; font-weight: 700; margin-bottom: 8px; }
    .ai-main { color: #fff; font-size: 20px; font-weight: 700; display: flex; align-items: center; gap: 10px; }
    .ai-desc { color: #c9d1d9; font-size: 14px; margin-top: 5px; line-height: 1.5; }
    .ai-tag { background: #238636; color: #fff; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }
    
    /* 指标卡片 */
    .metric-container {
        background-color: #1e222d; border: 1px solid #30363d; border-radius: 6px;
        padding: 10px 5px; text-align: center; margin-bottom: 8px; min-height: 90px;
    }
    .metric-label { font-size: 11px; color: #8b949e; margin-bottom: 5px; text-transform: uppercase; }
    .metric-value { font-size: 18px; font-weight: 700; color: #e1e1e1; font-family: 'Roboto Mono', monospace; }
    .metric-pos { color: #00E676 !important; }
    .metric-neg { color: #FF5252 !important; }
    
    /* 策略卡片 */
    .strat-card {
        background-color: #21262d; border: 1px solid #30363d; border-radius: 8px; 
        padding: 15px; margin-bottom: 15px; transition: 0.2s; height: 100%;
    }
    .strat-card:hover { border-color: #2962ff; transform: translateY(-3px); }
    .strat-tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; margin-right: 6px; background: #0d1117; border: 1px solid #30363d; color: #8b949e; }
    .strat-tag-active { border-color: #00E676; color: #00E676; }
    .strat-metric { font-size: 11px; color: #c9d1d9; margin-top: 4px; }
    
    .stMultiSelect label { display: none; }
    button[kind="primary"] { background-color: #2962ff !important; font-weight: 700; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🚀 QuantLab Pro: 全周期智能终端")

# ==========================================
# 2. 侧边栏：核心控制
# ==========================================
with st.sidebar:
    st.header("🎮 模式选择")
    app_mode = st.radio(
        "Mode", ["☁️ 策略超市", "🛠️ 策略工作台"], label_visibility="collapsed"
    )

    st.divider()
    st.header("⚙️ 市场数据")

    indices = {
        "🇨🇳 上证指数": {"c": "000001", "t": "cn_index"},
        "🇨🇳 沪深300": {"c": "000300", "t": "cn_index"},
        "🇨🇳 中证500": {"c": "000905", "t": "cn_index"},
        "🇨🇳 中证1000": {"c": "000852", "t": "cn_index"},
        "🇨🇳 科创50": {"c": "000688", "t": "cn_index"},
        "🇨🇳 创业板指": {"c": "399006", "t": "cn_index"},
        "🇺🇸 纳斯达克": {"c": ".IXIC", "t": "us_index"},
        "🇺🇸 标普500": {"c": ".INX", "t": "us_index"},
    }
    target = st.selectbox("标的 Asset", list(indices.keys()))

    # 只保留日/周/月，彻底砍掉分钟线
    periods = {
        "日线": "daily",
        "周线": "weekly",
        "月线": "monthly",
    }
    tf_label = st.selectbox(
        "周期 Timeframe", list(periods.keys()), index=0  # 默认日线
    )
    tf_val = periods[tf_label]

    start_dt = st.date_input("Start", datetime.now() - timedelta(days=365 * 5))
    end_dt = st.date_input("End", datetime.now())

    if st.button("🔄 刷新数据", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ==========================================
# 3. 数据引擎：只做 日/周/月
# ==========================================
@st.cache_data(ttl=60)
def get_market_data(info, tf, start, end):
    """
    精简版数据引擎：
    - 日线：A股/美股各一套
    - 周 / 月：日线重采样
    """
    code = info["c"]
    t = info["t"]
    df = None

    # 指数代码转换给 akshare 用：000001 -> sh000001  /  399006 -> sz399006
    def build_sym(c):
        return "sz" + c if c.startswith("399") else "sh" + c

    # ---------- 日线 ----------
    if tf == "daily":
        if t == "cn_index":
            sym = build_sym(code)
            try:
                df = ak.stock_zh_index_daily(symbol=sym)
                df.rename(
                    columns={
                        "date": "Date",
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "volume": "Volume",
                    },
                    inplace=True,
                )
            except:
                pass
        else:
            # 美股指数
            try:
                df = ak.index_us_stock_sina(symbol=code)
                df.rename(
                    columns={
                        "date": "Date",
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "volume": "Volume",
                    },
                    inplace=True,
                )
            except:
                pass

        if df is not None:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)

    # ---------- 周 / 月：用日线重采样 ----------
    elif tf in ["weekly", "monthly"]:
        if t == "cn_index":
            sym = build_sym(code)
            try:
                df = ak.stock_zh_index_daily(symbol=sym)
                df.rename(
                    columns={
                        "date": "Date",
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "volume": "Volume",
                    },
                    inplace=True,
                )
            except:
                pass
        else:
            try:
                df = ak.index_us_stock_sina(symbol=code)
                df.rename(
                    columns={
                        "date": "Date",
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "volume": "Volume",
                    },
                    inplace=True,
                )
            except:
                pass

        if df is not None:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)

            agg = {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
            if tf == "weekly":
                df = df.resample("W-FRI").agg(agg).dropna()
            else:  # "monthly"
                df = df.resample("M").agg(agg).dropna()

    if df is None or df.empty:
        return None

    df = df.sort_index()
    # 简单的日期过滤
    s_str = start.strftime("%Y-%m-%d")
    e_str = end.strftime("%Y-%m-%d")
    df = df.loc[s_str : e_str]
    return df


# ---------- 获取数据 ----------
with st.spinner("⏳ 正在构建全周期数据..."):
    df = get_market_data(indices[target], tf_val, start_dt, end_dt)

if df is None or df.empty:
    st.error(f"❌ 数据获取失败：{target} - {tf_label}。")
    st.warning("建议：1. 检查日期范围；2. Akshare 接口偶尔不稳定，请稍后刷新。")
    st.stop()

# ==========================================
# 4. 市场 Regime 诊断
# ==========================================
def analyze_market(df_in: pd.DataFrame):
    if len(df_in) < 50:
        return "数据不足", "观望", 0.0, "样本太少，暂以观望为主。"

    # ADX
    try:
        adx = df_in.ta.adx(14)
        adx_val = float(adx["ADX_14"].iloc[-1])
    except Exception:
        adx_val = 0.0

    price = float(df_in["Close"].iloc[-1])
    ma20 = float(df_in["Close"].rolling(20).mean().iloc[-1])
    ma60 = float(df_in["Close"].rolling(60).mean().iloc[-1])

    # ATR 波动率
    try:
        atr = float(df_in.ta.atr(14).iloc[-1])
        vol_pct = atr / price * 100
    except Exception:
        vol_pct = 0.0

    if adx_val > 25:
        if price > ma20 > ma60:
            regime = "🚀 强势多头"
            rec_type = "趋势"
            desc = "趋势明确向上，可考虑顺势持仓或回踩加仓。"
        elif price < ma20 < ma60:
            regime = "🐻 强势空头"
            rec_type = "趋势"
            desc = "中期下跌趋势，控制仓位或使用趋势空头策略。"
        else:
            regime = "🔄 宽幅震荡"
            rec_type = "震荡"
            desc = "方向不清晰但波动较大，适合通道/网格类策略。"
    else:
        regime = "🦀 窄幅盘整"
        rec_type = "震荡"
        desc = "趋势弱、波动有限，适合布林带回归或观望。"

    return regime, rec_type, float(vol_pct), desc


m_regime, rec_tag, m_vol, m_desc = analyze_market(df)

# ==========================================
# 5. 策略数据库（20 个）
# ==========================================
strategies = {
    # ================= 趋势类 =================
    "MACD趋势共振": {
        "type": "趋势",
        "code": """# MACD + RSI 趋势跟随 + 200日过滤
macd = df.ta.macd(12, 26, 9); hist = macd.iloc[:, 1]
rsi = df.ta.rsi(14)
ma200 = df.ta.sma(200)
df['Signal'] = 0.0
df.loc[(hist > 0) & (rsi > 50) & (df['Close'] > ma200), 'Signal'] = 1.0
df['Returns'] = df['Close'].pct_change()
df['Strategy_Return'] = df['Signal'].shift(1) * df['Returns']"""
    },

    "双均线系统": {
        "type": "趋势",
        "code": """# 双均线 + 长周期趋势方向过滤
s, l = 10, 60
ma_s = df.ta.sma(s); ma_l = df.ta.sma(l)
ma200 = df.ta.sma(200)
df['Signal'] = 0.0
df.loc[(ma_s > ma_l) & (df['Close'] > ma200), 'Signal'] = 1.0
df['Returns'] = df['Close'].pct_change()
df['Strategy_Return'] = df['Signal'].shift(1) * df['Returns']"""
    },

    "海龟交易法则": {
        "type": "趋势",
        "code": """# 唐奇安通道突破 + 防守止损
n_entry, n_exit = 20, 10
dc_entry = df.ta.donchian(n_entry)
dc_exit = df.ta.donchian(n_exit)
up_entry = dc_entry.iloc[:, 2]; lo_exit = dc_exit.iloc[:, 0]
df['Signal'] = 0.0
df.loc[df['Close'] > up_entry.shift(1), 'Signal'] = 1.0
df.loc[df['Close'] < lo_exit.shift(1), 'Signal'] = 0.0
df['Signal'] = df['Signal'].ffill().fillna(0)
df['Returns'] = df['Close'].pct_change()
df['Strategy_Return'] = df['Signal'].shift(1) * df['Returns']"""
    },

    "SuperTrend": {
        "type": "趋势",
        "code": """# SuperTrend + 200日方向过滤
factor = 3.0
st_val = df.ta.supertrend(10, factor)
dir_ = st_val.iloc[:, 1]   # 1=up, -1=down
ma200 = df.ta.sma(200)
df['Signal'] = 0.0
df.loc[(dir_ == 1) & (df['Close'] > ma200), 'Signal'] = 1.0
df['Returns'] = df['Close'].pct_change()
df['Strategy_Return'] = df['Signal'].shift(1) * df['Returns']"""
    },

    "均线+ADX趋势": {
        "type": "趋势",
        "code": """# 双均线 + ADX 趋势过滤
s, l, thr = 10, 60, 20
ma_s = df.ta.sma(s); ma_l = df.ta.sma(l)
adx = df.ta.adx(14)
df['Signal'] = 0.0
df.loc[(ma_s > ma_l) & (adx['ADX_14'] > thr), 'Signal'] = 1.0
df['Returns'] = df['Close'].pct_change()
df['Strategy_Return'] = df['Signal'].shift(1) * df['Returns']"""
    },

    "长周期200日趋势": {
        "type": "趋势",
        "code": """# 200日长周期趋势 + 回撤保护
ma200 = df.ta.sma(200)
peak = df['Close'].cummax()
drawdown = df['Close'] / peak - 1
max_dd = -0.25  # 允许最大回撤 -25%
df['Signal'] = 0.0
df.loc[(df['Close'] > ma200) & (drawdown > max_dd), 'Signal'] = 1.0
df['Returns'] = df['Close'].pct_change()
df['Strategy_Return'] = df['Signal'].shift(1) * df['Returns']"""
    },

    "布林带趋势突破": {
        "type": "趋势",
        "code": """# 布林带向上突破做多
n = 20
bb = df.ta.bbands(n, 2)
upper = bb.iloc[:, 2]; middle = bb.iloc[:, 1]
ma200 = df.ta.sma(200)
df['Signal'] = 0.0
df.loc[(df['Close'] > upper) & (df['Close'] > ma200), 'Signal'] = 1.0
df.loc[df['Close'] < middle, 'Signal'] = 0.0
df['Signal'] = df['Signal'].ffill().fillna(0)
df['Returns'] = df['Close'].pct_change()
df['Strategy_Return'] = df['Signal'].shift(1) * df['Returns']"""
    },

    "动量突破趋势": {
        "type": "趋势",
        "code": """# 20日动量 + 60日新高突破
lookback = 20
mom = df['Close'] / df['Close'].shift(lookback) - 1
rolling_max = df['Close'].rolling(60).max()
df['Signal'] = 0.0
df.loc[(mom > 0) & (df['Close'] >= rolling_max), 'Signal'] = 1.0
df['Returns'] = df['Close'].pct_change()
df['Strategy_Return'] = df['Signal'].shift(1) * df['Returns']"""
    },

    # ================= 震荡类 =================
    "布林带回归": {
        "type": "震荡",
        "code": """# 布林带均值回归
n = 20
bb = df.ta.bbands(n, 2)
lower = bb.iloc[:, 0]; upper = bb.iloc[:, 2]
df['Signal'] = 0.0
df.loc[df['Close'] < lower, 'Signal'] = 1.0
df.loc[df['Close'] > upper, 'Signal'] = 0.0
df['Signal'] = df['Signal'].ffill().fillna(0)
df['Returns'] = df['Close'].pct_change()
df['Strategy_Return'] = df['Signal'].shift(1) * df['Returns']"""
    },

    "RSI 极限反转": {
        "type": "震荡",
        "code": """# RSI 超买超卖反转
low, high = 30, 70
rsi = df.ta.rsi(14)
df['Signal'] = 0.0
df.loc[rsi < low, 'Signal'] = 1.0
df.loc[rsi > high, 'Signal'] = 0.0
df['Signal'] = df['Signal'].ffill().fillna(0)
df['Returns'] = df['Close'].pct_change()
df['Strategy_Return'] = df['Signal'].shift(1) * df['Returns']"""
    },

    "KD 随机指标": {
        "type": "震荡",
        "code": """# KDJ 20/80 区间反转
kdj = df.ta.kdj()
k = kdj.iloc[:, 0]; d = kdj.iloc[:, 1]
df['Signal'] = 0.0
df.loc[(k < 20) & (k > d), 'Signal'] = 1.0
df.loc[(k > 80) & (k < d), 'Signal'] = 0.0
df['Signal'] = df['Signal'].ffill().fillna(0)
df['Returns'] = df['Close'].pct_change()
df['Strategy_Return'] = df['Signal'].shift(1) * df['Returns']"""
    },

    "价格偏离均值回归": {
        "type": "震荡",
        "code": """# 价格相对60日均线的偏离回归
n = 60
ma = df.ta.sma(n)
dev = df['Close'] / ma - 1
thr = 0.1  # 10%
df['Signal'] = 0.0
df.loc[dev < -thr, 'Signal'] = 1.0
df.loc[dev > thr, 'Signal'] = 0.0
df['Signal'] = df['Signal'].ffill().fillna(0)
df['Returns'] = df['Close'].pct_change()
df['Strategy_Return'] = df['Signal'].shift(1) * df['Returns']"""
    },

    "RSI+布林组合回归": {
        "type": "震荡",
        "code": """# RSI + 布林带组合均值回归
bb = df.ta.bbands(20, 2)
lower = bb.iloc[:, 0]; upper = bb.iloc[:, 2]
rsi = df.ta.rsi(14)
df['Signal'] = 0.0
df.loc[(df['Close'] < lower) & (rsi < 35), 'Signal'] = 1.0
df.loc[(df['Close'] > upper) & (rsi > 65), 'Signal'] = 0.0
df['Signal'] = df['Signal'].ffill().fillna(0)
df['Returns'] = df['Close'].pct_change()
df['Strategy_Return'] = df['Signal'].shift(1) * df['Returns']"""
    },

    # ================= 多因子 =================
    "动量+波动率多因子": {
        "type": "多因子",
        "code": """# 20日动量 + 波动率过滤 + 长周期趋势
m_n, v_n = 20, 20
mom = df['Close'] / df['Close'].shift(m_n) - 1
ret = df['Close'].pct_change()
vol = ret.rolling(v_n).std()
vol_ma = vol.rolling(60).mean()
ma200 = df.ta.sma(200)
df['Signal'] = 0.0
df.loc[(mom > 0) & (vol < vol_ma * 1.2) & (df['Close'] > ma200), 'Signal'] = 1.0
df['Returns'] = ret
df['Strategy_Return'] = df['Signal'].shift(1) * df['Returns']"""
    },

    "趋势+动量多因子": {
        "type": "多因子",
        "code": """# 双均线趋势 + 60日动量共振
ma_short = df.ta.sma(20)
ma_long = df.ta.sma(100)
mom = df['Close'] / df['Close'].shift(60) - 1
df['Signal'] = 0.0
df.loc[(ma_short > ma_long) & (mom > 0), 'Signal'] = 1.0
df['Returns'] = df['Close'].pct_change()
df['Strategy_Return'] = df['Signal'].shift(1) * df['Returns']"""
    },

    "成交量放大动量因子": {
        "type": "多因子",
        "code": """# 价格动量 + 成交量放大
mom = df['Close'] / df['Close'].shift(20) - 1
vol = df['Volume']
vol_ma = vol.rolling(60).mean()
df['Signal'] = 0.0
df.loc[(mom > 0) & (vol > vol_ma * 1.2), 'Signal'] = 1.0
df['Returns'] = df['Close'].pct_change()
df['Strategy_Return'] = df['Signal'].shift(1) * df['Returns']"""
    },

    "反转+波动率多因子": {
        "type": "多因子",
        "code": """# 短期反转 + 低波动过滤
short_ret = df['Close'].pct_change(5)
ret = df['Close'].pct_change()
vol = ret.rolling(20).std()
vol_ma = vol.rolling(60).mean()
df['Signal'] = 0.0
df.loc[(short_ret < 0) & (vol < vol_ma), 'Signal'] = 1.0
df['Returns'] = ret
df['Strategy_Return'] = df['Signal'].shift(1) * df['Returns']"""
    },

    # ================= 资产配置 / 风控类 =================
    "波动率目标仓位": {
        "type": "资产配置",
        "code": """# 按波动率目标调整仓位（0~1 连续仓位）
target_vol = 0.15
ret = df['Close'].pct_change()
realized_vol = ret.rolling(60).std()
df['Signal'] = 0.0
df.loc[realized_vol > 0, 'Signal'] = np.minimum(1.0, target_vol / realized_vol)
df['Signal'] = df['Signal'].fillna(0.0)
df['Returns'] = ret
df['Strategy_Return'] = df['Signal'].shift(1) * df['Returns']"""
    },

    "简单趋势风控": {
        "type": "资产配置",
        "code": """# 买入持有 + MA 风控（跌破均线空仓）
ma = df.ta.sma(120)
df['Signal'] = 1.0
df.loc[df['Close'] < ma, 'Signal'] = 0.0
df['Signal'] = df['Signal'].ffill().fillna(1.0)
df['Returns'] = df['Close'].pct_change()
df['Strategy_Return'] = df['Signal'].shift(1) * df['Returns']"""
    },

    "均线阶梯加仓": {
        "type": "资产配置",
        "code": """# 价格相对120日均线的阶梯式加仓
ma = df.ta.sma(120)
dev = df['Close'] / ma - 1
df['Signal'] = 0.0
df.loc[dev > 0, 'Signal'] = 0.5
df.loc[dev > 0.05, 'Signal'] = 0.8
df.loc[dev > 0.10, 'Signal'] = 1.0
df['Signal'] = df['Signal'].fillna(0.0)
df['Returns'] = df['Close'].pct_change()
df['Strategy_Return'] = df['Signal'].shift(1) * df['Returns']"""
    },
}

if "active_code" not in st.session_state:
    st.session_state["active_code"] = strategies["MACD趋势共振"]["code"]

# ==========================================
# 6. 公共：基准年化 & 策略回测函数
# ==========================================
def compute_metrics_from_returns(ret: pd.Series):
    ret = ret.fillna(0)
    if len(ret) < 2:
        return 0.0, 0.0, 0.0
    eq = (1 + ret).cumprod()
    tot = float(eq.iloc[-1] - 1)
    days = max((ret.index[-1] - ret.index[0]).days, 1)
    ann = (1 + tot) ** (365 / days) - 1
    vol = float(ret.std() * np.sqrt(252))
    sharpe = ann / vol if vol != 0 else 0.0
    return ann, sharpe, tot


bench_ret = df["Close"].pct_change()
bench_ann, bench_sharpe, bench_tot = compute_metrics_from_returns(bench_ret)


def backtest_strategy_for_card(code_str: str, base_df: pd.DataFrame):
    """
    在“策略超市”里用：执行一次策略代码，返回年化、夏普、alpha。
    """
    local_env = {"df": base_df.copy(), "np": np, "pd": pd, "ta": ta}
    try:
        exec(code_str, {}, local_env)
        rdf = local_env["df"]
        if "Strategy_Return" not in rdf.columns or "Returns" not in rdf.columns:
            return None
        strat_ret = rdf["Strategy_Return"].fillna(0)
        ann, sharpe, _ = compute_metrics_from_returns(strat_ret)
        # 基准用策略里定义的 Returns（避免和外部 df 不一致）
        bench_ret_inner = rdf["Returns"].fillna(0)
        b_ann, _, _ = compute_metrics_from_returns(bench_ret_inner)
        alpha = ann - b_ann
        return {"ann": ann, "sharpe": sharpe, "alpha": alpha}
    except Exception:
        return None


# 在当前标的 + 时间区间上，预先跑一遍所有策略
strategy_metrics = {}
for name, data in strategies.items():
    strategy_metrics[name] = backtest_strategy_for_card(data["code"], df)

# ==========================================
# 7. 界面逻辑
# ==========================================

# ---------- 模式 A：策略超市 ----------
if app_mode == "☁️ 策略超市":
    st.markdown(
        f"""
    <div class="ai-war-room">
        <div class="ai-title">AI MARKET INTELLIGENCE</div>
        <div class="ai-main">
            {m_regime} 
            <span class="ai-tag">{tf_label}</span>
        </div>
        <div class="ai-desc">
            当前标的波动率约 <b>{m_vol:.2f}%</b>。{m_desc}<br>
            基于当前环境，优先关注：<b>{rec_tag}</b> 类策略。
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    i = 0

    # 排序规则：先按是否是推荐类型，再按策略年化收益（从高到低）
    def sort_key(item):
        name, data = item
        m = strategy_metrics.get(name)
        ann = m["ann"] if m else -999
        base = 0
        if data["type"] == rec_tag:
            base = -2
        elif data["type"] in ["多因子", "资产配置"]:
            base = -1
        else:
            base = 0
        # base 越小优先级越高，ann 越大越靠前
        return (base, -ann)

    sorted_strats = sorted(strategies.items(), key=sort_key)

    for name, data in sorted_strats:
        with cols[i % 3]:
            is_rec = data["type"] == rec_tag
            border = "2px solid #00e676" if is_rec else "1px solid #30363d"
            tag_cls = "strat-tag strat-tag-active" if is_rec else "strat-tag"

            m = strategy_metrics.get(name)
            if m:
                ann_str = f"{m['ann']*100:,.2f}%"
                alpha_str = f"{m['alpha']*100:,.2f}%"
                sharpe_str = f"{m['sharpe']:.2f}"
                alpha_cls = "metric-pos" if m["alpha"] > 0 else ("metric-neg" if m["alpha"] < 0 else "")
                metrics_html = f"""
                    <div class="strat-metric">
                        年化: <b>{ann_str}</b>，
                        夏普: <b>{sharpe_str}</b><br>
                        Alpha: <span class="{alpha_cls}">{alpha_str}</span>
                    </div>
                """
            else:
                metrics_html = '<div class="strat-metric">回测失败，请在工作台中检查代码。</div>'

            st.markdown(
                f"""
            <div class="strat-card" style="border:{border}">
                <div style="font-weight:bold;color:#fff;margin-bottom:8px;">{name}</div>
                <span class="{tag_cls}">{data['type']}</span>
                {metrics_html}
            </div>
            """,
                unsafe_allow_html=True,
            )

            if st.button(f"📥 加载：{name}", key=f"btn_{name}", use_container_width=True):
                st.session_state["active_code"] = data["code"]
                st.toast(f"已加载策略：{name}", icon="✅")
        i += 1

# ---------- 模式 B：策略工作台 ----------
else:
    st.header("🛠️ 策略工作台")

    # ---- 代码编辑区 ----
    with st.expander("📝 代码编辑器", expanded=True):
        c1, c2 = st.columns([5, 1])
        user_code = c1.text_area(
            "Code",
            st.session_state["active_code"],
            height=260,
            label_visibility="collapsed",
        )

        # 文件操作
        files = glob.glob(os.path.join(STRATEGY_DIR, "*.py"))
        f_names = [os.path.basename(f) for f in files]
        f_names.insert(0, "🆕 新建")
        sel_file = c2.selectbox("File", f_names, label_visibility="collapsed")

        if sel_file != "🆕 新建" and c2.button("📂 读取"):
            with open(
                os.path.join(STRATEGY_DIR, sel_file), "r", encoding="utf-8"
            ) as f:
                st.session_state["active_code"] = f.read()
                st.rerun()

        save_name = c2.text_input("Name", "strat.py", label_visibility="collapsed")
        if c2.button("💾 保存"):
            with open(
                os.path.join(STRATEGY_DIR, save_name), "w", encoding="utf-8"
            ) as f:
                f.write(user_code)
            st.success("Saved")
            st.rerun()

    # ---- 执行区域 ----
    l_vars = {"df": df.copy(), "np": np, "pd": pd, "ta": ta, "st": st}
    run_pressed = st.button(
        "🚀 运行回测 (Run Analysis)", type="primary", use_container_width=True
    )

    should_run = False
    if "st.slider" in user_code:
        # 如果你自己在代码里写了 slider，这里会先跑一遍生成控件，再按按钮真正回测
        st.markdown("##### 🎛️ 动态参数")
        exec(user_code, globals(), l_vars)
        should_run = True
    elif run_pressed:
        exec(user_code, globals(), l_vars)
        should_run = True

    if should_run:
        res_df = l_vars.get("df")

        if res_df is not None and "Strategy_Return" in res_df.columns:
            # ---------- 指标计算 ----------
            res_df = res_df.copy()
            res_df["Strategy_Return"] = res_df["Strategy_Return"].fillna(0)
            res_df["Returns"] = res_df["Returns"].fillna(0)

            eq = (1 + res_df["Strategy_Return"]).cumprod()
            bn = (1 + res_df["Returns"]).cumprod()

            tot = float(eq.iloc[-1] - 1)
            ben = float(bn.iloc[-1] - 1)
            days = max((res_df.index[-1] - res_df.index[0]).days, 1)
            ann = (1 + tot) ** (365 / days) - 1

            dd = (eq - eq.cummax()) / eq.cummax()
            mdd = float(dd.min())

            vol = float(res_df["Strategy_Return"].std() * np.sqrt(252))
            sharpe = float(ann / vol) if vol != 0 else 0.0

            # 逐笔交易统计
            trades = []
            act = res_df["Signal"].diff()
            entries = res_df[act > 0].index
            exits = res_df[act < 0].index

            p_e = 0
            while p_e < len(entries):
                t_in = entries[p_e]
                later_exits = exits[exits > t_in]
                if len(later_exits) > 0:
                    t_out = later_exits[0]
                    p1 = float(res_df.loc[t_in, "Close"])
                    p2 = float(res_df.loc[t_out, "Close"])
                    trades.append((p2 - p1) / p1)
                    p_e += 1
                else:
                    break

            trades = np.array(trades)
            n_t = len(trades)

            if n_t > 0:
                wins = trades[trades > 0]
                loss = trades[trades <= 0]
                w_rate = len(wins) / n_t
                avg_w = wins.mean() if len(wins) > 0 else 0.0
                avg_l = loss.mean() if len(loss) > 0 else 0.0
                pl_r = abs(avg_w / avg_l) if avg_l != 0 else 0.0
                max_w = trades.max()
                max_l = trades.min()
                kelly = w_rate - (1 - w_rate) / pl_r if pl_r != 0 else 0.0
            else:
                w_rate = avg_w = avg_l = pl_r = max_w = max_l = kelly = 0.0

            # ---------- 指标卡片展示 ----------
            st.divider()

            def card(label, value, fmt="{:.2%}", color=True):
                cls = "metric-value"
                if color:
                    if value > 0:
                        cls += " metric-pos"
                    elif value < 0:
                        cls += " metric-neg"
                return f"""
                <div class="metric-container">
                    <div class="metric-label">{label}</div>
                    <div class="{cls}">{fmt.format(value)}</div>
                </div>
                """

            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(card("累计收益 Total", tot), unsafe_allow_html=True)
            c2.markdown(card("年化收益 Ann", ann), unsafe_allow_html=True)
            c3.markdown(card("基准收益 Bench", ben), unsafe_allow_html=True)
            c4.markdown(card("超额收益 Alpha", tot - ben), unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(card("最大回撤 MaxDD", mdd), unsafe_allow_html=True)
            c2.markdown(card("夏普比率 Sharpe", sharpe, "{:.2f}"), unsafe_allow_html=True)
            c3.markdown(card("波动率 Vol", vol), unsafe_allow_html=True)
            c4.markdown(card("凯利仓位 Kelly", kelly), unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(card("胜率 WinRate", w_rate), unsafe_allow_html=True)
            c2.markdown(card("盈亏比 P/L Ratio", pl_r, "{:.2f}"), unsafe_allow_html=True)
            c3.markdown(card("交易次数 Trades", n_t, "{:.0f}", False), unsafe_allow_html=True)
            c4.markdown(
                card("当前持仓 Position", float(res_df["Signal"].iloc[-1]), "{:.2f}", False),
                unsafe_allow_html=True,
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(card("平均盈利 AvgWin", avg_w), unsafe_allow_html=True)
            c2.markdown(card("平均亏损 AvgLoss", avg_l, color=False), unsafe_allow_html=True)
            c3.markdown(card("最大单笔盈 MaxWin", max_w), unsafe_allow_html=True)
            c4.markdown(card("最大单笔亏 MaxLoss", max_l, color=False), unsafe_allow_html=True)

            # ---------- P&L 曲线 ----------
            st.write("")
            st.markdown("##### 📉 累计收益率")
            fig_pl = go.Figure()
            fig_pl.add_trace(
                go.Scatter(
                    x=res_df.index,
                    y=(eq - 1) * 100,
                    name="策略%",
                    line=dict(width=2.5),
                )
            )
            fig_pl.add_trace(
                go.Scatter(
                    x=res_df.index,
                    y=(bn - 1) * 100,
                    name="基准%",
                    line=dict(width=1.5, dash="dash"),
                )
            )
            fig_pl.add_hline(y=0, line_dash="dash", opacity=0.5)
            fig_pl.update_layout(
                height=350,
                template="plotly_dark",
                paper_bgcolor="#1e222d",
                plot_bgcolor="#1e222d",
                margin=dict(l=0, r=0, t=20, b=20),
            )
            st.plotly_chart(fig_pl, use_container_width=True)

            # ---------- K 线 + 指标 ----------
            with st.expander("🛠️ 图表指标"):
                cc1, cc2 = st.columns(2)
                ov = cc1.multiselect("主图", ["MA20", "BOLL"], ["MA20", "BOLL"])
                sb = cc2.multiselect(
                    "副图", ["Volume", "MACD", "RSI", "KDJ"], ["Volume", "MACD"]
                )

            num_subs = len(sb)
            if num_subs == 0:
                row_heights = [1]
            else:
                row_heights = [0.6] + [0.4 / num_subs] * num_subs

            subplot_titles = ["Price"] + [f"{s}" for s in sb]

            fig = make_subplots(
                rows=1 + num_subs,
                cols=1,
                shared_xaxes=True,
                row_heights=row_heights,
                subplot_titles=subplot_titles,
                vertical_spacing=0.03,
            )

            # ====== 主图 K线 ======
            fig.add_trace(
                go.Candlestick(
                    x=res_df.index,
                    open=res_df["Open"],
                    high=res_df["High"],
                    low=res_df["Low"],
                    close=res_df["Close"],
                    name="K",
                ),
                row=1,
                col=1,
            )

            # 主图均线
            if "MA20" in ov:
                fig.add_trace(
                    go.Scatter(
                        x=res_df.index,
                        y=res_df["Close"].rolling(20).mean(),
                        name="MA20",
                        line=dict(width=1.5),
                    ),
                    row=1,
                    col=1,
                )

            # 主图布林带
            if "BOLL" in ov:
                bb = res_df.ta.bbands(20)
                if bb is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=res_df.index, y=bb.iloc[:, 2], name="UP", line=dict(width=1)
                        ),
                        row=1,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=res_df.index, y=bb.iloc[:, 0], name="LO", line=dict(width=1)
                        ),
                        row=1,
                        col=1,
                    )

            # Buy / Sell 标签
            chg = res_df["Signal"].diff()
            b_pts = res_df[chg > 0]
            s_pts = res_df[chg < 0]

            if not b_pts.empty:
                fig.add_trace(
                    go.Scatter(
                        x=b_pts.index,
                        y=b_pts["Low"] * 0.995,
                        mode="markers+text",
                        marker=dict(symbol="triangle-up", size=12),
                        text=["BUY"] * len(b_pts),
                        textposition="bottom center",
                        name="Buy",
                    ),
                    row=1,
                    col=1,
                )

            if not s_pts.empty:
                fig.add_trace(
                    go.Scatter(
                        x=s_pts.index,
                        y=s_pts["High"] * 1.005,
                        mode="markers+text",
                        marker=dict(symbol="triangle-down", size=12),
                        text=["SELL"] * len(s_pts),
                        textposition="top center",
                        name="Sell",
                    ),
                    row=1,
                    col=1,
                )

            # ====== 副图 ======
            for i, ind in enumerate(sb):
                r = i + 2
                if ind == "Volume":
                    fig.add_trace(
                        go.Bar(
                            x=res_df.index,
                            y=res_df["Volume"],
                            name="Vol",
                        ),
                        row=r,
                        col=1,
                    )
                elif ind == "RSI":
                    fig.add_trace(
                        go.Scatter(
                            x=res_df.index,
                            y=res_df.ta.rsi(),
                            name="RSI",
                            line=dict(width=1),
                        ),
                        row=r,
                        col=1,
                    )
                elif ind == "MACD":
                    m = res_df.ta.macd()
                    if m is not None:
                        fig.add_trace(
                            go.Bar(
                                x=res_df.index,
                                y=m.iloc[:, 1],
                                name="Hist",
                            ),
                            row=r,
                            col=1,
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=res_df.index,
                                y=m.iloc[:, 0],
                                name="MACD",
                                line=dict(width=1),
                            ),
                            row=r,
                            col=1,
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=res_df.index,
                                y=m.iloc[:, 2],
                                name="Signal",
                                line=dict(width=1),
                            ),
                            row=r,
                            col=1,
                        )
                elif ind == "KDJ":
                    kdj = res_df.ta.kdj()
                    fig.add_trace(
                        go.Scatter(
                            x=res_df.index,
                            y=kdj.iloc[:, 0],
                            name="K",
                            line=dict(width=1),
                        ),
                        row=r,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=res_df.index,
                            y=kdj.iloc[:, 1],
                            name="D",
                            line=dict(width=1),
                        ),
                        row=r,
                        col=1,
                    )

            # ====== 主图光标逻辑：自由移动 + 虚线 + 右侧价格 ======
            fig.update_layout(
                height=600 + num_subs * 140,
                template="plotly_dark",
                paper_bgcolor="#131722",
                plot_bgcolor="#131722",
                margin=dict(t=40, b=20, l=0, r=60),
                xaxis_rangeslider_visible=False,
                hovermode="x",      # 沿 x 方向联动
                hoverdistance=0,    # 鼠标一到就触发
                spikedistance=0,    # 光标线紧贴鼠标
            )

            # 所有 x 轴：竖直虚线
            fig.update_xaxes(
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                spikethickness=1,
                spikedash="dot",
            )

            # 所有 y 轴：水平虚线 + 右侧价格标签
            fig.update_yaxes(
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                spikethickness=1,
                spikedash="dot",
                showline=True,
                ticks="outside",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"scrollZoom": True, "displayModeBar": True},
            )

            # ---------- AI 诊断 Prompt ----------
            st.write("---")
            st.subheader("🤖 AI 智能诊断")
            prompt = (
                f"策略诊断：标的 {target}，周期 {tf_label}\n"
                f"累计收益：{tot:.2%}\n"
                f"年化收益：{ann:.2%}\n"
                f"夏普：{sharpe:.2f}\n"
                f"胜率：{w_rate:.2%}\n"
                f"交易次数：{n_t}\n"
                f"最大回撤：{mdd:.2%}\n"
                f"请分析该策略优劣，并给出可执行的改进建议。"
            )
            st.info("👇 复制下面这段，直接丢给大模型就可以让它帮你诊断策略：")
            st.code(prompt, language="text")

        else:
            st.error("❌ 错误：未计算 'Strategy_Return'，请检查策略代码。")