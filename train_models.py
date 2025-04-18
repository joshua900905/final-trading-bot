# -*- coding: utf-8 -*-
# train_models.py - Script for Training Individual Stock Models

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import time
import os
import math
from collections import defaultdict
import traceback # Import traceback for better error printing

# --- Stable Baselines 3 Import ---
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback # Optional
    # from stable_baselines3.common.callbacks import ProgressBarCallback # Optional progress bar
except ImportError:
    print("錯誤：找不到 stable_baselines3。請運行 'pip install stable_baselines3'")
    exit()

# --- Stock API Class ---
class Stock_API:
    """
    用於與股票交易模擬平台 API 互動的類別。
    包含獲取股票資訊、查詢用戶持股、提交買賣預約單的功能。
    (已修改 Buy/Sell Stock 以處理 "張" 單位)
    """
    def __init__(self, account, password):
        self.account = account
        self.password = password
        self.base_url = 'http://140.116.86.242:8081/stock/api/v1'

    def Get_Stock_Informations(self, stock_code, start_date, stop_date):
        information_url = (f"{self.base_url}/api_get_stock_info_from_date_json/"
                           f"{stock_code}/{start_date}/{stop_date}")
        try:
            response = requests.get(information_url, timeout=15)
            response.raise_for_status()
            result = response.json()
            if result.get('result') == 'success':
                data = result.get('data', [])
                return data if isinstance(data, list) else []
            else:
                 print(f"API 錯誤 (Get_Stock_Informations - {stock_code}, {start_date}-{stop_date}): {result.get('status', '未知狀態')}")
                 return []
        except requests.exceptions.Timeout:
            print(f"請求超時 (Get_Stock_Informations - {stock_code})")
            return []
        except requests.exceptions.RequestException as e:
            print(f"網路錯誤 (Get_Stock_Informations - {stock_code}): {e}")
            return []
        except Exception as e:
            print(f"處理股票資訊時出錯 (Get_Stock_Informations - {stock_code}): {e}")
            return []

    def Get_User_Stocks(self):
        # Note: This might not be strictly needed during isolated training,
        # but the StockTradingEnv might use it for confirmation simulation.
        data = {'account': self.account, 'password': self.password}
        search_url = f'{self.base_url}/get_user_stocks'
        try:
            response = requests.post(search_url, data=data, timeout=15)
            response.raise_for_status()
            result = response.json()
            if result.get('result') == 'success':
                holdings_data = result.get('data', [])
                if isinstance(holdings_data, list):
                    processed_holdings = []
                    for stock in holdings_data:
                         try:
                              processed_stock = {
                                   'stock_code': str(stock.get('stock_code')),
                                   'shares': int(stock.get('shares', 0)),
                                   'price': float(stock.get('price', 0.0)),
                                   'amount': float(stock.get('amount', 0.0))
                              }
                              processed_holdings.append(processed_stock)
                         except (ValueError, TypeError) as e:
                              print(f"處理持股數據類型轉換錯誤 for {stock.get('stock_code')}: {e}")
                    return processed_holdings
                else:
                    #print(f"警告: Get_User_Stocks 返回非列表數據: {holdings_data}")
                    return []
            else:
                # During training, API errors might be less critical if the env handles them
                # print(f"API 錯誤 (Get_User_Stocks): {result.get('status', '未知狀態')}")
                return []
        except requests.exceptions.Timeout:
            # print("請求超時 (Get_User_Stocks)")
            return []
        except requests.exceptions.RequestException as e:
            # print(f"網路錯誤 (Get_User_Stocks): {e}")
            return []
        except Exception as e:
            # print(f"處理用戶持股時出錯 (Get_User_Stocks): {e}")
            return []
        return [] # Return empty list on failure during training simulation

    def Buy_Stock(self, stock_code, stock_shares, stock_price):
        """提交購買股票預約單 (訓練時可能僅模擬成功)"""
        stock_shares = int(stock_shares)
        if stock_shares <= 0 or stock_shares % 1000 != 0:
            #print(f"模擬買單股數錯誤 ({stock_shares} 股 {stock_code})。")
            return False # Simulate failure
        sheets = stock_shares / 1000
        #print(f"模擬提交買單: {sheets:.0f} 張 ({stock_shares} 股) {stock_code} @ 目標價 {stock_price:.2f}")
        # In training, often simplify and assume API call succeeds
        return True

    def Sell_Stock(self, stock_code, stock_shares, stock_price):
        """提交賣出股票預約單 (訓練時可能僅模擬成功)"""
        stock_shares = int(stock_shares)
        if stock_shares <= 0 or stock_shares % 1000 != 0:
            #print(f"模擬賣單股數錯誤 ({stock_shares} 股 {stock_code})。")
            return False # Simulate failure
        sheets = stock_shares / 1000
        #print(f"模擬提交賣單: {sheets:.0f} 張 ({stock_shares} 股) {stock_code} @ 目標價 {stock_price:.2f}")
        # In training, often simplify and assume API call succeeds
        return True

# --- StockTradingEnv Class (Single-Stock Training Environment) ---
#     ==========================================================
#     !!! 以下是完整的 StockTradingEnv 類別定義，用於訓練獨立模型 !!!
#     ==========================================================
class StockTradingEnv(gym.Env):
    """
    用於獨立訓練單支股票模型的 Gymnasium 環境。
    遵循「盤前下單、盤中不動、盤後確認」的模擬規則。
    (訓練時，API交互和成交確認會被簡化)
    """
    metadata = {'render_modes': ['human', None], 'render_fps': 1}

    def __init__(self, stock_code, start_date, end_date, api_account, api_password,
                 initial_capital=1000000, shares_per_trade=1000, # 訓練時使用固定股數
                 ma_long=50, rsi_period=14, atr_period=14,
                 sl_atr_multiplier=2.0, tp_atr_multiplier=3.0,
                 window_size=60, reward_scaling=1.0, # Reward scaling for training
                 penalty_hold_below_sl = 0.1, render_mode=None):
        super().__init__()
        self.stock_code = stock_code
        self.start_date_str = start_date
        self.end_date_str = end_date
        # Use a separate API instance for each env during training if needed,
        # or simplify if training doesn't hit the real API frequently.
        self.api = Stock_API(api_account, api_password)
        self.initial_capital = initial_capital
        self.shares_per_trade = shares_per_trade # 固定買賣股數 (1張)

        # Technical Indicator Parameters
        self.ma_long_period = ma_long
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.sl_atr_multiplier = sl_atr_multiplier
        self.tp_atr_multiplier = tp_atr_multiplier
        self.window_size = window_size

        self.reward_scaling = reward_scaling
        self.penalty_hold_below_sl = penalty_hold_below_sl
        self.render_mode = render_mode

        # Load and preprocess data
        self.data_df = self._load_and_preprocess_data(start_date, end_date)
        if self.data_df is None or len(self.data_df) < self.window_size:
            raise ValueError(f"股票 {stock_code} 數據載入失敗或數據量不足 (需要 {self.window_size}, 實際 {len(self.data_df) if self.data_df is not None else 0})")
        self.end_step = len(self.data_df) - 1

        # Define action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Define observation space shape (7 features as planned)
        self.features_per_stock = 7
        self.observation_shape = (self.features_per_stock,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.observation_shape, dtype=np.float32)

        # Environment state variables
        self.current_step = 0
        self.cash = 0.0
        self.shares_held = 0 # 單位: 股
        self.entry_price = 0.0
        self.entry_atr = 0.0
        self.portfolio_value = 0.0
        self.done = False
        # Note: Don't call reset in init, Stable Baselines wrappers will call it

    def _load_and_preprocess_data(self, start_date, end_date):
        """載入、清洗單支股票數據並計算指標。"""
        print(f"  StockTradingEnv ({self.stock_code}): 載入數據 {start_date} to {end_date}")
        # Calculate required start date for window_size
        try:
            start_dt_obj = pd.to_datetime(start_date, format='%Y%m%d')
            buffer_days = 30 # Extra buffer
            # 向前推算足夠天數（考慮週末和假期，乘以約1.5倍）
            required_start_dt = start_dt_obj - pd.Timedelta(days=(self.window_size + buffer_days) * 1.5)
            required_start_date_str = required_start_dt.strftime('%Y%m%d')
            # print(f"    為滿足窗口需求 ({self.window_size}天)，請求數據起始日期: {required_start_date_str}")
        except ValueError:
            print(f"    錯誤：起始日期格式無效 {start_date}")
            return None

        raw_data = self.api.Get_Stock_Informations(self.stock_code, required_start_date_str, end_date)
        if not raw_data:
            print(f"    無法從 API 獲取 {self.stock_code} 的數據。")
            return None
        try:
            df = pd.DataFrame(raw_data)
            df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
            df = df.sort_values('date').set_index('date') # 確保按日期排序
            df = df.rename(columns={
                'opening_price': 'open', 'highest_price': 'high', 'lowest_price': 'low',
                'closing_price': 'close', 'transaction_shares': 'volume'})
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=numeric_cols) # 移除無法轉換的行
            if df.empty:
                print(f"    {self.stock_code} 數據清洗後為空。")
                return None

            # 計算技術指標
            df.ta.sma(length=self.ma_long_period, append=True, col_names=(f'SMA_{self.ma_long_period}',))
            df.ta.rsi(length=self.rsi_period, append=True, col_names=(f'RSI_{self.rsi_period}',))
            df.ta.atr(length=self.atr_period, append=True, col_names=(f'ATR_{self.atr_period}',))
            # 檢查 ATR 是否成功計算
            if f'ATR_{self.atr_period}' not in df.columns:
                 print(f"    錯誤：ATR 指標未能成功計算 ({self.stock_code})。")
                 # 如果 ATR 失敗，可能無法繼續
                 return None
            df[f'ATR_norm_{self.atr_period}'] = df[f'ATR_{self.atr_period}'] / df['close']
            df = df.dropna() # 移除因指標計算產生的 NaN (會移除前面 window_size-1 行)

            # 過濾回原始的 start_date 之後的數據用於模擬
            df_filtered = df[df.index >= start_dt_obj]
            # print(f"    > {self.stock_code} 數據處理完成，用於模擬的數據: {len(df_filtered)} 行")
            if len(df_filtered) == 0:
                # print(f"    警告：{self.stock_code} 在指定日期範圍內沒有數據。")
                return None
            return df_filtered
        except Exception as e:
            print(f"    StockTradingEnv ({self.stock_code}): 處理數據時出錯: {e}")
            traceback.print_exc() # 打印詳細錯誤
            return None

    def _get_observation(self, step):
        """根據給定的時間步 (T日收盤後) 計算觀察向量。"""
        # step 是相對於 self.data_df 的索引 (從 0 開始)
        if step < 0 or step >= len(self.data_df):
             # print(f"警告 ({self.stock_code}): _get_observation 收到無效的 step {step}")
             return np.zeros(self.observation_shape, dtype=np.float32)

        obs_data = self.data_df.iloc[step]
        close_price = obs_data['close']
        # Handle potential missing indicator columns if calculation failed earlier
        atr_val = obs_data.get(f'ATR_{self.atr_period}', 0.0)
        atr_norm_val = obs_data.get(f'ATR_norm_{self.atr_period}', 0.0)
        ma_long_val = obs_data.get(f'SMA_{self.ma_long_period}', close_price) # Default to close if MA missing
        rsi_val = obs_data.get(f'RSI_{self.rsi_period}', 50.0) / 100.0 # Default to neutral RSI

        price_ma_ratio = close_price / ma_long_val if ma_long_val != 0 else 1.0

        holding_position = 1.0 if self.shares_held > 0 else 0.0

        # 停損/停利相關特徵
        potential_sl = 0.0
        potential_tp = 0.0
        distance_to_sl_norm = 0.0 # 標準化距離, >0 表示高於 SL
        distance_to_tp_norm = 0.0 # 標準化距離, >0 表示低於 TP
        is_below_potential_sl = 0.0

        # 僅在持有倉位且進場資訊有效時計算
        if self.shares_held > 0 and self.entry_price > 0 and self.entry_atr > 0:
            potential_sl = self.entry_price - self.sl_atr_multiplier * self.entry_atr
            potential_tp = self.entry_price + self.tp_atr_multiplier * self.entry_atr
            if close_price > 0:
                distance_to_sl_norm = (close_price - potential_sl) / close_price
                distance_to_tp_norm = (potential_tp - close_price) / close_price # 價格低於 TP 時為正
            if close_price < potential_sl:
                is_below_potential_sl = 1.0

        features = [
            price_ma_ratio, rsi_val, atr_norm_val, holding_position,
            distance_to_sl_norm, distance_to_tp_norm, is_below_potential_sl
        ]
        observation = np.array(features, dtype=np.float32)
        # 處理可能的 NaN 或 Inf
        observation = np.nan_to_num(observation, nan=0.0, posinf=1e9, neginf=-1e9)
        return observation

    def _calculate_portfolio_value(self, step):
        """計算在 T 日收盤時的投資組合總價值。"""
        if step < 0 or step >= len(self.data_df):
             # print(f"警告 ({self.stock_code}): 計算價值時 step {step} 無效")
             return self.portfolio_value # 返回上一個已知值

        close_price = self.data_df.iloc[step]['close']
        stock_value = self.shares_held * close_price
        return self.cash + stock_value

    def reset(self, seed=None, options=None):
        """重設環境到初始狀態。"""
        super().reset(seed=seed)
        # 確保 data_df 已成功加載
        if self.data_df is None or len(self.data_df) < self.window_size:
             # This should ideally be caught in __init__, but double-check
             raise RuntimeError(f"無法重設環境 ({self.stock_code}): 數據未成功加載或不足。")

        # start_step 應該是第一個可以計算完整指標的點
        # 因為 dropna() 會移除前面 window_size-1 行，所以索引 0 是第一個有效點
        self.current_step = 0 # 從 data_df 的第一個有效索引開始
        self.cash = self.initial_capital
        self.shares_held = 0
        self.entry_price = 0.0
        self.entry_atr = 0.0
        self.portfolio_value = self.initial_capital # 初始價值等於現金
        self.done = False
        # print(f"    StockTradingEnv ({self.stock_code}): Environment reset at step {self.current_step}.")

        observation = self._get_observation(self.current_step)
        info = {"message": f"{self.stock_code} Env reset"}
        return observation, info

    def step(self, action):
        """執行一個時間步（一個交易日循環）。"""
        if self.done:
            # 確保已結束的環境被正確處理
            # print(f"警告 ({self.stock_code}): 在環境結束後調用了 step()")
            obs = self._get_observation(self.current_step if self.current_step < len(self.data_df) else self.current_step - 1)
            return obs, 0.0, True, False, {"message": "回合已結束."}

        previous_portfolio_value = self.portfolio_value # T日收盤價值
        current_data_T = self.data_df.iloc[self.current_step]
        close_price_T = current_data_T['close']
        # Handle cases where indicators might be missing (though dropna should prevent this)
        atr_T = current_data_T.get(f'ATR_{self.atr_period}', 0.0)

        placed_order_type = 'hold'
        # 在訓練中，我們可以簡化API調用和成交確認
        order_api_call_successful = True # 假設API調用總是成功
        trade_executed = False          # 假設默認未成交

        # --- 檢查止損條件 (用於獎勵 shaping) ---
        potential_sl_level = 0.0
        was_below_sl_on_T = False # 標記 T 日收盤是否已低於 SL
        if self.shares_held > 0 and self.entry_price > 0 and self.entry_atr > 0:
             potential_sl_level = self.entry_price - self.sl_atr_multiplier * self.entry_atr
             if close_price_T < potential_sl_level:
                 was_below_sl_on_T = True

        # --- 模擬 T+1 開盤前的決策與下單 ---
        if action == 1: # Buy
            if self.shares_held == 0: # 只能在空倉時買入
                # 使用固定的 shares_per_trade
                estimated_cost = self.shares_per_trade * close_price_T
                if self.cash >= estimated_cost:
                    # Simulate API call (we just assume success here)
                    # success = self.api.Buy_Stock(self.stock_code, self.shares_per_trade, close_price_T)
                    # if success:
                    placed_order_type = 'buy'
                    trade_executed = True # 訓練中假設買單會成交
                    # else: placed_order_type = 'buy_failed' # If simulating API failure
                else: placed_order_type = 'hold_cant_buy_cash'
            else: placed_order_type = 'hold_cant_buy_holding'
        elif action == 2: # Sell
            if self.shares_held > 0:
                 # Simulate API call
                 # success = self.api.Sell_Stock(self.stock_code, self.shares_held, close_price_T)
                 # if success:
                 placed_order_type = 'sell'
                 trade_executed = True # 訓練中假設賣單會成交
                 # else: placed_order_type = 'sell_failed'
            else: placed_order_type = 'hold_cant_sell'
        else: # Hold
             placed_order_type = 'hold'

        # --- 時間推進到 T+1 ---
        self.current_step += 1
        terminated = self.current_step >= self.end_step # 是否到達數據末尾
        truncated = False # 沒有步數限制

        # --- 模擬 T+1 的成交結算與狀態更新 ---
        # 在訓練中，我們不實際調用 Get_User_Stocks，而是基於 trade_executed 標誌更新
        if not terminated:
             current_data_T1 = self.data_df.iloc[self.current_step]
             price_T1_open = current_data_T1['open'] # 用 T+1 開盤價模擬成交價
             atr_T1 = current_data_T1.get(f'ATR_{self.atr_period}', 0.0) # T+1 的 ATR

             if placed_order_type == 'buy' and trade_executed:
                  cost = self.shares_per_trade * price_T1_open
                  # 確保現金足夠（理論上前面檢查過，但做個防護）
                  if self.cash >= cost:
                       self.cash -= cost
                       self.shares_held += self.shares_per_trade
                       self.entry_price = price_T1_open
                       self.entry_atr = atr_T1
                  else: # 如果因價格跳空導致現金不足（訓練中少見）
                       trade_executed = False # 視為未成交
                       print(f"警告 ({self.stock_code}): 模擬買入時現金不足 (T+1 Open 跳空?)")

             elif placed_order_type == 'sell' and trade_executed:
                  proceeds = self.shares_held * price_T1_open
                  self.cash += proceeds
                  self.shares_held = 0
                  self.entry_price = 0.0
                  self.entry_atr = 0.0

             # 更新 T+1 收盤後的投資組合價值
             self.portfolio_value = self._calculate_portfolio_value(self.current_step)

        else: # 回合結束，基於最後一步的收盤價計算最終價值
             # Use the closing price of the last valid step (current_step - 1)
             self.portfolio_value = self._calculate_portfolio_value(self.current_step - 1)

        # --- 計算獎勵 ---
        reward = 0.0
        # 避免在初始步驟（portfolio_value可能為0）或初始資本為0時除以零
        if previous_portfolio_value != 0:
             reward = (self.portfolio_value - previous_portfolio_value) / previous_portfolio_value
        elif self.initial_capital != 0: # Use initial capital as denominator if previous was 0
             reward = (self.portfolio_value - self.initial_capital) / self.initial_capital if self.initial_capital else 0
        # else: reward remains 0

        # --- 獎勵塑形 (Penalty) ---
        # 如果 T 日收盤已低於 SL，但 T+1 的動作是持有或買入（即沒有嘗試賣出）
        if was_below_sl_on_T and action != 2:
             reward -= self.penalty_hold_below_sl
             # print(f"    ({self.stock_code}) Penalty applied: Held below SL")

        # 應用獎勵縮放
        reward *= self.reward_scaling

        # --- 獲取下一個觀察狀態 ---
        # 如果 terminated，獲取的是最後一步的觀察狀態
        observation = self._get_observation(self.current_step if not terminated else self.current_step - 1)

        self.done = terminated # 更新環境結束標誌

        # --- Info 字典 ---
        info = {
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "shares_held": self.shares_held,
            "placed_order_type": placed_order_type, # 意圖的動作
            "trade_executed (simulated)": trade_executed, # 模擬的成交結果
            "reward": reward # 最終獎勵
        }

        # Optional rendering
        # if self.render_mode == 'human': self.render(info)

        return observation, reward, terminated, truncated, info

    def render(self, info=None):
        """可選的渲染方法，用於打印狀態信息。"""
        if self.render_mode == 'human':
             # Handle potential index out of bounds if called after termination
             if self.current_step < len(self.data_df):
                 date_str = self.data_df.index[self.current_step].strftime('%Y-%m-%d')
             else:
                 date_str = self.data_df.index[-1].strftime('%Y-%m-%d') + " (End)"

             print(f"[{self.stock_code} @ {date_str}] Step: {self.current_step}, Portf: {self.portfolio_value:.2f}, Cash: {self.cash:.2f}, Shares: {self.shares_held}, "
                   f"Order: {info.get('placed_order_type', 'N/A')}, Executed: {info.get('trade_executed (simulated)', 'N/A')}, Reward: {info.get('reward', 0):.4f}")

    def close(self):
        """清理環境資源（如果需要）。"""
        # print(f"    StockTradingEnv ({self.stock_code}): Closing environment.")
        pass


# --- Main Execution Block for Training ---
if __name__ == '__main__':

    # --- Configuration ---
    # !!! 重要：請替換為您的 API 憑證 !!!
    API_ACCOUNT = "YOUR_API_ACCOUNT"
    API_PASSWORD = "YOUR_API_PASSWORD"
    STOCK_CODES_LIST = ['2330','2454','2317','2308','2881','2891','2382','2303','2882','2412',
                       '2886','3711','2884','2357','1216','2885','3034','3231','2892','2345']

    # --- Phase Selection (Set RUN_TRAINING=True, RUN_EVALUATION=False for this script) ---
    RUN_TRAINING = True
    RUN_EVALUATION = False # Evaluation code is in a separate script

    # --- Training Parameters (if RUN_TRAINING is True) ---
    START_DATE_TRAIN = '20220101'
    END_DATE_TRAIN = '20221231'
    INITIAL_CAPITAL_PER_MODEL = 1000000.0 # 訓練時每個模型假設的虛擬資金
    SHARES_PER_TRADE_TRAIN = 1000      # 訓練環境使用的固定股數 (1張)
    MA_LONG_TRAIN = 50                 # 長均線週期
    RSI_PERIOD_TRAIN = 14              # RSI 週期
    ATR_PERIOD_TRAIN = 14              # ATR 週期
    SL_ATR_MULT_TRAIN = 2.0            # 停損 ATR 乘數 (用於觀察值)
    TP_ATR_MULT_TRAIN = 3.0            # 停利 ATR 乘數 (用於觀察值)
    WINDOW_SIZE_TRAIN = MA_LONG_TRAIN + 10 # 窗口大小 (確保 >= MA_LONG_TRAIN)
    TOTAL_TIMESTEPS_PER_MODEL = 50000  # 每個模型的訓練步數 (可增加)
    REWARD_SCALING_TRAIN = 100.0       # 訓練時的獎勵縮放 (可調整)
    PENALTY_HOLD_BELOW_SL_TRAIN = 0.1  # 訓練時的持有懲罰 (可調整)
    MODELS_SAVE_DIR = "trained_individual_models" # 模型儲存目錄
    TENSORBOARD_BASE_DIR = "./individual_tensorboard/" # TensorBoard 日誌目錄

    # --- Execute Selected Phase(s) ---

    if RUN_TRAINING:
        print("\n=============== 開始獨立模型訓練階段 ===============")
        # Check if StockTradingEnv is defined and seems valid
        if 'StockTradingEnv' not in globals() or not hasattr(StockTradingEnv, 'step'):
             print("\n錯誤：StockTradingEnv 類別未定義或不完整。請確保已正確粘貼定義。無法進行訓練。\n")
        else:
            os.makedirs(MODELS_SAVE_DIR, exist_ok=True)
            os.makedirs(TENSORBOARD_BASE_DIR, exist_ok=True)

            for stock_code in STOCK_CODES_LIST:
                print(f"\n--- Training: {stock_code} ---")
                try:
                    # 創建單股票訓練環境
                    env = StockTradingEnv(
                        stock_code=stock_code, start_date=START_DATE_TRAIN, end_date=END_DATE_TRAIN,
                        api_account=API_ACCOUNT, api_password=API_PASSWORD,
                        initial_capital=INITIAL_CAPITAL_PER_MODEL,
                        shares_per_trade=SHARES_PER_TRADE_TRAIN,
                        ma_long=MA_LONG_TRAIN, rsi_period=RSI_PERIOD_TRAIN, atr_period=ATR_PERIOD_TRAIN,
                        sl_atr_multiplier=SL_ATR_MULT_TRAIN, tp_atr_multiplier=TP_ATR_MULT_TRAIN,
                        window_size=WINDOW_SIZE_TRAIN,
                        reward_scaling=REWARD_SCALING_TRAIN,
                        penalty_hold_below_sl=PENALTY_HOLD_BELOW_SL_TRAIN,
                        render_mode=None # 訓練時通常不渲染
                    )
                    # 使用 DummyVecEnv 包裝
                    vec_env = DummyVecEnv([lambda: env])

                    # 創建 PPO 模型
                    model = PPO("MlpPolicy", vec_env, verbose=0, # verbose=0 減少訓練過程輸出
                                tensorboard_log=os.path.join(TENSORBOARD_BASE_DIR, stock_code),
                                seed=42, # 設置隨機種子以保證可重複性
                                # 可選：調整 PPO 超參數
                                # learning_rate=0.0003,
                                # n_steps=2048,
                                # batch_size=64,
                                # gamma=0.99,
                                # gae_lambda=0.95,
                                )

                    # 訓練模型
                    print(f"  開始訓練 {TOTAL_TIMESTEPS_PER_MODEL} 步...")
                    model.learn(total_timesteps=TOTAL_TIMESTEPS_PER_MODEL, log_interval=50) # 調整日誌間隔
                    print(f"  訓練完成。")

                    # 儲存模型
                    save_path = os.path.join(MODELS_SAVE_DIR, f"ppo_agent_{stock_code}")
                    model.save(save_path)
                    print(f"  模型已儲存: {save_path}.zip")

                    vec_env.close() # 關閉環境

                except ValueError as e:
                    print(f"股票 {stock_code} 環境初始化或數據錯誤: {e}")
                except Exception as e:
                    print(f"訓練股票 {stock_code} 時發生未預期的錯誤: {e}")
                    traceback.print_exc() # 打印詳細錯誤信息
            print("\n=============== 獨立模型訓練階段完成 ===============")

    if RUN_EVALUATION:
        print("\n=============== 評估階段 (在此腳本中禁用) ===============")
        print("請運行包含 Refactored Classes (DataManager, PortfolioManager, etc.) 的評估腳本。")

    if not RUN_TRAINING and not RUN_EVALUATION:
        print("\n請設置 RUN_TRAINING 為 True 來執行相應階段。")

    print("\n--- 程序執行完畢 ---")