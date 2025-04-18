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
                # Add check for empty data list
                if not data:
                     print(f"API 警告 (Get_Stock_Informations - {stock_code}): 返回成功但數據列表為空。")
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

# --- StockTradingEnv Class (Single-Stock Training Environment - MODIFIED FOR API RESPONSE) ---
class StockTradingEnv(gym.Env):
    """
    用於獨立訓練單支股票模型的 Gymnasium 環境。
    (已修改以處理新的 API 數據格式)
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
        self.api = Stock_API(api_account, api_password)
        self.initial_capital = initial_capital
        self.shares_per_trade = shares_per_trade

        self.ma_long_period = ma_long
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.sl_atr_multiplier = sl_atr_multiplier
        self.tp_atr_multiplier = tp_atr_multiplier
        self.window_size = window_size

        self.reward_scaling = reward_scaling
        self.penalty_hold_below_sl = penalty_hold_below_sl
        self.render_mode = render_mode

        self.data_df = self._load_and_preprocess_data(start_date, end_date)
        if self.data_df is None or len(self.data_df) < self.window_size:
            raise ValueError(f"股票 {stock_code} 數據載入失敗或數據量不足 (需要 {self.window_size}, 實際 {len(self.data_df) if self.data_df is not None else 0})")
        self.end_step = len(self.data_df) - 1

        self.action_space = spaces.Discrete(3) # 0:Hold, 1:Buy, 2:Sell
        self.features_per_stock = 7
        self.observation_shape = (self.features_per_stock,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.observation_shape, dtype=np.float32)

        self.current_step = 0
        self.cash = 0.0
        self.shares_held = 0
        self.entry_price = 0.0
        self.entry_atr = 0.0
        self.portfolio_value = 0.0
        self.done = False

    def _load_and_preprocess_data(self, start_date, end_date):
        """載入、清洗數據並計算指標 (已適應新的 API 格式)。"""
        print(f"  StockTradingEnv ({self.stock_code}): 載入數據 {start_date} to {end_date}")
        try:
            start_dt_obj = pd.to_datetime(start_date, format='%Y%m%d')
            buffer_days = 30
            required_start_dt = start_dt_obj - pd.Timedelta(days=(self.window_size + buffer_days) * 1.5)
            required_start_date_str = required_start_dt.strftime('%Y%m%d')
        except ValueError:
            print(f"    錯誤：起始日期格式無效 {start_date}")
            return None

        raw_data = self.api.Get_Stock_Informations(self.stock_code, required_start_date_str, end_date)
        if not raw_data:
            print(f"    無法從 API 獲取 {self.stock_code} 的數據。")
            return None
        try:
            df = pd.DataFrame(raw_data)
            if df.empty:
                 print(f"    API 返回空數據列表 ({self.stock_code})。")
                 return None

            # --- 修改: 處理 Unix Timestamp 日期 ---
            df['date'] = pd.to_datetime(df['date'], unit='s')
            df = df.sort_values('date').set_index('date')

            # --- 修改: 處理欄位名稱和成交量 ---
            # API 直接提供 open, high, low, close, turnover
            required_cols = ['open', 'high', 'low', 'close', 'turnover']
            if not all(col in df.columns for col in required_cols):
                 print(f"    錯誤：{self.stock_code} 缺少必要的 OHLC 或 Turnover 欄位。")
                 return None

            # 轉換數值類型
            numeric_cols = ['open', 'high', 'low', 'close', 'turnover', 'capacity', 'transaction_volume'] # 包括所有可能的數值列
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # 計算近似成交量 volume = turnover / close
            if 'turnover' in df.columns and 'close' in df.columns:
                 # 確保 close 不是 0 或 NaN
                 df['volume'] = np.where(df['close'].isna() | (df['close'] == 0),
                                         0, # If close is NaN or 0, volume is 0
                                         df['turnover'] / df['close']) # Otherwise calculate
                 df['volume'] = df['volume'].fillna(0).replace([np.inf, -np.inf], 0)
                 df['volume'] = df['volume'].round().astype(np.int64) # 使用 np.int64 避免溢出
            else:
                 df['volume'] = 0 # 如果缺少欄位，設為 0
                 print(f"    警告：無法計算 {self.stock_code} 的近似成交量 (volume)。")

            # --- 後續處理 ---
            # 移除基礎欄位 (OHLC) 包含 NaN 的行
            indicator_base_cols = ['open', 'high', 'low', 'close']
            df = df.dropna(subset=indicator_base_cols)
            if df.empty:
                print(f"    {self.stock_code} 在移除 OHLC NaN 後為空。")
                return None

            # 計算技術指標 (SMA, RSI, ATR 不需要 volume)
            df.ta.sma(length=self.ma_long_period, close='close', append=True, col_names=(f'SMA_{self.ma_long_period}',))
            df.ta.rsi(length=self.rsi_period, close='close', append=True, col_names=(f'RSI_{self.rsi_period}',))
            # ATR 需要 high, low, close
            df.ta.atr(length=self.atr_period, high='high', low='low', close='close', append=True, col_names=(f'ATR_{self.atr_period}',))

            if f'ATR_{self.atr_period}' not in df.columns:
                 print(f"    錯誤：ATR 指標未能成功計算 ({self.stock_code})。")
                 return None
            df[f'ATR_norm_{self.atr_period}'] = df[f'ATR_{self.atr_period}'] / df['close']
            # 再次處理可能因 ATR_norm 計算產生的 inf/-inf
            df[f'ATR_norm_{self.atr_period}'] = df[f'ATR_norm_{self.atr_period}'].replace([np.inf, -np.inf], 0)


            df = df.dropna() # 移除因指標計算產生的 NaN
            if df.empty:
                print(f"    {self.stock_code} 在計算指標後為空。")
                return None

            # 過濾回原始的 start_date 之後的數據用於模擬
            df_filtered = df[df.index >= start_dt_obj]
            print(f"    > {self.stock_code} 數據處理完成，用於模擬的數據: {len(df_filtered)} 行")
            if len(df_filtered) < self.window_size: # 確保過濾後仍有足夠數據
                print(f"    警告：{self.stock_code} 在指定日期範圍內數據不足 ({len(df_filtered)} < {self.window_size})。")
                return None
            return df_filtered
        except Exception as e:
            print(f"    StockTradingEnv ({self.stock_code}): 處理數據時出錯: {e}")
            traceback.print_exc()
            return None

    # _get_observation, _calculate_portfolio_value, reset, step, render, close 方法保持不變
    # (它們使用的欄位名稱在 _load_and_preprocess_data 中已處理好)
    def _get_observation(self, step):
        """根據給定的時間步 (T日收盤後) 計算觀察向量。"""
        if step < 0 or step >= len(self.data_df):
             return np.zeros(self.observation_shape, dtype=np.float32)

        obs_data = self.data_df.iloc[step]
        close_price = obs_data['close']
        atr_val = obs_data.get(f'ATR_{self.atr_period}', 0.0)
        atr_norm_val = obs_data.get(f'ATR_norm_{self.atr_period}', 0.0)
        ma_long_val = obs_data.get(f'SMA_{self.ma_long_period}', close_price)
        rsi_val = obs_data.get(f'RSI_{self.rsi_period}', 50.0) / 100.0
        holding_position = 1.0 if self.shares_held > 0 else 0.0
        potential_sl, potential_tp, distance_to_sl_norm, distance_to_tp_norm, is_below_potential_sl = 0.0, 0.0, 0.0, 0.0, 0.0
        entry_p = self.entry_price
        entry_a = self.entry_atr
        if holding_position > 0 and entry_p > 0 and entry_a > 0:
            potential_sl = entry_p - self.sl_atr_multiplier * entry_a
            potential_tp = entry_p + self.tp_atr_multiplier * entry_a
            if close_price > 0:
                distance_to_sl_norm = (close_price - potential_sl) / close_price
                distance_to_tp_norm = (potential_tp - close_price) / close_price
            if close_price < potential_sl: is_below_potential_sl = 1.0

        features = [
            price_ma_ratio, rsi_val, atr_norm_val, holding_position,
            distance_to_sl_norm, distance_to_tp_norm, is_below_potential_sl
        ]
        observation = np.array(features, dtype=np.float32)
        observation = np.nan_to_num(observation, nan=0.0, posinf=1e9, neginf=-1e9)
        return observation

    def _calculate_portfolio_value(self, step):
        """計算在 T 日收盤時的投資組合總價值。"""
        if step < 0 or step >= len(self.data_df):
             return self.portfolio_value

        close_price = self.data_df.iloc[step]['close']
        stock_value = self.shares_held * close_price
        return self.cash + stock_value

    def reset(self, seed=None, options=None):
        """重設環境到初始狀態。"""
        super().reset(seed=seed)
        if self.data_df is None or len(self.data_df) < self.window_size:
             raise RuntimeError(f"無法重設環境 ({self.stock_code}): 數據未成功加載或不足。")
        self.current_step = 0 # Reset to the first valid index of the filtered dataframe
        self.cash = self.initial_capital
        self.shares_held = 0
        self.entry_price = 0.0
        self.entry_atr = 0.0
        self.portfolio_value = self.initial_capital
        self.done = False
        observation = self._get_observation(self.current_step)
        info = {"message": f"{self.stock_code} Env reset"}
        return observation, info

    def step(self, action):
        """執行一個時間步（一個交易日循環）。"""
        if self.done:
            obs = self._get_observation(self.current_step if self.current_step < len(self.data_df) else self.current_step - 1)
            return obs, 0.0, True, False, {"message": "回合已結束."}

        previous_portfolio_value = self.portfolio_value
        current_data_T = self.data_df.iloc[self.current_step]
        close_price_T = current_data_T['close']
        atr_T = current_data_T.get(f'ATR_{self.atr_period}', 0.0)

        placed_order_type = 'hold'
        order_api_call_successful = True
        trade_executed = False

        potential_sl_level = 0.0
        was_below_sl_on_T = False
        if self.shares_held > 0 and self.entry_price > 0 and self.entry_atr > 0:
             potential_sl_level = self.entry_price - self.sl_atr_multiplier * self.entry_atr
             if close_price_T < potential_sl_level: was_below_sl_on_T = True

        if action == 1: # Buy
            if self.shares_held == 0:
                estimated_cost = self.shares_per_trade * close_price_T
                if self.cash >= estimated_cost:
                    placed_order_type = 'buy'
                    trade_executed = True
                else: placed_order_type = 'hold_cant_buy_cash'
            else: placed_order_type = 'hold_cant_buy_holding'
        elif action == 2: # Sell
            if self.shares_held > 0:
                 placed_order_type = 'sell'
                 trade_executed = True
            else: placed_order_type = 'hold_cant_sell'
        else: # Hold
             placed_order_type = 'hold'

        self.current_step += 1
        terminated = self.current_step >= self.end_step
        truncated = False

        if not terminated:
             current_data_T1 = self.data_df.iloc[self.current_step]
             price_T1_open = current_data_T1['open']
             atr_T1 = current_data_T1.get(f'ATR_{self.atr_period}', 0.0)

             if placed_order_type == 'buy' and trade_executed:
                  cost = self.shares_per_trade * price_T1_open
                  if self.cash >= cost:
                       self.cash -= cost
                       self.shares_held += self.shares_per_trade
                       self.entry_price = price_T1_open
                       self.entry_atr = atr_T1
                  else:
                       trade_executed = False
                       # print(f"警告 ({self.stock_code}): 模擬買入時現金不足")
             elif placed_order_type == 'sell' and trade_executed:
                  proceeds = self.shares_held * price_T1_open
                  self.cash += proceeds
                  self.shares_held = 0
                  self.entry_price = 0.0
                  self.entry_atr = 0.0

             self.portfolio_value = self._calculate_portfolio_value(self.current_step)
        else:
             self.portfolio_value = self._calculate_portfolio_value(self.current_step - 1)

        reward = 0.0
        if previous_portfolio_value != 0:
             reward = (self.portfolio_value - previous_portfolio_value) / previous_portfolio_value
        elif self.initial_capital != 0:
             reward = (self.portfolio_value - self.initial_capital) / self.initial_capital if self.initial_capital else 0

        if was_below_sl_on_T and action != 2:
             reward -= self.penalty_hold_below_sl

        reward *= self.reward_scaling
        observation = self._get_observation(self.current_step if not terminated else self.current_step - 1)
        self.done = terminated
        info = { "step": self.current_step, "portfolio_value": self.portfolio_value, "cash": self.cash,
                 "shares_held": self.shares_held, "placed_order_type": placed_order_type,
                 "trade_executed (simulated)": trade_executed, "reward": reward }
        return observation, reward, terminated, truncated, info

    def render(self, info=None):
        if self.render_mode == 'human':
             if self.current_step < len(self.data_df): date_str = self.data_df.index[self.current_step].strftime('%Y-%m-%d')
             else: date_str = self.data_df.index[-1].strftime('%Y-%m-%d') + " (End)"
             print(f"[{self.stock_code} @ {date_str}] Step: {self.current_step}, Portf: {self.portfolio_value:.2f}, Cash: {self.cash:.2f}, Shares: {self.shares_held}, "
                   f"Order: {info.get('placed_order_type', 'N/A')}, Executed: {info.get('trade_executed (simulated)', 'N/A')}, Reward: {info.get('reward', 0):.4f}")

    def close(self): pass


# --- Main Execution Block for Training ---
if __name__ == '__main__':

    # --- Configuration ---
    # !!! 重要：請替換為您的 API 憑證 !!!
    API_ACCOUNT = "YOUR_API_ACCOUNT"
    API_PASSWORD = "YOUR_API_PASSWORD"
    STOCK_CODES_LIST = ['2330','2454','2317','2308','2881','2891','2382','2303','2882','2412',
                       '2886','3711','2884','2357','1216','2885','3034','3231','2892','2345']

    # --- Phase Selection (Set RUN_TRAINING=True for this script) ---
    RUN_TRAINING = True
    RUN_EVALUATION = False # Evaluation code is in a separate script

    # --- Training Parameters ---
    START_DATE_TRAIN = '20220101'
    END_DATE_TRAIN = '20221231'
    INITIAL_CAPITAL_PER_MODEL = 1000000.0
    SHARES_PER_TRADE_TRAIN = 1000 # 1張
    MA_LONG_TRAIN = 50
    RSI_PERIOD_TRAIN = 14
    ATR_PERIOD_TRAIN = 14
    SL_ATR_MULT_TRAIN = 2.0
    TP_ATR_MULT_TRAIN = 3.0
    WINDOW_SIZE_TRAIN = MA_LONG_TRAIN + 10
    TOTAL_TIMESTEPS_PER_MODEL = 50000 # 減少步數以加速測試，實際需要更多
    REWARD_SCALING_TRAIN = 100.0
    PENALTY_HOLD_BELOW_SL_TRAIN = 0.1
    MODELS_SAVE_DIR = "trained_individual_models"
    TENSORBOARD_BASE_DIR = "./individual_tensorboard/"

    if RUN_TRAINING:
        print("\n=============== 開始獨立模型訓練階段 ===============")
        if 'StockTradingEnv' not in globals() or not hasattr(StockTradingEnv, 'step'):
             print("\n錯誤：StockTradingEnv 類別未定義或不完整。請確保已正確粘貼定義。無法進行訓練。\n")
        else:
            os.makedirs(MODELS_SAVE_DIR, exist_ok=True)
            os.makedirs(TENSORBOARD_BASE_DIR, exist_ok=True)

            successful_trains = 0
            failed_trains = []

            for stock_code in STOCK_CODES_LIST:
                print(f"\n--- Training: {stock_code} ---")
                try:
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
                        render_mode=None
                    )
                    vec_env = DummyVecEnv([lambda: env])
                    model = PPO("MlpPolicy", vec_env, verbose=0,
                                tensorboard_log=os.path.join(TENSORBOARD_BASE_DIR, stock_code),
                                seed=42)
                    print(f"  開始訓練 {TOTAL_TIMESTEPS_PER_MODEL} 步...")
                    model.learn(total_timesteps=TOTAL_TIMESTEPS_PER_MODEL, log_interval=100) # Reduced log interval
                    print(f"  訓練完成。")
                    save_path = os.path.join(MODELS_SAVE_DIR, f"ppo_agent_{stock_code}")
                    model.save(save_path)
                    print(f"  模型已儲存: {save_path}.zip")
                    vec_env.close()
                    successful_trains += 1
                except ValueError as e:
                    print(f"股票 {stock_code} 環境初始化或數據錯誤，跳過訓練: {e}")
                    failed_trains.append(f"{stock_code} (Init/Data Error)")
                except Exception as e:
                    print(f"訓練股票 {stock_code} 時發生未預期的錯誤，跳過訓練: {e}")
                    traceback.print_exc()
                    failed_trains.append(f"{stock_code} (Runtime Error)")

            print("\n=============== 獨立模型訓練階段完成 ===============")
            print(f"成功訓練模型數量: {successful_trains}")
            if failed_trains:
                 print(f"失敗/跳過的股票 ({len(failed_trains)}): {', '.join(failed_trains)}")


    if RUN_EVALUATION:
        print("\n=============== 評估階段 (在此腳本中禁用) ===============")
        print("請運行 evaluate_models.py 進行評估。")

    if not RUN_TRAINING and not RUN_EVALUATION:
        print("\n請設置 RUN_TRAINING 為 True 來執行相應階段。")

    print("\n--- 程序執行完畢 ---")