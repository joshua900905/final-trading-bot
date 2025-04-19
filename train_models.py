# -*- coding: utf-8 -*-
# train_models.py - Script for Training Individual Stock Models
# (MA10, RSI14, ATR14, Enhanced Reward Function)

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
                if not data: pass
                return data if isinstance(data, list) else []
            else: print(f"API 錯誤 (Get_Stock_Informations - {stock_code}, {start_date}-{stop_date}): {result.get('status', '未知狀態')}"); return []
        except requests.exceptions.Timeout: print(f"請求超時 (Get_Stock_Informations - {stock_code})"); return []
        except requests.exceptions.RequestException as e: print(f"網路錯誤 (Get_Stock_Informations - {stock_code}): {e}"); return []
        except Exception as e: print(f"處理股票資訊時出錯 (Get_Stock_Informations - {stock_code}): {e}"); return []

    def Get_User_Stocks(self):
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
                         try: processed_holdings.append({'stock_code': str(stock.get('stock_code')), 'shares': int(stock.get('shares', 0)), 'price': float(stock.get('price', 0.0)), 'amount': float(stock.get('amount', 0.0))})
                         except (ValueError, TypeError) as e: print(f"處理持股數據類型轉換錯誤 for {stock.get('stock_code')}: {e}")
                    return processed_holdings
                else: return []
            else: return []
        except requests.exceptions.Timeout: return []
        except requests.exceptions.RequestException as e: return []
        except Exception as e: return []
        return []

    def Buy_Stock(self, stock_code, stock_shares, stock_price):
        stock_shares = int(stock_shares)
        if stock_shares <= 0 or stock_shares % 1000 != 0: return False
        sheets = stock_shares / 1000
        # print(f"模擬提交買單: {sheets:.0f} 張 ({stock_shares} 股) {stock_code} @ 目標價 {stock_price:.2f}")
        return True

    def Sell_Stock(self, stock_code, stock_shares, stock_price):
        stock_shares = int(stock_shares)
        if stock_shares <= 0 or stock_shares % 1000 != 0: return False
        sheets = stock_shares / 1000
        # print(f"模擬提交賣單: {sheets:.0f} 張 ({stock_shares} 股) {stock_code} @ 目標價 {stock_price:.2f}")
        return True

# --- StockTradingEnv Class (Single-Stock Training Environment - Enhanced Rewards) ---
class StockTradingEnv(gym.Env):
    """
    用於獨立訓練單支股票模型的 Gymnasium 環境。
    (已修改以處理新的 API 數據格式, 使用 MA10, 包含增強的獎勵函數)
    """
    metadata = {'render_modes': ['human', None], 'render_fps': 1}

    def __init__(self, stock_code, start_date, end_date, api_account, api_password,
                 initial_capital=1000000, shares_per_trade=1000,
                 ma_short=10, rsi_period=14, atr_period=14,
                 sl_atr_multiplier=2.0, tp_atr_multiplier=3.0,
                 window_size=60,
                 # --- 新增: 獎勵/懲罰係數 ---
                 reward_scaling=1.0,          # 基礎獎勵縮放
                 sl_penalty_factor=0.5,      # 1. 價格低於止損線的懲罰係數
                 profit_bonus_factor=0.1,    # 2. 實現盈利的獎勵係數
                 loss_penalty_factor=0.2,    # 2. 實現虧損的懲罰係數
                 holding_loss_penalty=0.01,  # 3. 持有未實現虧損的每步懲罰
                 transaction_penalty=0.005,  # 4. 每次成交的固定懲罰 (模擬成本)
                 # ---
                 render_mode=None):
        super().__init__()
        self.stock_code = stock_code
        self.start_date_str = start_date
        self.end_date_str = end_date
        self.api = Stock_API(api_account, api_password)
        self.initial_capital = initial_capital
        self.shares_per_trade = shares_per_trade

        self.ma_short_period = ma_short
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.sl_atr_multiplier = sl_atr_multiplier
        self.tp_atr_multiplier = tp_atr_multiplier
        self.window_size = max(self.ma_short_period, self.rsi_period, self.atr_period) + 10

        # --- 存儲獎勵/懲罰係數 ---
        self.reward_scaling = reward_scaling
        self.sl_penalty_factor = sl_penalty_factor
        self.profit_bonus_factor = profit_bonus_factor
        self.loss_penalty_factor = loss_penalty_factor
        self.holding_loss_penalty = holding_loss_penalty
        self.transaction_penalty = transaction_penalty
        # ---
        self.render_mode = render_mode

        self.data_df = self._load_and_preprocess_data(start_date, end_date)
        if self.data_df is None or len(self.data_df) < self.window_size:
             actual_len = len(self.data_df) if self.data_df is not None else 0
             raise ValueError(f"股票 {stock_code} 預處理後數據不足 (需要 {self.window_size}, 實際 {actual_len})")
        self.end_step = len(self.data_df) - 1

        self.action_space = spaces.Discrete(3)
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
        """載入、清洗數據並計算指標 (已適應新的 API 格式, 使用 MA10)。"""
        print(f"  StockTradingEnv ({self.stock_code}): 載入數據 {start_date} to {end_date}")
        try:
            start_dt_obj = pd.to_datetime(start_date, format='%Y%m%d')
            buffer_days = 30
            required_start_dt = start_dt_obj - pd.Timedelta(days=(self.window_size + buffer_days) * 1.5)
            required_start_date_str = required_start_dt.strftime('%Y%m%d')
        except ValueError: print(f"    錯誤：起始日期格式無效 {start_date}"); return None

        raw_data = self.api.Get_Stock_Informations(self.stock_code, required_start_date_str, end_date)
        if not raw_data: print(f"    無法從 API 獲取 {self.stock_code} 的數據。"); return None
        try:
            df = pd.DataFrame(raw_data); df['date'] = pd.to_datetime(df['date'], unit='s')
            df = df.sort_values('date').set_index('date')
            required_cols = ['open', 'high', 'low', 'close', 'turnover']
            if not all(col in df.columns for col in required_cols): print(f"    錯誤：{self.stock_code} 缺少必要的 OHLC 或 Turnover 欄位。"); return None
            numeric_cols = ['open', 'high', 'low', 'close', 'turnover', 'capacity', 'transaction_volume']
            for col in numeric_cols:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
            if 'turnover' in df.columns and 'close' in df.columns:
                 df['volume'] = np.where(df['close'].isna() | (df['close'] == 0), 0, df['turnover'] / df['close'])
                 df['volume'] = df['volume'].fillna(0).replace([np.inf, -np.inf], 0).round().astype(np.int64)
            else: df['volume'] = 0
            indicator_base_cols = ['open', 'high', 'low', 'close']; df = df.dropna(subset=indicator_base_cols)
            if df.empty: print(f"    {self.stock_code} 在移除 OHLC NaN 後為空。"); return None
            df.ta.sma(length=self.ma_short_period, close='close', append=True, col_names=(f'SMA_{self.ma_short_period}',))
            df.ta.rsi(length=self.rsi_period, close='close', append=True, col_names=(f'RSI_{self.rsi_period}',))
            df.ta.atr(length=self.atr_period, high='high', low='low', close='close', append=True, col_names=(f'ATR_{self.atr_period}',))
            if f'ATR_{self.atr_period}' not in df.columns: print(f"    錯誤：ATR 指標未能成功計算 ({self.stock_code})。"); return None
            df[f'ATR_norm_{self.atr_period}'] = df[f'ATR_{self.atr_period}'] / df['close']; df[f'ATR_norm_{self.atr_period}'] = df[f'ATR_norm_{self.atr_period}'].replace([np.inf, -np.inf], 0)
            df = df.dropna()
            if df.empty: print(f"    {self.stock_code} 在計算指標後為空。"); return None
            df_filtered = df[df.index >= start_dt_obj]
            if len(df_filtered) < self.window_size : print(f"    警告：{self.stock_code} 在指定日期範圍內數據不足 ({len(df_filtered)} < {self.window_size})。"); return None
            print(f"    > {self.stock_code} 數據處理完成，用於模擬的數據: {len(df_filtered)} 行")
            return df_filtered
        except Exception as e: print(f"    StockTradingEnv ({self.stock_code}): 處理數據時出錯: {e}"); traceback.print_exc(); return None

    def _get_observation(self, step):
        """計算觀察向量 (使用 MA10)。"""
        if step < 0 or step >= len(self.data_df): return np.zeros(self.observation_shape, dtype=np.float32)
        try:
            obs_data = self.data_df.iloc[step]; close_price = obs_data['close']
            atr_val = obs_data.get(f'ATR_{self.atr_period}', 0.0); atr_norm_val = obs_data.get(f'ATR_norm_{self.atr_period}', 0.0)
            ma_short_val = obs_data.get(f'SMA_{self.ma_short_period}', close_price)
            rsi_val_raw = obs_data.get(f'RSI_{self.rsi_period}', 50.0)
            price_ma_ratio = close_price / ma_short_val if ma_short_val != 0 else 1.0
            rsi_val = rsi_val_raw / 100.0
            holding_position = 1.0 if self.shares_held > 0 else 0.0
            potential_sl, potential_tp, distance_to_sl_norm, distance_to_tp_norm, is_below_potential_sl = 0.0, 0.0, 0.0, 0.0, 0.0
            entry_p, entry_a = self.entry_price, self.entry_atr
            if holding_position > 0 and entry_p > 0 and entry_a > 0:
                potential_sl = entry_p - self.sl_atr_multiplier * entry_a
                potential_tp = entry_p + self.tp_atr_multiplier * entry_a
                if close_price > 0:
                    distance_to_sl_norm = (close_price - potential_sl) / close_price
                    distance_to_tp_norm = (potential_tp - close_price) / close_price
                if close_price < potential_sl and potential_sl > 0: is_below_potential_sl = 1.0
            features = [ price_ma_ratio, rsi_val, atr_norm_val, holding_position, distance_to_sl_norm, distance_to_tp_norm, is_below_potential_sl ]
            observation = np.array(features, dtype=np.float32)
            observation = np.nan_to_num(observation, nan=0.0, posinf=1e9, neginf=-1e9)
            return observation
        except Exception as e: print(f"錯誤 ({self.stock_code}): Obs 未知錯誤 step {step}: {e}"); traceback.print_exc(); return np.zeros(self.observation_shape, dtype=np.float32)

    def _calculate_portfolio_value(self, step):
        """計算投資組合總價值。"""
        if step < 0 or step >= len(self.data_df): return self.portfolio_value
        close_price = self.data_df.iloc[step]['close']
        stock_value = self.shares_held * close_price
        return self.cash + stock_value

    def reset(self, seed=None, options=None):
        """重設環境。"""
        super().reset(seed=seed)
        if self.data_df is None or len(self.data_df) < self.window_size: raise RuntimeError(f"無法重設環境 ({self.stock_code}): 數據不足。")
        self.current_step = 0
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
        """執行一步，包含增強的獎勵函數。"""
        if self.done:
            obs = self._get_observation(self.current_step if self.current_step < len(self.data_df) else self.current_step - 1)
            return obs, 0.0, True, False, {"message": "回合已結束."}

        previous_portfolio_value = self.portfolio_value
        previous_shares_held = self.shares_held # 記錄 T 日持股
        previous_entry_price = self.entry_price # 記錄 T 日成本（用於計算 PnL）

        current_data_T = self.data_df.iloc[self.current_step]
        close_price_T = current_data_T['close']
        atr_T = current_data_T.get(f'ATR_{self.atr_period}', 0.0)

        placed_order_type, trade_executed = 'hold', False
        order_api_call_successful = True # 訓練中假設 API 調用成功

        # --- SL 檢查 (用於獎勵 shaping) ---
        potential_sl_level = 0.0
        was_below_sl_on_T = False
        if previous_shares_held > 0 and previous_entry_price > 0 and self.entry_atr > 0:
             potential_sl_level = previous_entry_price - self.sl_atr_multiplier * self.entry_atr
             if close_price_T < potential_sl_level and potential_sl_level > 0: was_below_sl_on_T = True

        # --- 模擬 T+1 開盤前的決策與下單 ---
        if action == 1 and previous_shares_held == 0:
            estimated_cost = self.shares_per_trade * close_price_T
            if self.cash >= estimated_cost: placed_order_type, trade_executed = 'buy', True
            else: placed_order_type = 'hold_cant_buy_cash'
        elif action == 2 and previous_shares_held > 0:
            placed_order_type, trade_executed = 'sell', True
        elif action == 0: placed_order_type = 'hold'
        else: placed_order_type = 'hold_invalid_condition'

        # --- 時間推進到 T+1 ---
        self.current_step += 1
        terminated = self.current_step >= self.end_step
        truncated = False

        # --- 模擬 T+1 的成交結算與狀態更新 ---
        realized_pnl = 0 # 初始化已實現盈虧
        cost_basis_before_sell = 0 # 初始化賣出前成本基礎

        if not terminated:
             current_data_T1 = self.data_df.iloc[self.current_step]
             price_T1_open = current_data_T1['open']
             atr_T1 = current_data_T1.get(f'ATR_{self.atr_period}', 0.0)
             close_price_T1 = current_data_T1['close'] # 獲取 T+1 收盤價

             if placed_order_type == 'buy' and trade_executed:
                  cost = self.shares_per_trade * price_T1_open
                  if self.cash >= cost:
                       self.cash -= cost; self.shares_held += self.shares_per_trade
                       self.entry_price, self.entry_atr = price_T1_open, atr_T1
                  else: trade_executed = False
             elif placed_order_type == 'sell' and trade_executed:
                  proceeds = previous_shares_held * price_T1_open # 使用賣出前的股數計算收益
                  cost_basis_before_sell = previous_shares_held * previous_entry_price # 使用賣出前的成本計算基礎
                  realized_pnl = proceeds - cost_basis_before_sell # 計算已實現盈虧
                  self.cash += proceeds; self.shares_held = 0
                  self.entry_price, self.entry_atr = 0.0, 0.0

             self.portfolio_value = self._calculate_portfolio_value(self.current_step) # 計算 T+1 收盤價值
        else:
             self.portfolio_value = self._calculate_portfolio_value(self.current_step - 1) # 使用 T 日收盤價值
             close_price_T1 = self.data_df.iloc[self.current_step - 1]['close'] # 使用 T 日收盤價檢查

        # --- 計算增強的獎勵 ---
        reward = 0.0
        # 基礎獎勵：組合價值變化率
        if previous_portfolio_value != 0: reward = (self.portfolio_value - previous_portfolio_value) / previous_portfolio_value
        elif self.initial_capital != 0: reward = (self.portfolio_value - self.initial_capital) / self.initial_capital if self.initial_capital else 0

        # 懲罰與獎勵項 (需要在 reward_scaling 之前應用，或調整係數大小)
        # 1. 更強的止損懲罰 (檢查 T+1 收盤價)
        if previous_shares_held > 0 and previous_entry_price > 0 and self.entry_atr > 0:
            potential_sl_level_t1_check = previous_entry_price - self.sl_atr_multiplier * self.entry_atr
            if close_price_T1 > 0 and close_price_T1 < potential_sl_level_t1_check:
                reward -= self.sl_penalty_factor # 應用較強止損懲罰
                # print(f"    ({self.stock_code}) Strong SL Penalty: T+1 Close < Entry SL")

        # 2. 已實現盈虧獎勵/懲罰
        if realized_pnl > 0:
            # 按比例增加獎勵，這裡用 realized_pnl / previous_portfolio_value 作為相對大小
            # 注意 previous_portfolio_value 可能為 0
            relative_pnl = realized_pnl / previous_portfolio_value if previous_portfolio_value != 0 else 0
            reward += relative_pnl * self.profit_bonus_factor
            # print(f"    ({self.stock_code}) Profit Bonus applied.")
        elif realized_pnl < 0:
            relative_pnl = realized_pnl / previous_portfolio_value if previous_portfolio_value != 0 else 0
            reward += relative_pnl * self.loss_penalty_factor # PnL 是負數，懲罰係數為正
            # print(f"    ({self.stock_code}) Loss Penalty applied.")

        # 3. 持有未實現虧損懲罰
        if self.shares_held > 0 and self.entry_price > 0:
            current_close_price = close_price_T1 if not terminated else self.data_df.iloc[self.current_step - 1]['close']
            if current_close_price > 0 and current_close_price < self.entry_price:
                reward -= self.holding_loss_penalty
                # print(f"    ({self.stock_code}) Holding Loss Penalty applied.")

        # 4. 交易成本懲罰
        if trade_executed:
            reward -= self.transaction_penalty
            # print(f"    ({self.stock_code}) Transaction Penalty applied.")


        # 應用最終的獎勵縮放
        reward *= self.reward_scaling

        observation = self._get_observation(self.current_step if not terminated else self.current_step - 1)
        self.done = terminated
        info = { "step": self.current_step, "portfolio_value": self.portfolio_value, "cash": self.cash,
                 "shares_held": self.shares_held, "placed_order_type": placed_order_type,
                 "trade_executed (simulated)": trade_executed, "reward": reward, "realized_pnl": realized_pnl}
        return observation, reward, terminated, truncated, info

    def render(self, info=None): pass
    def close(self): pass


# --- Main Execution Block for Training ---
if __name__ == '__main__':

    # --- Configuration ---
    API_ACCOUNT = "N26132089"
    API_PASSWORD = "joshua900905"
    STOCK_CODES_LIST = ['2330','2454','2317','2308','2881','2891','2382','2303','2882','2412',
                       '2886','3711','2884','2357','1216','2885','3034','3231','2892','2345']

    RUN_TRAINING = True
    RUN_EVALUATION = False

    # --- Training Parameters ---
    START_DATE_TRAIN = '20220101'
    END_DATE_TRAIN = '20221231'
    INITIAL_CAPITAL_PER_MODEL = 5000000.0
    SHARES_PER_TRADE_TRAIN = 1000 # 1張
    MA_SHORT_TRAIN = 10
    RSI_PERIOD_TRAIN = 14
    ATR_PERIOD_TRAIN = 14
    SL_ATR_MULT_TRAIN = 2.0
    TP_ATR_MULT_TRAIN = 3.0
    WINDOW_SIZE_TRAIN = max(MA_SHORT_TRAIN, RSI_PERIOD_TRAIN, ATR_PERIOD_TRAIN) + 10
    TOTAL_TIMESTEPS_PER_MODEL = 100000  # <<<--- 建議增加步數
    # --- 獎勵/懲罰係數 (需要調優) ---
    REWARD_SCALING_TRAIN = 1.0          # <<<--- 建議從 1 開始嘗試
    SL_PENALTY_FACTOR_TRAIN = 0.5      # 價格低於 SL 的懲罰
    PROFIT_BONUS_FACTOR_TRAIN = 0.2    # 實現盈利的獎勵
    LOSS_PENALTY_FACTOR_TRAIN = 0.4    # 實現虧損的懲罰 (通常 > 盈利獎勵)
    HOLDING_LOSS_PENALTY_TRAIN = 0.01  # 持有未實現虧損的懲罰
    TRANSACTION_PENALTY_TRAIN = 0.005  # 模擬交易成本的懲罰
    # ---
    MODELS_SAVE_DIR = "trained_individual_models_ma10_enhanced_reward" # 新目錄
    TENSORBOARD_BASE_DIR = "./individual_tensorboard_ma10_enhanced_reward/" # 新目錄

    if RUN_TRAINING:
        print("\n=============== 開始獨立模型訓練階段 (MA10, 增強獎勵) ===============")
        if 'StockTradingEnv' not in globals() or not hasattr(StockTradingEnv, 'step'):
             print("\n錯誤：StockTradingEnv 類別未定義或不完整。\n")
        else:
            os.makedirs(MODELS_SAVE_DIR, exist_ok=True)
            os.makedirs(TENSORBOARD_BASE_DIR, exist_ok=True)
            successful_trains, failed_trains = 0, []

            for stock_code in STOCK_CODES_LIST:
                print(f"\n--- Training: {stock_code} ---")
                try:
                    env = StockTradingEnv(
                        stock_code=stock_code, start_date=START_DATE_TRAIN, end_date=END_DATE_TRAIN,
                        api_account=API_ACCOUNT, api_password=API_PASSWORD,
                        initial_capital=INITIAL_CAPITAL_PER_MODEL,
                        shares_per_trade=SHARES_PER_TRADE_TRAIN,
                        ma_short=MA_SHORT_TRAIN,
                        rsi_period=RSI_PERIOD_TRAIN, atr_period=ATR_PERIOD_TRAIN,
                        sl_atr_multiplier=SL_ATR_MULT_TRAIN, tp_atr_multiplier=TP_ATR_MULT_TRAIN,
                        window_size=WINDOW_SIZE_TRAIN,
                        # --- 傳遞新的獎勵係數 ---
                        reward_scaling=REWARD_SCALING_TRAIN,
                        sl_penalty_factor=SL_PENALTY_FACTOR_TRAIN,
                        profit_bonus_factor=PROFIT_BONUS_FACTOR_TRAIN,
                        loss_penalty_factor=LOSS_PENALTY_FACTOR_TRAIN,
                        holding_loss_penalty=HOLDING_LOSS_PENALTY_TRAIN,
                        transaction_penalty=TRANSACTION_PENALTY_TRAIN,
                        # ---
                        render_mode=None
                    )
                    vec_env = DummyVecEnv([lambda: env])
                    model = PPO("MlpPolicy", vec_env, verbose=0,
                                tensorboard_log=os.path.join(TENSORBOARD_BASE_DIR, stock_code),
                                seed=42,
                                # --- 可選: 調整 PPO 超參數以增加探索 ---
                                ent_coef=0.01, # 稍微增加熵係數
                                # learning_rate=0.0001, # 嘗試更小的學習率
                                # n_steps=4096, # 增加步數
                                # batch_size=128, # 調整批次大小
                                )
                    print(f"  開始訓練 {TOTAL_TIMESTEPS_PER_MODEL} 步...")
                    model.learn(total_timesteps=TOTAL_TIMESTEPS_PER_MODEL, log_interval=100)
                    print(f"  訓練完成。")
                    save_path = os.path.join(MODELS_SAVE_DIR, f"ppo_agent_{stock_code}")
                    model.save(save_path)
                    print(f"  模型已儲存: {save_path}.zip")
                    vec_env.close()
                    successful_trains += 1
                except ValueError as e: print(f"股票 {stock_code} 環境初始化或數據錯誤，跳過訓練: {e}"); failed_trains.append(f"{stock_code} (Init/Data Error)")
                except Exception as e: print(f"訓練股票 {stock_code} 時發生未預期的錯誤，跳過訓練: {e}"); traceback.print_exc(); failed_trains.append(f"{stock_code} (Runtime Error)")

            print("\n=============== 獨立模型訓練階段完成 ===============")
            print(f"成功訓練模型數量: {successful_trains}")
            if failed_trains: print(f"失敗/跳過的股票 ({len(failed_trains)}): {', '.join(failed_trains)}")

    if RUN_EVALUATION: print("\n=============== 評估階段 (在此腳本中禁用) ===============")
    if not RUN_TRAINING and not RUN_EVALUATION: print("\n請設置 RUN_TRAINING 為 True 來執行相應階段。")
    print("\n--- 程序執行完畢 ---")