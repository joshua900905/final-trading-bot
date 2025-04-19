# -*- coding: utf-8 -*-
# tune_single_stock.py - Script for Training a SINGLE Stock Model for Parameter Tuning
# (MA10, MA20, MA50, RSI14, ATR14, Enhanced Reward Function + Holding Penalty, with Monitor)

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
import traceback

# --- Stable Baselines 3 Import ---
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
except ImportError:
    print("錯誤：找不到 stable_baselines3 或其組件。請運行 'pip install stable_baselines3[extra]'")
    exit()

# --- Stock API Class ---
class Stock_API:
    # ... (與 train_models.py 中版本相同) ...
    def __init__(self, account, password):
        self.account = account; self.password = password
        self.base_url = 'http://140.116.86.242:8081/stock/api/v1'
    def Get_Stock_Informations(self, stock_code, start_date, stop_date):
        information_url = (f"{self.base_url}/api_get_stock_info_from_date_json/{stock_code}/{start_date}/{stop_date}")
        try:
            response = requests.get(information_url, timeout=15); response.raise_for_status(); result = response.json()
            if result.get('result') == 'success': data = result.get('data', []); return data if isinstance(data, list) else []
            else: print(f"API 錯誤 (Get_Stock_Informations - {stock_code}, {start_date}-{stop_date}): {result.get('status', '未知狀態')}"); return []
        except Exception as e: print(f"Get_Stock_Informations 出錯 ({stock_code}): {e}"); return []
    def Get_User_Stocks(self):
        data = {'account': self.account, 'password': self.password}; search_url = f'{self.base_url}/get_user_stocks'
        try:
            response = requests.post(search_url, data=data, timeout=15); response.raise_for_status(); result = response.json()
            if result.get('result') == 'success': return result.get('data', []) if isinstance(result.get('data', []), list) else []
            else: return []
        except Exception: return []
    def Buy_Stock(self, stock_code, stock_shares, stock_price): return True # Simulate success
    def Sell_Stock(self, stock_code, stock_shares, stock_price): return True # Simulate success


# --- StockTradingEnv Class (Single-Stock Training Environment - Multi MA Features) ---
class StockTradingEnv(gym.Env):
    """
    用於獨立訓練單支股票模型的 Gymnasium 環境。
    (使用 MA10, MA20, MA50, 增強獎勵 + 不作為懲罰)
    """
    metadata = {'render_modes': ['human', None], 'render_fps': 1}

    # --- 修改: 添加 MA20, MA50 週期 (內部使用固定值), 更新 observation_shape ---
    def __init__(self, stock_code, start_date, end_date, api_account, api_password,
                 initial_capital=1000000, shares_per_trade=1000,
                 ma_short=10, rsi_period=14, atr_period=14, # MA20, MA50 使用固定值
                 sl_atr_multiplier=2.0, tp_atr_multiplier=3.0,
                 # window_size 會在內部基於最長週期 (MA50) 計算
                 reward_scaling=1.0, sl_penalty_factor=0.5, profit_bonus_factor=0.1,
                 loss_penalty_factor=0.2, holding_loss_penalty=0.01, transaction_penalty=0.005,
                 max_holding_penalty = 0.1, holding_penalty_increase_rate = 0.001,
                 render_mode=None):
        super().__init__()
        self.stock_code = stock_code; self.start_date_str = start_date; self.end_date_str = end_date
        self.api = Stock_API(api_account, api_password); self.initial_capital = initial_capital; self.shares_per_trade = shares_per_trade
        self.ma_short_period = ma_short
        self.ma_medium_period = 20 # 固定中期均線
        self.ma_long_period = 50  # 固定長期均線
        self.rsi_period = rsi_period; self.atr_period = atr_period
        self.sl_atr_multiplier = sl_atr_multiplier; self.tp_atr_multiplier = tp_atr_multiplier
        # --- 修改: window_size 基於 MA50 ---
        self.window_size = max(self.ma_short_period, self.ma_medium_period, self.ma_long_period, self.rsi_period, self.atr_period) + 10

        self.reward_scaling = reward_scaling; self.sl_penalty_factor = sl_penalty_factor; self.profit_bonus_factor = profit_bonus_factor
        self.loss_penalty_factor = loss_penalty_factor; self.holding_loss_penalty = holding_loss_penalty; self.transaction_penalty = transaction_penalty
        self.max_holding_penalty = max_holding_penalty; self.holding_penalty_increase_rate = holding_penalty_increase_rate
        self.render_mode = render_mode
        self.data_df = self._load_and_preprocess_data(start_date, end_date)
        if self.data_df is None or len(self.data_df) < self.window_size: actual_len = len(self.data_df) if self.data_df is not None else 0; raise ValueError(f"股票 {stock_code} 數據不足 (需 {self.window_size}, 實 {actual_len})")
        self.end_step = len(self.data_df) - 1; self.action_space = spaces.Discrete(3)

        # --- 修改: 觀察空間維度增加到 11 ---
        # 1. price/MA10, 2. price/MA20, 3. price/MA50
        # 4. MA10/MA20, 5. MA20/MA50
        # 6. RSI, 7. ATR_norm
        # 8. holding, 9. dist_sl, 10. dist_tp, 11. below_sl
        self.features_per_stock = 11
        self.observation_shape = (self.features_per_stock,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.observation_shape, dtype=np.float32)
        # ---

        self.current_step = 0; self.cash = 0.0; self.shares_held = 0; self.entry_price = 0.0; self.entry_atr = 0.0; self.portfolio_value = 0.0; self.done = False
        self.consecutive_hold_days = 0

    def _load_and_preprocess_data(self, start_date, end_date):
        """載入、清洗數據並計算指標 (MA10, MA20, MA50, RSI14, ATR14)。"""
        print(f"  StockTradingEnv ({self.stock_code}): 載入數據 {start_date} to {end_date}")
        try: start_dt_obj = pd.to_datetime(start_date, format='%Y%m%d'); buffer_days = 30; required_start_dt = start_dt_obj - pd.Timedelta(days=(self.window_size + buffer_days) * 1.5); required_start_date_str = required_start_dt.strftime('%Y%m%d')
        except ValueError: print(f"    錯誤：起始日期格式無效 {start_date}"); return None
        raw_data = self.api.Get_Stock_Informations(self.stock_code, required_start_date_str, end_date)
        if not raw_data: print(f"    無法從 API 獲取 {self.stock_code} 的數據。"); return None
        try:
            df = pd.DataFrame(raw_data); df['date'] = pd.to_datetime(df['date'], unit='s'); df = df.sort_values('date').set_index('date')
            required_cols = ['open', 'high', 'low', 'close', 'turnover'];
            if not all(col in df.columns for col in required_cols): print(f"    錯誤：{self.stock_code} 缺少必要欄位。"); return None
            numeric_cols = ['open', 'high', 'low', 'close', 'turnover', 'capacity', 'transaction_volume']
            for col in numeric_cols:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
            if 'turnover' in df.columns and 'close' in df.columns: df['volume'] = np.where(df['close'].isna() | (df['close'] == 0), 0, df['turnover'] / df['close']); df['volume'] = df['volume'].fillna(0).replace([np.inf, -np.inf], 0).round().astype(np.int64)
            else: df['volume'] = 0
            indicator_base_cols = ['open', 'high', 'low', 'close']; df = df.dropna(subset=indicator_base_cols)
            if df.empty: print(f"    {self.stock_code} 在移除 OHLC NaN 後為空。"); return None

            # --- 修改: 計算 MA10, MA20, MA50 ---
            df.ta.sma(length=self.ma_short_period, close='close', append=True, col_names=(f'SMA_{self.ma_short_period}',))
            df.ta.sma(length=self.ma_medium_period, close='close', append=True, col_names=(f'SMA_{self.ma_medium_period}',))
            df.ta.sma(length=self.ma_long_period, close='close', append=True, col_names=(f'SMA_{self.ma_long_period}',))
            # ---
            df.ta.rsi(length=self.rsi_period, close='close', append=True, col_names=(f'RSI_{self.rsi_period}',))
            df.ta.atr(length=self.atr_period, high='high', low='low', close='close', append=True, col_names=(f'ATR_{self.atr_period}',))
            if f'ATR_{self.atr_period}' not in df.columns: print(f"    錯誤：ATR 指標未能成功計算 ({self.stock_code})。"); return None
            df[f'ATR_norm_{self.atr_period}'] = df[f'ATR_{self.atr_period}'] / df['close']; df[f'ATR_norm_{self.atr_period}'] = df[f'ATR_norm_{self.atr_period}'].replace([np.inf, -np.inf], 0)

            df = df.dropna() # 移除所有指標計算產生的 NaN
            if df.empty: print(f"    {self.stock_code} 在計算指標後為空。"); return None
            df_filtered = df[df.index >= start_dt_obj]
            if len(df_filtered) < self.window_size : print(f"    警告：{self.stock_code} 數據不足 ({len(df_filtered)} < {self.window_size})。"); return None
            print(f"    > {self.stock_code} 數據處理完成，用於模擬的數據: {len(df_filtered)} 行")
            return df_filtered
        except Exception as e: print(f"    StockTradingEnv ({self.stock_code}): 處理數據時出錯: {e}"); traceback.print_exc(); return None

    def _get_observation(self, step):
        """計算觀察向量 (包含 MA10, MA20, MA50 相關特徵)。"""
        if step < 0 or step >= len(self.data_df): return np.zeros(self.observation_shape, dtype=np.float32)
        try:
            obs_data = self.data_df.iloc[step]; close_price = obs_data['close']
            # --- 獲取所有需要的指標值 ---
            atr_val = obs_data.get(f'ATR_{self.atr_period}', 0.0); atr_norm_val = obs_data.get(f'ATR_norm_{self.atr_period}', 0.0)
            ma10_val = obs_data.get(f'SMA_{self.ma_short_period}', close_price)
            ma20_val = obs_data.get(f'SMA_{self.ma_medium_period}', close_price)
            ma50_val = obs_data.get(f'SMA_{self.ma_long_period}', close_price)
            rsi_val_raw = obs_data.get(f'RSI_{self.rsi_period}', 50.0)

            # --- 計算特徵 ---
            price_ma10_ratio = close_price / ma10_val if ma10_val != 0 else 1.0
            price_ma20_ratio = close_price / ma20_val if ma20_val != 0 else 1.0
            price_ma50_ratio = close_price / ma50_val if ma50_val != 0 else 1.0
            ma10_ma20_ratio = ma10_val / ma20_val if ma20_val != 0 else 1.0
            ma20_ma50_ratio = ma20_val / ma50_val if ma50_val != 0 else 1.0
            rsi_val = rsi_val_raw / 100.0
            holding_position = 1.0 if self.shares_held > 0 else 0.0
            potential_sl, potential_tp, distance_to_sl_norm, distance_to_tp_norm, is_below_potential_sl = 0.0, 0.0, 0.0, 0.0, 0.0
            entry_p, entry_a = self.entry_price, self.entry_atr
            if holding_position > 0 and entry_p > 0 and entry_a > 0:
                potential_sl = entry_p - self.sl_atr_multiplier * entry_a; potential_tp = entry_p + self.tp_atr_multiplier * entry_a
                if close_price > 0: distance_to_sl_norm = (close_price - potential_sl) / close_price; distance_to_tp_norm = (potential_tp - close_price) / close_price
                if close_price < potential_sl and potential_sl > 0: is_below_potential_sl = 1.0

            # --- 修改: 組裝 11 個特徵 ---
            features = [
                price_ma10_ratio, price_ma20_ratio, price_ma50_ratio, # Price vs MAs
                ma10_ma20_ratio, ma20_ma50_ratio,                     # MA vs MA
                rsi_val, atr_norm_val,                                # Oscillators/Vol
                holding_position,                                     # Position Info
                distance_to_sl_norm, distance_to_tp_norm, is_below_potential_sl # SL/TP Info
            ]
            observation = np.array(features, dtype=np.float32)
            observation = np.nan_to_num(observation, nan=0.0, posinf=1e9, neginf=-1e9); return observation
        except Exception as e: print(f"錯誤 ({self.stock_code}): Obs 未知錯誤 step {step}: {e}"); traceback.print_exc(); return np.zeros(self.observation_shape, dtype=np.float32)

    # _calculate_portfolio_value, reset, step, render, close 方法保持不變
    # (因為它們不直接依賴於觀察空間的具體內容，step 中的獎勵計算邏輯也適用)
    def _calculate_portfolio_value(self, step):
        if step < 0 or step >= len(self.data_df): return self.portfolio_value
        close_price = self.data_df.iloc[step]['close']; stock_value = self.shares_held * close_price; return self.cash + stock_value
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.data_df is None or len(self.data_df) < self.window_size: raise RuntimeError(f"無法重設環境 ({self.stock_code}): 數據不足。")
        self.current_step = 0; self.cash = self.initial_capital; self.shares_held = 0; self.entry_price = 0.0; self.entry_atr = 0.0
        self.portfolio_value = self.initial_capital; self.done = False; self.consecutive_hold_days = 0
        observation = self._get_observation(self.current_step); info = {"message": f"{self.stock_code} Env reset"}; return observation, info
    def step(self, action):
        if self.done: obs = self._get_observation(self.current_step if self.current_step < len(self.data_df) else self.current_step - 1); return obs, 0.0, True, False, {"message": "回合已結束."}
        previous_portfolio_value = self.portfolio_value; previous_shares_held = self.shares_held; previous_entry_price = self.entry_price
        current_data_T = self.data_df.iloc[self.current_step]; close_price_T = current_data_T['close']; atr_T = current_data_T.get(f'ATR_{self.atr_period}', 0.0)
        placed_order_type, trade_executed = 'hold', False; order_api_call_successful = True
        potential_sl_level, was_below_sl_on_T = 0.0, False
        if previous_shares_held > 0 and previous_entry_price > 0 and self.entry_atr > 0: potential_sl_level = previous_entry_price - self.sl_atr_multiplier * self.entry_atr;
        if close_price_T < potential_sl_level and potential_sl_level > 0: was_below_sl_on_T = True
        if action == 1 and previous_shares_held == 0:
            estimated_cost = self.shares_per_trade * close_price_T
            if self.cash >= estimated_cost: placed_order_type, trade_executed, self.consecutive_hold_days = 'buy', True, 0
            else: placed_order_type = 'hold_cant_buy_cash'; self.consecutive_hold_days += 1
        elif action == 2 and previous_shares_held > 0: placed_order_type, trade_executed, self.consecutive_hold_days = 'sell', True, 0
        elif action == 0: placed_order_type = 'hold'; self.consecutive_hold_days += 1
        else: placed_order_type = 'hold_invalid_condition'; self.consecutive_hold_days += 1
        self.current_step += 1; terminated = self.current_step >= self.end_step; truncated = False
        realized_pnl, cost_basis_before_sell = 0, 0; close_price_T1 = 0.0
        if not terminated:
             current_data_T1 = self.data_df.iloc[self.current_step]; price_T1_open = current_data_T1['open']; atr_T1 = current_data_T1.get(f'ATR_{self.atr_period}', 0.0); close_price_T1 = current_data_T1['close']
             if placed_order_type == 'buy' and trade_executed:
                  cost = self.shares_per_trade * price_T1_open
                  if self.cash >= cost: self.cash -= cost; self.shares_held += self.shares_per_trade; self.entry_price, self.entry_atr = price_T1_open, atr_T1
                  else: trade_executed = False
             elif placed_order_type == 'sell' and trade_executed:
                  cost_basis_before_sell = previous_shares_held * previous_entry_price; proceeds = previous_shares_held * price_T1_open
                  realized_pnl = proceeds - cost_basis_before_sell; self.cash += proceeds; self.shares_held = 0; self.entry_price, self.entry_atr = 0.0, 0.0
             self.portfolio_value = self._calculate_portfolio_value(self.current_step)
        else: self.portfolio_value = self._calculate_portfolio_value(self.current_step - 1);
        if self.current_step > 0: close_price_T1 = self.data_df.iloc[self.current_step - 1]['close']
        reward = 0.0
        if previous_portfolio_value != 0: reward = (self.portfolio_value - previous_portfolio_value) / previous_portfolio_value
        elif self.initial_capital != 0: reward = (self.portfolio_value - self.initial_capital) / self.initial_capital if self.initial_capital else 0
        if previous_shares_held > 0 and previous_entry_price > 0 and self.entry_atr > 0:
            potential_sl_level_t1_check = previous_entry_price - self.sl_atr_multiplier * self.entry_atr
            if close_price_T1 > 0 and close_price_T1 < potential_sl_level_t1_check: reward -= self.sl_penalty_factor
        if realized_pnl != 0:
            relative_pnl = realized_pnl / previous_portfolio_value if previous_portfolio_value != 0 else 0
            if realized_pnl > 0: reward += relative_pnl * self.profit_bonus_factor
            else: reward += relative_pnl * self.loss_penalty_factor
        if self.shares_held > 0 and self.entry_price > 0:
            current_close_price = close_price_T1 if not terminated else self.data_df.iloc[self.current_step - 1]['close']
            if current_close_price > 0 and current_close_price < self.entry_price: reward -= self.holding_loss_penalty
        if trade_executed: reward -= self.transaction_penalty
        if action == 0 or placed_order_type == 'hold_invalid_condition':
            current_holding_penalty = min(self.consecutive_hold_days * self.holding_penalty_increase_rate, self.max_holding_penalty)
            reward -= current_holding_penalty
        reward *= self.reward_scaling
        observation = self._get_observation(self.current_step if not terminated else self.current_step - 1); self.done = terminated
        info = { "step": self.current_step, "portfolio_value": self.portfolio_value, "cash": self.cash, "shares_held": self.shares_held, "placed_order_type": placed_order_type, "trade_executed (simulated)": trade_executed, "reward": reward, "realized_pnl": realized_pnl, "hold_days": self.consecutive_hold_days}
        return observation, reward, terminated, truncated, info
    def render(self, info=None): pass
    def close(self): pass


# --- Main Execution Block for Single Stock Tuning ---
if __name__ == '__main__':

    # --- Configuration ---
    API_ACCOUNT = "N26132089"
    API_PASSWORD = "joshua900905"
    TARGET_STOCK_CODE = '2330'

    RUN_TRAINING = True
    RUN_EVALUATION = False

    # --- Training Parameters (Using Multi MA) ---
    START_DATE_TRAIN = '20180101'
    END_DATE_TRAIN = '20231231'
    INITIAL_CAPITAL_PER_MODEL = 500000000.0
    SHARES_PER_TRADE_TRAIN = 1000
    MA_SHORT_TRAIN = 10
    RSI_PERIOD_TRAIN = 14
    ATR_PERIOD_TRAIN = 14
    SL_ATR_MULT_TRAIN = 2.0
    TP_ATR_MULT_TRAIN = 3.0
    # --- WINDOW_SIZE 會在 Env 內部根據 MA50 自動計算 ---
    # --- 增加訓練步數以適應更複雜的狀態 ---
    TOTAL_TIMESTEPS_PER_MODEL = 300000  # <<<--- 強烈建議增加步數
    # --- 獎勵/懲罰係數 (可能需要重新調優) ---
    REWARD_SCALING_TRAIN = 1.0
    SL_PENALTY_FACTOR_TRAIN = 0.2
    PROFIT_BONUS_FACTOR_TRAIN = 0.3
    LOSS_PENALTY_FACTOR_TRAIN = 0.2 # 可以嘗試對稱或略高於盈利
    HOLDING_LOSS_PENALTY_TRAIN = 0.005
    TRANSACTION_PENALTY_TRAIN = 0.001
    MAX_HOLDING_PENALTY_TRAIN = 0.05
    HOLDING_PENALTY_INCREASE_RATE_TRAIN = 0.001
    # --- PPO 超參數 ---
    PPO_ENT_COEF = 0.01 # 可以保持或略微調整
    PPO_LEARNING_RATE = 0.0003
    PPO_N_STEPS = 2048
    PPO_BATCH_SIZE = 64
    # --- 輸出目錄 ---
    experiment_name = "multi_ma_enhanced_reward_holding_penalty_v1" # 新實驗名稱
    MODELS_SAVE_DIR = f"tuned_models/{TARGET_STOCK_CODE}/{experiment_name}"
    TENSORBOARD_LOG_DIR = f"./tuning_tensorboard/{TARGET_STOCK_CODE}/{experiment_name}/"
    MONITOR_LOG_DIR = os.path.join(TENSORBOARD_LOG_DIR, "monitor_logs")

    if RUN_TRAINING:
        print(f"\n=============== 開始單股票調試訓練 ({TARGET_STOCK_CODE} - {experiment_name}) ===============")
        if 'StockTradingEnv' not in globals() or not hasattr(StockTradingEnv, 'step'):
             print("\n錯誤：StockTradingEnv 類別未定義或不完整。\n"); exit()
        else:
            os.makedirs(MODELS_SAVE_DIR, exist_ok=True)
            os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
            os.makedirs(MONITOR_LOG_DIR, exist_ok=True)
            try:
                env = StockTradingEnv(
                    stock_code=TARGET_STOCK_CODE, start_date=START_DATE_TRAIN, end_date=END_DATE_TRAIN,
                    api_account=API_ACCOUNT, api_password=API_PASSWORD,
                    initial_capital=INITIAL_CAPITAL_PER_MODEL, shares_per_trade=SHARES_PER_TRADE_TRAIN,
                    ma_short=MA_SHORT_TRAIN, rsi_period=RSI_PERIOD_TRAIN, atr_period=ATR_PERIOD_TRAIN,
                    sl_atr_multiplier=SL_ATR_MULT_TRAIN, tp_atr_multiplier=TP_ATR_MULT_TRAIN,
                    # window_size 在 Env 內部計算
                    reward_scaling=REWARD_SCALING_TRAIN, sl_penalty_factor=SL_PENALTY_FACTOR_TRAIN,
                    profit_bonus_factor=PROFIT_BONUS_FACTOR_TRAIN, loss_penalty_factor=LOSS_PENALTY_FACTOR_TRAIN,
                    holding_loss_penalty=HOLDING_LOSS_PENALTY_TRAIN, transaction_penalty=TRANSACTION_PENALTY_TRAIN,
                    max_holding_penalty = MAX_HOLDING_PENALTY_TRAIN, holding_penalty_increase_rate = HOLDING_PENALTY_INCREASE_RATE_TRAIN,
                    render_mode=None
                )
                monitor_path = os.path.join(MONITOR_LOG_DIR, f"{TARGET_STOCK_CODE}")
                env = Monitor(env, monitor_path, allow_early_resets=True)
                vec_env = DummyVecEnv([lambda: env])
                model = PPO("MlpPolicy", vec_env, verbose=1,
                            tensorboard_log=TENSORBOARD_LOG_DIR,
                            seed=42, ent_coef=PPO_ENT_COEF, learning_rate=PPO_LEARNING_RATE,
                            n_steps=PPO_N_STEPS, batch_size=PPO_BATCH_SIZE)

                print(f"  開始訓練 {TOTAL_TIMESTEPS_PER_MODEL} 步...")
                model.learn(total_timesteps=TOTAL_TIMESTEPS_PER_MODEL, log_interval=1)
                print(f"  訓練完成。")
                save_path = os.path.join(MODELS_SAVE_DIR, f"ppo_agent_{TARGET_STOCK_CODE}_final")
                model.save(save_path); print(f"  最終模型已儲存: {save_path}.zip")
                vec_env.close()

            except ValueError as e: print(f"股票 {TARGET_STOCK_CODE} 環境初始化或數據錯誤: {e}")
            except Exception as e: print(f"訓練股票 {TARGET_STOCK_CODE} 時發生未預期的錯誤: {e}"); traceback.print_exc();

        print("\n=============== 單股票調試訓練階段完成 ===============")

    if RUN_EVALUATION: print("\n錯誤：此腳本僅用於訓練調試，不執行評估階段。")
    if not RUN_TRAINING and not RUN_EVALUATION: print("\n請設置 RUN_TRAINING = True 來執行。")
    print("\n--- 程序執行完畢 ---")