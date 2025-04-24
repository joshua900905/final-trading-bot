# -*- coding: utf-8 -*-
# train_ema5_touch_atr_rr_v2_2units.py - Train EMA5 Touch Strategy (AI Units 1/2, ATR TP/SL)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import pandas_ta as ta # Needed for EMA and ATR
import requests
import time
import os
import math
# import random # Not needed
from collections import defaultdict
import traceback

# --- Stable Baselines 3 Import ---
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor
except ImportError:
    print("錯誤：找不到 stable_baselines3 或其組件。請運行 'pip install stable_baselines3[extra]'")
    exit()


# --- Stock API Class (保持不變) ---
class Stock_API:
    # ... (程式碼不變) ...
    """Stock API Class"""
    def __init__(self, account, password): self.account = account; self.password = password; self.base_url = 'http://140.116.86.242:8081/stock/api/v1'
    def Get_Stock_Informations(self, stock_code, start_date, stop_date):
        information_url = (f"{self.base_url}/api_get_stock_info_from_date_json/{stock_code}/{start_date}/{stop_date}");
        try:
            response = requests.get(information_url, timeout=15); response.raise_for_status(); result = response.json();
            if result.get('result') == 'success': data = result.get('data', []); return data if isinstance(data, list) else []
            else: print(f"API 錯誤 (Info - {stock_code}): {result.get('status', '未知')}"); return []
        except Exception as e: print(f"Get_Stock_Informations 出錯 ({stock_code}): {e}"); return []


# --- StockTradingEnv Class (EMA5 Touch + AI Units 1/2 + ATR TP/SL) ---
class StockTradingEnvEMA5RR(gym.Env):
    """
    環境：EMA5觸及進場，固定 ATR 止盈(2x)/止損(1x)出場。
    AI 學習：決定買入張數 (1 或 2 張)。
    執行：T 日決策，T+1 模擬買入成交，T+1 檢查止盈/止損並立即模擬成交。
    """
    metadata = {'render_modes': ['human', None], 'render_fps': 1}

    def __init__(self, stock_code, start_date, end_date, api_account, api_password,
                 initial_capital=1000000,
                 # Strategy Rules
                 ema_period=5,
                 atr_period=14,
                 stop_loss_atr_multiplier=1.0,
                 take_profit_atr_multiplier=2.0,
                 # AI Action Space
                 buy_units=[1, 2], # <<< 修改為 1 或 2 張
                 # Reward Shaping Params
                 reward_scaling=1.0,
                 transaction_penalty_per_unit=50,
                 profit_reward_factor=1.1,
                 stop_loss_penalty_factor=1.2,
                 # Env Params
                 window_size=20,
                 shares_per_unit=1000,
                 render_mode=None):
        super().__init__()
        # ... (Core & API Params) ...
        self.stock_code = stock_code; self.start_date_str = start_date; self.end_date_str = end_date
        self.api = Stock_API(api_account, api_password); self.initial_capital = initial_capital;
        # Strategy Params
        self.ema_period = ema_period; self.ema_col_name = f'EMA_{self.ema_period}'
        self.atr_period = atr_period; self.atr_col_name = f'ATR_{self.atr_period}'
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.take_profit_atr_multiplier = take_profit_atr_multiplier
        self.buy_units = buy_units; self.num_buy_choices = len(buy_units) # <<< num_buy_choices 會自動變為 2
        # ... (Reward & Env Params) ...
        self.reward_scaling = reward_scaling; self.transaction_penalty_per_unit = transaction_penalty_per_unit
        self.profit_reward_factor = profit_reward_factor
        self.stop_loss_penalty_factor = stop_loss_penalty_factor
        self.window_size = max(window_size, ema_period + 1, atr_period + 1)
        self.shares_per_unit = shares_per_unit; self.render_mode = render_mode


        # Load Data
        self.data_df = self._load_and_preprocess_data(start_date, end_date)
        if self.data_df is None or len(self.data_df) < self.window_size: raise ValueError("數據不足")
        self.end_step = len(self.data_df) - 1

        # Action Space (自動變為 Discrete(2))
        self.action_space = spaces.Discrete(self.num_buy_choices)

        # Observation Space (保持不變)
        self.features_per_stock = 2 # [NormATR, NormDistToEMA5]
        self.observation_shape = (self.features_per_stock,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.observation_shape, dtype=np.float32)

        # Internal State Variables (保持不變)
        self.current_step = 0; self.cash = 0.0; self.shares_held = 0
        self.entry_price = 0.0; self.stop_loss_price = 0.0; self.take_profit_price = 0.0
        self.entry_units = 0
        self.portfolio_value = 0.0; self.done = False

    def _load_and_preprocess_data(self, start_date, end_date):
        # (與上版本一致)
        # ...
        print(f"  StockTradingEnv ({self.stock_code}): 載入數據 {start_date} to {end_date}")
        try:
            start_dt_obj = pd.to_datetime(start_date, format='%Y%m%d'); buffer_days = max(self.ema_period, self.atr_period) + 5
            required_start_dt = start_dt_obj - pd.Timedelta(days=buffer_days * 1.5); required_start_date_str = required_start_dt.strftime('%Y%m%d')
        except ValueError: print(f"    錯誤：起始日期格式無效 {start_date}"); return None
        raw_data = self.api.Get_Stock_Informations(self.stock_code, required_start_date_str, end_date)
        if not raw_data: print(f"    警告：API 未返回數據"); return None
        try:
            df = pd.DataFrame(raw_data); df['date'] = pd.to_datetime(df['date'], unit='s'); df = df.sort_values('date'); df = df[~df['date'].duplicated(keep='first')]; df = df.set_index('date')
            required_cols = ['open', 'high', 'low', 'close'];
            if not all(col in df.columns for col in required_cols): print(f"    錯誤: 缺少必要欄位"); return None
            numeric_cols = ['open', 'high', 'low', 'close', 'turnover', 'capacity', 'transaction_volume']
            for col in numeric_cols:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
            indicator_base_cols = ['open', 'high', 'low', 'close']; df = df.dropna(subset=indicator_base_cols)
            if df.empty: print(f"    錯誤: 清理後數據為空 (1)"); return None
            df.ta.ema(length=self.ema_period, close='close', append=True, col_names=(self.ema_col_name,))
            df[self.ema_col_name] = df[self.ema_col_name].shift(1)
            df.ta.atr(length=self.atr_period, append=True, col_names=(self.atr_col_name,))
            df[self.atr_col_name] = df[self.atr_col_name].shift(1)
            df = df.dropna();
            if df.empty: print(f"    錯誤: 清理後數據為空 (2)"); return None
            df_filtered = df[df.index >= start_dt_obj]
            if len(df_filtered) < self.window_size : print(f"    錯誤: 過濾後數據不足"); return None
            print(f"    > 數據處理完成: {len(df_filtered)} 行")
            return df_filtered
        except Exception as e: print(f"    處理數據時出錯: {e}"); traceback.print_exc(); return None


    def _get_observation(self, step):
        # (與上版本一致)
        # ...
        if step < 0 or step >= len(self.data_df): return np.zeros(self.observation_shape, dtype=np.float32)
        try:
            obs_data = self.data_df.iloc[step]; close_price = obs_data['close']; atr_val = obs_data[self.atr_col_name]; ema_val = obs_data[self.ema_col_name]
            if pd.isna(close_price) or pd.isna(atr_val) or pd.isna(ema_val) or close_price <= 0 or atr_val <= 0 or ema_val <= 0: return np.zeros(self.observation_shape, dtype=np.float32)
            norm_atr = atr_val / close_price
            norm_dist_to_ema5 = (close_price - ema_val) / atr_val if atr_val > 0 else 0.0
            features = [norm_atr, norm_dist_to_ema5]
            observation = np.array(features, dtype=np.float32)
            observation = np.nan_to_num(observation, nan=0.0, posinf=10.0, neginf=-10.0); observation = np.clip(observation, -10.0, 10.0)
            if observation.shape[0] != self.features_per_stock: print(f"錯誤: Obs 維度錯誤"); return np.zeros(self.observation_shape, dtype=np.float32)
            return observation
        except KeyError as e: print(f"錯誤: Obs 計算缺少欄位 {e}"); return np.zeros(self.observation_shape, dtype=np.float32)
        except Exception as e: print(f"錯誤: Obs 未知錯誤: {e}"); traceback.print_exc(); return np.zeros(self.observation_shape, dtype=np.float32)


    def _calculate_portfolio_value(self, step):
        # (與上版本一致)
        # ...
        if step < 0 or step >= len(self.data_df): return self.portfolio_value
        try:
            current_data = self.data_df.iloc[step]; close_price = current_data['close']
            if pd.isna(close_price) or close_price <= 0:
                 if step > 0:
                      prev_data = self.data_df.iloc[step - 1]; close_price = prev_data['close']
                      if pd.isna(close_price) or close_price <= 0: stock_value = 0.0
                      else: stock_value = self.shares_held * close_price
                 else: stock_value = 0.0
            else: stock_value = self.shares_held * close_price
            return self.cash + stock_value
        except IndexError: return self.portfolio_value
        except Exception as e: print(f"錯誤: _calculate_portfolio_value: {e}"); return self.portfolio_value


    def reset(self, seed=None, options=None):
        # (與上版本一致)
        # ...
        super().reset(seed=seed)
        if self.data_df is None or len(self.data_df) < self.window_size: raise RuntimeError("數據不足")
        self.current_step = 0; self.cash = self.initial_capital
        self.shares_held = 0; self.entry_price = 0.0; self.stop_loss_price = 0.0; self.take_profit_price = 0.0
        self.entry_units = 0;
        self.portfolio_value = self.initial_capital; self.done = False
        observation = self._get_observation(self.current_step)
        return observation, {"message": f"{self.stock_code} Env reset"}


    def step(self, action):
        # (step 邏輯與上版本一致，只是 planned_buy_units 會從 buy_units[0] 或 buy_units[1] 取值)
        if self.done:
            obs = self._get_observation(self.current_step if self.current_step < len(self.data_df) else self.current_step - 1)
            return obs, 0.0, True, False, {"message": "回合已結束."}

        previous_portfolio_value = self.portfolio_value
        # --- T 日收盤 ---
        current_data_T = self.data_df.iloc[self.current_step]
        close_price_T = current_data_T['close']; low_T = current_data_T['low']
        ema5_T = current_data_T[self.ema_col_name]; atr_T = current_data_T[self.atr_col_name]

        entry_signal = False; planned_buy_units = 0
        next_stop_loss = 0.0; next_take_profit = 0.0

        # 檢查數據有效性
        if pd.isna(close_price_T) or pd.isna(low_T) or pd.isna(ema5_T) or pd.isna(atr_T) or \
           close_price_T <= 0 or low_T <= 0 or ema5_T <= 0 or atr_T <= 0:
             print(f"警告: Step {self.current_step} T日數據無效，跳過。")
             self.current_step += 1; terminated = self.current_step >= self.end_step; self.done = terminated
             obs = self._get_observation(self.current_step if not terminated else self.current_step - 1)
             self.portfolio_value = self._calculate_portfolio_value(self.current_step -1)
             return obs, 0.0, terminated, False, {"message": "T日數據無效"}

        # --- T 日決策 ---
        if self.shares_held == 0:
            if low_T <= ema5_T and close_price_T >= ema5_T:
                entry_signal = True
                action_idx = action # AI 決定買 1 張 (idx 0) 或 2 張 (idx 1)
                planned_buy_units = self.buy_units[action_idx]
                entry_price_est = close_price_T
                next_stop_loss = entry_price_est - self.stop_loss_atr_multiplier * atr_T
                next_take_profit = entry_price_est + self.take_profit_atr_multiplier * atr_T
                if next_stop_loss >= entry_price_est:
                     print(f"警告: Step {self.current_step} 止損無效，取消買入。")
                     entry_signal = False; planned_buy_units = 0; next_stop_loss = 0.0; next_take_profit = 0.0
                else:
                     print(f"      >> T日EMA觸及，AI選 {planned_buy_units} 張，計劃 SL={next_stop_loss:.2f}, TP={next_take_profit:.2f}")

        # --- 模擬 T+1 ---
        next_step_idx = self.current_step + 1
        terminated = next_step_idx >= self.end_step
        trade_executed_buy = False; trade_executed_sell = False
        realized_pnl = 0.0; sell_price = None; sell_type = None
        entry_units_for_reward = self.entry_units

        # 獲取 T+1 價格
        price_T1_open = close_price_T; low_T1 = low_T; high_T1 = close_price_T; close_price_T1 = close_price_T
        if not terminated:
             next_data_T1 = self.data_df.iloc[next_step_idx]
             if next_data_T1 is not None:
                  price_T1_open = next_data_T1['open'] if pd.notna(next_data_T1['open']) and next_data_T1['open'] > 0 else price_T1_open
                  low_T1 = next_data_T1['low'] if pd.notna(next_data_T1['low']) and next_data_T1['low'] > 0 else low_T1
                  high_T1 = next_data_T1['high'] if pd.notna(next_data_T1['high']) and next_data_T1['high'] > 0 else high_T1
                  close_price_T1 = next_data_T1['close'] if pd.notna(next_data_T1['close']) and next_data_T1['close'] > 0 else close_price_T1

        # --- 處理 T+1 止盈止損 (如果持倉) ---
        if self.shares_held > 0:
            stop_loss_target = self.stop_loss_price; take_profit_target = self.take_profit_price
            if price_T1_open <= stop_loss_target: sell_price = price_T1_open; sell_type = 'stop_loss_gap'
            elif low_T1 <= stop_loss_target: sell_price = stop_loss_target; sell_type = 'stop_loss'
            elif price_T1_open >= take_profit_target: sell_price = price_T1_open; sell_type = 'take_profit_gap'
            elif high_T1 >= take_profit_target: sell_price = take_profit_target; sell_type = 'take_profit'

            if sell_type is not None:
                 # (執行賣出邏輯，與上版本一致)
                 shares_to_sell = self.shares_held; cost_basis = self.shares_held * self.entry_price
                 entry_units_for_reward = self.entry_units
                 simulated_sell_price = sell_price
                 if sell_type == 'stop_loss_gap': simulated_sell_price = price_T1_open
                 elif sell_type == 'take_profit_gap': simulated_sell_price = price_T1_open
                 elif sell_type == 'stop_loss': simulated_sell_price = stop_loss_target
                 elif sell_type == 'take_profit': simulated_sell_price = take_profit_target
                 if pd.notna(simulated_sell_price) and simulated_sell_price > 0:
                     proceeds = shares_to_sell * simulated_sell_price; realized_pnl = proceeds - cost_basis
                     self.cash += proceeds; print(f"      >> T+1 賣出 {shares_to_sell/self.shares_per_unit:.0f} 張 ({sell_type}) @ {simulated_sell_price:.2f}, PnL: {realized_pnl:.2f}")
                     self.shares_held = 0; self.entry_price = 0.0; self.stop_loss_price = 0.0; self.take_profit_price = 0.0; self.entry_units = 0
                     trade_executed_sell = True
                 else:
                     print(f"警告: T+1 賣出 ({sell_type}) 失敗，成交價無效({simulated_sell_price})，強制清倉記錄。"); realized_pnl = -cost_basis
                     self.shares_held = 0; self.entry_price = 0.0; self.stop_loss_price = 0.0; self.take_profit_price = 0.0; self.entry_units = 0
                     trade_executed_sell = True

        # --- 處理 T+1 買入 (如果 T日計劃且T+1未賣出) ---
        if entry_signal and planned_buy_units > 0 and not trade_executed_sell:
             # (買入邏輯與上版本一致)
             shares_to_buy = planned_buy_units * self.shares_per_unit
             transaction_price = price_T1_open
             if transaction_price > 0:
                 cost = shares_to_buy * transaction_price
                 if self.cash >= cost:
                     self.cash -= cost; self.shares_held = shares_to_buy
                     self.entry_price = transaction_price; self.stop_loss_price = next_stop_loss; self.take_profit_price = next_take_profit
                     self.entry_units = planned_buy_units; trade_executed_buy = True
                     print(f"      >> T+1 買入 {planned_buy_units} 張 @ {transaction_price:.2f} (SL:{self.stop_loss_price:.2f}, TP:{self.take_profit_price:.2f})")
                 else: print(f"      >> T+1 買入失敗 (現金不足) - 放棄訂單")
             else: print(f"      >> T+1 買入失敗 (價格無效) - 放棄訂單")

        # --- T+1 收盤後 ---
        self.current_step = next_step_idx
        self.portfolio_value = self._calculate_portfolio_value(self.current_step -1)

        # --- 計算獎勵 ---
        reward = 0.0
        trade_executed = trade_executed_buy or trade_executed_sell
        if trade_executed_sell:
             # (獎勵邏輯與上版本一致)
             initial_cost_guess = entry_units_for_reward * self.shares_per_unit * self.entry_price if self.entry_price > 0 else 1
             profit_ratio = realized_pnl / initial_cost_guess if initial_cost_guess != 0 else realized_pnl
             entry_units_factor = entry_units_for_reward # 張數本身作為因子
             if sell_type.startswith('take_profit'):
                 reward += self.profit_reward_factor * profit_ratio * entry_units_factor
             elif sell_type.startswith('stop_loss'):
                 reward -= self.stop_loss_penalty_factor * abs(profit_ratio) * entry_units_factor
             reward -= self.transaction_penalty_per_unit * entry_units_for_reward
        elif trade_executed_buy:
             reward -= self.transaction_penalty_per_unit * self.entry_units

        reward *= self.reward_scaling

        # --- 返回 ---
        observation = self._get_observation(self.current_step)
        terminated = self.current_step >= self.end_step; self.done = terminated
        info = { "step": self.current_step, "portfolio_value": self.portfolio_value, "cash": self.cash,
                 "shares_held": self.shares_held, "stop_loss": self.stop_loss_price, "take_profit": self.take_profit_price,
                 "entry_units": self.entry_units,
                 "placed_order_type": sell_type if trade_executed_sell else ('buy' if trade_executed_buy else 'hold'),
                 "trade_executed": trade_executed,
                 "reward": reward, "realized_pnl (step)": realized_pnl }
        return observation, reward, terminated, False, info

    def render(self, info=None): pass
    def close(self): pass

# --- 主執行區塊 (使用 EMA5RR Env, 修改買入單位) ---
if __name__ == '__main__':

    # --- Configuration ---
    API_ACCOUNT = "N26132089"; API_PASSWORD = "joshua900905"
    TARGET_STOCK_CODE = '2454'
    RUN_TRAINING = True; RUN_EVALUATION = False

    # --- Training Parameters ---
    START_DATE_TRAIN = '20200317'; END_DATE_TRAIN = '20240816'
    INITIAL_CAPITAL_PER_MODEL = 50000000.0; SHARES_PER_UNIT_TRAIN = 1000
    # Strategy Params
    EMA_PERIOD_TRAIN = 5; ATR_PERIOD_TRAIN = 14
    STOP_LOSS_ATR_MULT_TRAIN = 1.0; TAKE_PROFIT_ATR_MULT_TRAIN = 2.0
    BUY_UNITS_TRAIN = [1, 2] # <<< 修改為 1 或 2 張
    # Reward Params
    REWARD_SCALING_TRAIN = 1.0; TRANSACTION_PENALTY_PER_UNIT_TRAIN = 50
    FINAL_PROFIT_REWARD_FACTOR_TRAIN = 1.1
    STOP_LOSS_PENALTY_FACTOR_TRAIN = 1.2
    # Env Params
    WINDOW_SIZE_TRAIN = max(20, ATR_PERIOD_TRAIN + 1, EMA_PERIOD_TRAIN + 1)
    TOTAL_TIMESTEPS_PER_MODEL = 300000
    # PPO Params
    PPO_ENT_COEF = 0.01; PPO_LEARNING_RATE = 0.0003
    PPO_N_STEPS = 4096; PPO_BATCH_SIZE = 128
    # Output Dirs
    experiment_name = "ai_ema5_touch_atr_rr_units_1or2_v1" # <<< 更新實驗名稱
    MODELS_SAVE_DIR = f"tuned_models/{TARGET_STOCK_CODE}/{experiment_name}"
    TENSORBOARD_LOG_DIR = f"./tuning_tensorboard/{TARGET_STOCK_CODE}/{experiment_name}/"
    MONITOR_LOG_DIR = os.path.join(TENSORBOARD_LOG_DIR, "monitor_logs")
    VEC_NORMALIZE_STATS_PATH = os.path.join(MODELS_SAVE_DIR, "vecnormalize.pkl")

    if RUN_TRAINING:
        print(f"\n=============== 開始訓練 ({TARGET_STOCK_CODE} - {experiment_name}) ===============")
        if 'StockTradingEnvEMA5RR' not in globals() or not hasattr(StockTradingEnvEMA5RR, 'step'):
            print("\n錯誤：StockTradingEnvEMA5RR 類別未定義。\n"); exit()
        else:
            os.makedirs(MODELS_SAVE_DIR, exist_ok=True); os.makedirs(MONITOR_LOG_DIR, exist_ok=True)
            try:
                env_lambda = lambda: StockTradingEnvEMA5RR(
                    stock_code=TARGET_STOCK_CODE, start_date=START_DATE_TRAIN, end_date=END_DATE_TRAIN,
                    api_account=API_ACCOUNT, api_password=API_PASSWORD, initial_capital=INITIAL_CAPITAL_PER_MODEL,
                    ema_period=EMA_PERIOD_TRAIN,
                    atr_period=ATR_PERIOD_TRAIN, stop_loss_atr_multiplier=STOP_LOSS_ATR_MULT_TRAIN,
                    take_profit_atr_multiplier=TAKE_PROFIT_ATR_MULT_TRAIN,
                    buy_units=BUY_UNITS_TRAIN, # <<< 傳遞修改後的張數選項
                    reward_scaling=REWARD_SCALING_TRAIN, transaction_penalty_per_unit=TRANSACTION_PENALTY_PER_UNIT_TRAIN,
                    profit_reward_factor=FINAL_PROFIT_REWARD_FACTOR_TRAIN,
                    stop_loss_penalty_factor=STOP_LOSS_PENALTY_FACTOR_TRAIN,
                    window_size=WINDOW_SIZE_TRAIN, shares_per_unit=SHARES_PER_UNIT_TRAIN,
                    render_mode=None
                )
                monitor_path = os.path.join(MONITOR_LOG_DIR, f"{TARGET_STOCK_CODE}")
                os.makedirs(os.path.dirname(monitor_path), exist_ok=True)
                monitored_env_lambda = lambda: Monitor(env_lambda(), monitor_path, allow_early_resets=True)
                dummy_vec_env = DummyVecEnv([monitored_env_lambda])
                print("--- Monitor 環境已啟用 ---")
                vec_env = VecNormalize(dummy_vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
                print("--- VecNormalize 環境已啟用 (訓練模式) ---")

                model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=TENSORBOARD_LOG_DIR, seed=42,
                            ent_coef=PPO_ENT_COEF, learning_rate=PPO_LEARNING_RATE,
                            n_steps=PPO_N_STEPS, batch_size=PPO_BATCH_SIZE,
                           )

                print(f"  開始訓練 {TOTAL_TIMESTEPS_PER_MODEL} 步...")
                model.learn(total_timesteps=TOTAL_TIMESTEPS_PER_MODEL, log_interval=100)
                print(f"  訓練完成。")

                vec_env.save(VEC_NORMALIZE_STATS_PATH)
                print(f"  VecNormalize 統計數據已儲存: {VEC_NORMALIZE_STATS_PATH}")
                save_path = os.path.join(MODELS_SAVE_DIR, f"ppo_agent_{TARGET_STOCK_CODE}_final")
                model.save(save_path); print(f"  最終模型已儲存: {save_path}.zip")

                vec_env.close()

            except ValueError as e: print(f"股票 {TARGET_STOCK_CODE} 環境初始化或數據錯誤: {e}")
            except Exception as e: print(f"訓練股票 {TARGET_STOCK_CODE} 時發生未預期的錯誤: {e}"); traceback.print_exc();
        print("\n=============== 單股票訓練階段完成 ===============")

    if RUN_EVALUATION: print("\n錯誤：此腳本僅用於訓練。")
    if not RUN_TRAINING and not RUN_EVALUATION: print("\n請設置 RUN_TRAINING = True 來執行。")
    print("\n--- 程序執行完畢 ---")