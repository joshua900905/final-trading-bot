# -*- coding: utf-8 -*-
# train_pos_ratio_model.py - Train model to decide position ratio (NO RSI)
# (Features: MA10, MA20, MACD, ATR; Simplified Reward + Trend Align Reward; Monitor)

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
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    print("錯誤：找不到 stable_baselines3 或其組件。請運行 'pip install stable_baselines3[extra]'")
    exit()

# --- Stock API Class ---
class Stock_API:
    """Stock API Class - Buy/Sell methods won't be called by TradeExecutor in backtest mode."""
    def __init__(self, account, password): self.account = account; self.password = password; self.base_url = 'http://140.116.86.242:8081/stock/api/v1'
    def Get_Stock_Informations(self, stock_code, start_date, stop_date):
        information_url = (f"{self.base_url}/api_get_stock_info_from_date_json/{stock_code}/{start_date}/{stop_date}");
        try: 
            response = requests.get(information_url, timeout=15); response.raise_for_status(); result = response.json();
            if result.get('result') == 'success': data = result.get('data', []); return data if isinstance(data, list) else []
            else: print(f"API 錯誤 (Info - {stock_code}): {result.get('status', '未知')}"); return []
        except Exception as e: print(f"Get_Stock_Informations 出錯 ({stock_code}): {e}"); return []
    def Get_User_Stocks(self): print("警告：Get_User_Stocks 在純回測模式下不應被調用。"); return []
    def Buy_Stock(self, stock_code, stock_shares, stock_price): print(f"警告：Buy_Stock 在純回測模式下不應被調用 ({stock_code})。"); return False
    def Sell_Stock(self, stock_code, stock_shares, stock_price): print(f"警告：Sell_Stock 在純回測模式下不應被調用 ({stock_code})。"); return False

# --- StockTradingEnv Class (AI learns Position Ratio, Indicator Exit, Trend Reward, No RSI) ---
class StockTradingEnvPosRatio(gym.Env):
    """
    訓練 AI 決定開倉倉位比例的環境。移除 RSI 特徵。
    """
    metadata = {'render_modes': ['human', None], 'render_fps': 1}

    # --- 修改: 移除 rsi_period ---
    def __init__(self, stock_code, start_date, end_date, api_account, api_password,
                 initial_capital=1000000,
                 ma_short=10, ma_medium=20, atr_period=14, # <<<--- 移除 rsi_period
                 macd_fast=12, macd_slow=26, macd_signal=9,
                 reward_scaling=1.0,
                 transaction_penalty=0.005,
                 position_ratios = [0.1, 0.2, 0.3],
                 shares_per_level = 1000,
                 holding_trend_bonus_factor = 0.01,
                 holding_counter_trend_penalty_factor = 0.02,
                 render_mode=None):
        super().__init__()
        self.stock_code = stock_code; self.start_date_str = start_date; self.end_date_str = end_date
        self.api = Stock_API(api_account, api_password); self.initial_capital = initial_capital;
        self.ma_short_period = ma_short; self.ma_medium_period = ma_medium
        self.atr_period = atr_period # rsi_period 已移除
        self.macd_fast = macd_fast; self.macd_slow = macd_slow; self.macd_signal = macd_signal
        # --- 修改: window_size 不再依賴 rsi_period ---
        self.window_size = max(self.ma_short_period, self.ma_medium_period, self.atr_period, self.macd_slow) + 10
        # ---
        self.reward_scaling = reward_scaling; self.transaction_penalty = transaction_penalty
        self.position_ratios = position_ratios; self.shares_per_level = shares_per_level
        if not self.position_ratios or not all(0 < r <= 1 for r in self.position_ratios): raise ValueError("position_ratios must contain only positive values <= 1.")
        self.holding_trend_bonus_factor = holding_trend_bonus_factor
        self.holding_counter_trend_penalty_factor = holding_counter_trend_penalty_factor
        self.render_mode = render_mode
        self.data_df = self._load_and_preprocess_data(start_date, end_date)
        if self.data_df is None or len(self.data_df) < self.window_size: actual_len = len(self.data_df) if self.data_df is not None else 0; raise ValueError(f"股票 {stock_code} 數據不足 (需 {self.window_size}, 實 {actual_len})")
        self.end_step = len(self.data_df) - 1;
        self.action_space = spaces.Discrete(len(self.position_ratios))

        # --- 修改: 觀察空間維度為 9 ---
        self.features_per_stock = 8 # MA Ratios(3) + ATRn(1) + MACD(3) + Holding(1) + SL/TP (2, placeholder?)
        self.observation_shape = (self.features_per_stock,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.observation_shape, dtype=np.float32)
        # ---

        self.current_step = 0; self.cash = 0.0; self.shares_held = 0; self.entry_price = 0.0; self.entry_atr = 0.0;
        self.portfolio_value = 0.0; self.done = False;

    # --- 修改: _load_and_preprocess_data 移除 RSI 計算 ---
    def _load_and_preprocess_data(self, start_date, end_date):
        # print(f"  StockTradingEnv ({self.stock_code}): 載入數據 {start_date} to {end_date}")
        try: start_dt_obj = pd.to_datetime(start_date, format='%Y%m%d'); buffer_days = 30; required_start_dt = start_dt_obj - pd.Timedelta(days=(self.window_size + buffer_days) * 1.5); required_start_date_str = required_start_dt.strftime('%Y%m%d')
        except ValueError: print(f"    錯誤：起始日期格式無效 {start_date}"); return None
        raw_data = self.api.Get_Stock_Informations(self.stock_code, required_start_date_str, end_date)
        if not raw_data: return None
        try:
            df = pd.DataFrame(raw_data); df['date'] = pd.to_datetime(df['date'], unit='s'); df = df.sort_values('date'); df = df[~df['date'].duplicated(keep='first')]; df = df.set_index('date')
            required_cols = ['open', 'high', 'low', 'close', 'turnover'];
            if not all(col in df.columns for col in required_cols): return None
            numeric_cols = ['open', 'high', 'low', 'close', 'turnover', 'capacity', 'transaction_volume']
            for col in numeric_cols:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
            if 'turnover' in df.columns and 'close' in df.columns: df['volume'] = np.where(df['close'].isna() | (df['close'] == 0), 0, df['turnover'] / df['close']); df['volume'] = df['volume'].fillna(0).replace([np.inf, -np.inf], 0).round().astype(np.int64)
            else: df['volume'] = 0
            indicator_base_cols = ['open', 'high', 'low', 'close']; df = df.dropna(subset=indicator_base_cols)
            if df.empty: return None
            df.ta.sma(length=self.ma_short_period, close='close', append=True, col_names=(f'SMA_{self.ma_short_period}',))
            df.ta.sma(length=self.ma_medium_period, close='close', append=True, col_names=(f'SMA_{self.ma_medium_period}',))
            # --- 移除 RSI ---
            # df.ta.rsi(length=self.rsi_period, close='close', append=True, col_names=(f'RSI_{self.rsi_period}',))
            # ---
            df.ta.atr(length=self.atr_period, high='high', low='low', close='close', append=True, col_names=(f'ATR_{self.atr_period}',))
            macd_df = df.ta.macd(fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal, close='close')
            if macd_df is None or macd_df.empty: return None
            macd_df.columns = ['MACD', 'MACDh', 'MACDs']; df = pd.concat([df, macd_df], axis=1)
            if f'ATR_{self.atr_period}' not in df.columns: return None
            df[f'ATR_norm_{self.atr_period}'] = df[f'ATR_{self.atr_period}'] / df['close']; df[f'ATR_norm_{self.atr_period}'] = df[f'ATR_norm_{self.atr_period}'].replace([np.inf, -np.inf], 0)
            df = df.dropna();
            if df.empty: return None
            df_filtered = df[df.index >= start_dt_obj]
            if len(df_filtered) < self.window_size : return None
            print(f"    > {self.stock_code} 數據處理完成，用於模擬的數據: {len(df_filtered)} 行")
            return df_filtered
        except Exception as e: print(f"    StockTradingEnv ({self.stock_code}): 處理數據時出錯: {e}"); traceback.print_exc(); return None

    # --- 修改: _get_observation 計算 9 維特徵 ---
    def _get_observation(self, step):
        if step < 0 or step >= len(self.data_df): return np.zeros(self.observation_shape, dtype=np.float32)
        try:
            obs_data = self.data_df.iloc[step]; close_price = obs_data['close']
            atr_val = obs_data.get(f'ATR_{self.atr_period}', 0.0); atr_norm_val = obs_data.get(f'ATR_norm_{self.atr_period}', 0.0)
            ma10_val = obs_data.get(f'SMA_{self.ma_short_period}', close_price); ma20_val = obs_data.get(f'SMA_{self.ma_medium_period}', close_price)
            # --- 移除 RSI ---
            macd_val = obs_data.get('MACD', 0.0); macdh_val = obs_data.get('MACDh', 0.0); macds_val = obs_data.get('MACDs', 0.0)
            macd_norm = macd_val / close_price if close_price != 0 else 0.0; macdh_norm = macdh_val / close_price if close_price != 0 else 0.0; macds_norm = macds_val / close_price if close_price != 0 else 0.0
            price_ma10_ratio = close_price / ma10_val if ma10_val != 0 else 1.0; price_ma20_ratio = close_price / ma20_val if ma20_val != 0 else 1.0; ma10_ma20_ratio = ma10_val / ma20_val if ma20_val != 0 else 1.0
            holding_position = 1.0 if self.shares_held > 0 else 0.0
            # --- 組裝 9 個特徵 ---
            features = [
                price_ma10_ratio, price_ma20_ratio, ma10_ma20_ratio, # MA Ratios (3)
                atr_norm_val,                                       # Volatility (1)
                macd_norm, macdh_norm, macds_norm,                  # MACD (3)
                holding_position                                    # Holding (1)
            ]
            observation = np.array(features, dtype=np.float32); observation = np.nan_to_num(observation, nan=0.0, posinf=1e9, neginf=-1e9);
            if observation.shape[0] != self.features_per_stock: print(f"錯誤: 觀察值維度 ({observation.shape[0]}) 與預期 ({self.features_per_stock}) 不符！"); return np.zeros(self.observation_shape, dtype=np.float32)
            return observation
        except Exception as e: print(f"錯誤 ({self.stock_code}): Obs 未知錯誤 step {step}: {e}"); traceback.print_exc(); return np.zeros(self.observation_shape, dtype=np.float32)

    # _calculate_portfolio_value, reset, _check_entry_signal, _check_exit_signal 保持不變
    def _calculate_portfolio_value(self, step):
        if step < 0 or step >= len(self.data_df): return self.portfolio_value
        close_price = self.data_df.iloc[step]['close']; stock_value = self.shares_held * close_price; return self.cash + stock_value
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.data_df is None or len(self.data_df) < self.window_size: raise RuntimeError(f"無法重設環境 ({self.stock_code}): 數據不足。")
        self.current_step = 0; self.cash = self.initial_capital; self.shares_held = 0; self.entry_price = 0.0; self.entry_atr = 0.0;
        self.portfolio_value = self.initial_capital; self.done = False;
        observation = self._get_observation(self.current_step); info = {"message": f"{self.stock_code} Env reset"}; return observation, info
    def _check_entry_signal(self, current_step):
        if current_step < 1: return False
        today_data = self.data_df.iloc[current_step]; yesterday_data = self.data_df.iloc[current_step - 1]
        ma10_today = today_data.get(f'SMA_{self.ma_short_period}', np.nan); ma20_today = today_data.get(f'SMA_{self.ma_medium_period}', np.nan)
        ma10_yesterday = yesterday_data.get(f'SMA_{self.ma_short_period}', np.nan); ma20_yesterday = yesterday_data.get(f'SMA_{self.ma_medium_period}', np.nan)
        close_today = today_data['close']
        if pd.isna(ma10_today) or pd.isna(ma20_today) or pd.isna(ma10_yesterday) or pd.isna(ma20_yesterday) or pd.isna(close_today): return False
        crossed_up = ma10_yesterday <= ma20_yesterday and ma10_today > ma20_today; price_above_ma20 = close_today > ma20_today
        if crossed_up and price_above_ma20: return True
        return False
    def _check_exit_signal(self, current_step):
        if current_step < 1: return False
        today_data = self.data_df.iloc[current_step]; yesterday_data = self.data_df.iloc[current_step - 1]
        ma10_today = today_data.get(f'SMA_{self.ma_short_period}', np.nan); ma20_today = today_data.get(f'SMA_{self.ma_medium_period}', np.nan)
        ma10_yesterday = yesterday_data.get(f'SMA_{self.ma_short_period}', np.nan); ma20_yesterday = yesterday_data.get(f'SMA_{self.ma_medium_period}', np.nan)
        if pd.isna(ma10_today) or pd.isna(ma20_today) or pd.isna(ma10_yesterday) or pd.isna(ma20_yesterday): return False
        crossed_down = ma10_yesterday >= ma20_yesterday and ma10_today < ma20_today
        if crossed_down: return True
        return False

    # step 方法修改: 獎勵函數移除 sl_penalty_factor
    def step(self, action):
        if self.done: obs = self._get_observation(self.current_step if self.current_step < len(self.data_df) else self.current_step - 1); return obs, 0.0, True, False, {"message": "回合已結束."}
        previous_portfolio_value = self.portfolio_value; previous_shares_held = self.shares_held;
        current_data_T = self.data_df.iloc[self.current_step]; close_price_T = current_data_T['close'];
        atr_T = current_data_T.get(f'ATR_{self.atr_period}', 0.0)
        placed_order_type, trade_executed = 'hold', False
        shares_to_buy_sim, shares_to_sell_sim = 0, 0
        entry_signal_triggered = self._check_entry_signal(self.current_step)
        exit_signal_triggered = self._check_exit_signal(self.current_step)
        if entry_signal_triggered and self.shares_held == 0:
            if action >= 0 and action < len(self.position_ratios):
                 target_ratio = self.position_ratios[action]; target_capital = self.portfolio_value * target_ratio
                 max_capital_stock = self.initial_capital / 20.0; target_capital = min(target_capital, max_capital_stock)
                 target_shares = math.floor(target_capital / close_price_T / self.shares_per_level) * self.shares_per_level if close_price_T > 0 else 0
                 estimated_cost = target_shares * close_price_T
                 if self.cash >= estimated_cost and target_shares > 0: shares_to_buy_sim = target_shares; placed_order_type = f'buy_{target_ratio*100:.0f}%'; trade_executed = True
                 else: placed_order_type = 'hold_cant_buy_cash'
            else: placed_order_type = 'hold_invalid_action'
        elif exit_signal_triggered and self.shares_held > 0:
            shares_to_sell_sim = self.shares_held; placed_order_type = 'sell_indicator'; trade_executed = True
        else: placed_order_type = 'hold'
        self.current_step += 1; terminated = self.current_step >= self.end_step; truncated = False
        realized_pnl = 0; cost_basis_before_sell = 0;
        if not terminated:
             current_data_T1 = self.data_df.iloc[self.current_step]; price_T1_open = current_data_T1['open'];
             atr_T1 = current_data_T1.get(f'ATR_{self.atr_period}', 0.0); close_price_T1 = current_data_T1['close'];
             if placed_order_type.startswith('buy') and trade_executed:
                  cost = shares_to_buy_sim * price_T1_open
                  if self.cash >= cost: self.cash -= cost; self.shares_held = shares_to_buy_sim; self.entry_price, self.entry_atr = price_T1_open, atr_T1
                  else: trade_executed = False
             elif placed_order_type == 'sell_indicator' and trade_executed:
                  cost_basis_before_sell = shares_to_sell_sim * self.entry_price; proceeds = shares_to_sell_sim * price_T1_open
                  realized_pnl = proceeds - cost_basis_before_sell; self.cash += proceeds; self.shares_held = 0; self.entry_price, self.entry_atr = 0.0, 0.0
             self.portfolio_value = self._calculate_portfolio_value(self.current_step)
        else: self.portfolio_value = self._calculate_portfolio_value(self.current_step - 1);
        reward = 0.0
        if previous_portfolio_value != 0: reward = (self.portfolio_value - previous_portfolio_value) / previous_portfolio_value
        elif self.initial_capital != 0: reward = (self.portfolio_value - self.initial_capital) / self.initial_capital if self.initial_capital else 0
        # --- 簡化獎勵: 只保留交易成本 + 趨勢獎懲 ---
        if trade_executed: reward -= self.transaction_penalty
        if self.shares_held > 0:
            data_now = self.data_df.iloc[self.current_step if not terminated else self.current_step -1]
            close_now = data_now['close']; ma10_now = data_now.get(f'SMA_{self.ma_short_period}', close_now); ma20_now = data_now.get(f'SMA_{self.ma_medium_period}', close_now)
            is_uptrend = close_now > ma10_now and ma10_now > ma20_now
            if is_uptrend: reward += self.holding_trend_bonus_factor
            else: reward -= self.holding_counter_trend_penalty_factor
        reward *= self.reward_scaling
        observation = self._get_observation(self.current_step if not terminated else self.current_step - 1); self.done = terminated
        info = { "step": self.current_step, "portfolio_value": self.portfolio_value, "cash": self.cash, "shares_held": self.shares_held, "placed_order_type": placed_order_type, "trade_executed (simulated)": trade_executed, "reward": reward, "realized_pnl": realized_pnl}
        return observation, reward, terminated, truncated, info

    def render(self, info=None): pass
    def close(self): pass

# --- Main Execution Block for Single Stock Tuning ---
if __name__ == '__main__':

    # --- Configuration ---
    API_ACCOUNT = "N26132089"
    API_PASSWORD = "joshua900905"
    TARGET_STOCK_CODE = '2454'

    RUN_TRAINING = True
    RUN_EVALUATION = False

    # --- Training Parameters (AI Pos Ratio, Indicator Exit, Trend Reward/Penalty, No RSI) ---
    START_DATE_TRAIN = '20160101'
    END_DATE_TRAIN = '20191231'
    INITIAL_CAPITAL_PER_MODEL = 50000000.0
    POSITION_RATIOS_TRAIN = [0.1, 0.2, 0.3]
    SHARES_PER_LEVEL_TRAIN = 1000
    MA_SHORT_TRAIN = 10; MA_MEDIUM_TRAIN = 20; ATR_PERIOD_TRAIN = 14 # <<<--- 移除 RSI_PERIOD_TRAIN
    MACD_FAST_TRAIN = 12; MACD_SLOW_TRAIN = 26; MACD_SIGNAL_TRAIN = 9
    # SL/TP Multiplier 不再需要
    # WINDOW_SIZE is calculated in Env
    TOTAL_TIMESTEPS_PER_MODEL = 50000
    # --- 簡化獎勵係數 + 趨勢獎懲 ---
    REWARD_SCALING_TRAIN = 1.0
    SL_PENALTY_FACTOR_TRAIN = 0.0      # <<<--- 移除
    TRANSACTION_PENALTY_TRAIN = 0.005
    HOLDING_TREND_BONUS_FACTOR_TRAIN = 0.01
    HOLDING_COUNTER_TREND_PENALTY_FACTOR = 0.02
    # --- PPO 超參數 ---
    PPO_ENT_COEF = 0.01
    PPO_LEARNING_RATE = 0.0003
    PPO_N_STEPS = 4096
    PPO_BATCH_SIZE = 128
    # --- 輸出目錄 ---
    experiment_name = "ai_pos_ratio_indicator_exit_trend_reward_no_rsi_v1" # 新實驗名稱
    MODELS_SAVE_DIR = f"tuned_models/{TARGET_STOCK_CODE}/{experiment_name}"
    TENSORBOARD_LOG_DIR = f"./tuning_tensorboard/{TARGET_STOCK_CODE}/{experiment_name}/"
    MONITOR_LOG_DIR = os.path.join(TENSORBOARD_LOG_DIR, "monitor_logs")

    if RUN_TRAINING:
        print(f"\n=============== 開始單股票調試訓練 ({TARGET_STOCK_CODE} - {experiment_name}) ===============")
        if 'StockTradingEnvPosRatio' not in globals() or not hasattr(StockTradingEnvPosRatio, 'step'): print("\n錯誤：StockTradingEnvPosRatio 類別未定義。\n"); exit()
        else:
            os.makedirs(MODELS_SAVE_DIR, exist_ok=True); os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True); os.makedirs(MONITOR_LOG_DIR, exist_ok=True)
            try:
                env = StockTradingEnvPosRatio(
                    stock_code=TARGET_STOCK_CODE, start_date=START_DATE_TRAIN, end_date=END_DATE_TRAIN,
                    api_account=API_ACCOUNT, api_password=API_PASSWORD, initial_capital=INITIAL_CAPITAL_PER_MODEL,
                    ma_short=MA_SHORT_TRAIN, ma_medium=MA_MEDIUM_TRAIN, atr_period=ATR_PERIOD_TRAIN, # <<<--- 移除 rsi_period
                    macd_fast=MACD_FAST_TRAIN, macd_slow=MACD_SLOW_TRAIN, macd_signal=MACD_SIGNAL_TRAIN,
                    reward_scaling=REWARD_SCALING_TRAIN,
                    transaction_penalty=TRANSACTION_PENALTY_TRAIN, # 只傳遞需要的獎勵參數
                    position_ratios=POSITION_RATIOS_TRAIN, shares_per_level=SHARES_PER_LEVEL_TRAIN,
                    holding_trend_bonus_factor = HOLDING_TREND_BONUS_FACTOR_TRAIN,
                    holding_counter_trend_penalty_factor = HOLDING_COUNTER_TREND_PENALTY_FACTOR,
                    render_mode=None )
                monitor_path = os.path.join(MONITOR_LOG_DIR, f"{TARGET_STOCK_CODE}"); env = Monitor(env, monitor_path, allow_early_resets=True)
                vec_env = DummyVecEnv([lambda: env])
                policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])
                model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=TENSORBOARD_LOG_DIR, seed=42,
                            ent_coef=PPO_ENT_COEF, learning_rate=PPO_LEARNING_RATE, n_steps=PPO_N_STEPS, batch_size=PPO_BATCH_SIZE,
                            policy_kwargs=policy_kwargs )

                print(f"  開始訓練 {TOTAL_TIMESTEPS_PER_MODEL} 步...")
                model.learn(total_timesteps=TOTAL_TIMESTEPS_PER_MODEL, log_interval=1)
                print(f"  訓練完成。")
                save_path = os.path.join(MODELS_SAVE_DIR, f"ppo_agent_{TARGET_STOCK_CODE}_final")
                model.save(save_path); print(f"  最終模型已儲存: {save_path}.zip")
                vec_env.close()
            except ValueError as e: print(f"股票 {TARGET_STOCK_CODE} 環境初始化或數據錯誤: {e}")
            except Exception as e: print(f"訓練股票 {TARGET_STOCK_CODE} 時發生未預期的錯誤: {e}"); traceback.print_exc();
        print("\n=============== 單股票調試訓練階段完成 ===============")

    if RUN_EVALUATION: print("\n錯誤：此腳本僅用於訓練調試。")
    if not RUN_TRAINING and not RUN_EVALUATION: print("\n請設置 RUN_TRAINING = True 來執行。")
    print("\n--- 程序執行完畢 ---")