# -*- coding: utf-8 -*-
# evaluate_ema5_touch_atr_rr_v2_2units.py - Evaluate EMA5 Touch Strategy (AI Units 1/2, ATR TP/SL)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import pandas_ta as ta # Needed for EMA and ATR
import requests
import time
import os
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import traceback
from datetime import datetime

# --- Stable Baselines 3 Import ---
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError:
    print("錯誤：找不到 stable_baselines3 或其組件。請運行 'pip install stable_baselines3[extra]'")
    exit()

# --- Stock API Class (保持不變) ---
class Stock_API:
    """Stock API Class"""
    def __init__(self, account, password): self.account = account; self.password = password; self.base_url = 'http://140.116.86.242:8081/stock/api/v1'
    def Get_Stock_Informations(self, stock_code, start_date, stop_date):
        information_url = (f"{self.base_url}/api_get_stock_info_from_date_json/{stock_code}/{start_date}/{stop_date}");
        try:
            response = requests.get(information_url, timeout=15); response.raise_for_status(); result = response.json();
            if result.get('result') == 'success': data = result.get('data', []); return data if isinstance(data, list) else []
            else: print(f"API 錯誤 (Info - {stock_code}): {result.get('status', '未知')}"); return []
        except Exception as e: print(f"Get_Stock_Informations 出錯 ({stock_code}): {e}"); return []
    # Other methods remain same...


# --- Refactored Evaluation Classes ---

class DataManager:
    """數據管理器 (計算 EMA 和 ATR)"""
    def __init__(self, stock_codes_initial, api, window_size,
                 ema_period=5, atr_period=14): # <<< 使用 EMA 週期
        self.stock_codes_initial = list(stock_codes_initial); self.api = api;
        self.ema_period = ema_period; self.ema_col_name = f'EMA_{self.ema_period}' # <<<
        self.atr_period = atr_period; self.atr_col_name = f'ATR_{self.atr_period}'
        self.window_size = window_size
        self.data_dict = {}; self.common_dates = None; self.stock_codes = []

    def _load_and_preprocess_single_stock(self, stock_code, start_date, end_date):
        # 計算 EMA 和 ATR
        raw_data = self.api.Get_Stock_Informations(stock_code, start_date, end_date);
        if not raw_data: return None
        try:
            df = pd.DataFrame(raw_data); df['date'] = pd.to_datetime(df['date'], unit='s'); df = df.sort_values('date'); df = df[~df['date'].duplicated(keep='first')]; df = df.set_index('date')
            required_cols = ['open', 'high', 'low', 'close'];
            if not all(col in df.columns for col in required_cols): return None
            numeric_cols = ['open', 'high', 'low', 'close', 'turnover', 'capacity', 'transaction_volume']
            for col in numeric_cols:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
            indicator_base_cols = ['open', 'high', 'low', 'close']; df = df.dropna(subset=indicator_base_cols)
            if df.empty: return None

            # 計算 EMA (不需要 shift，因為回測會在 T 日使用 T 日的 EMA)
            df.ta.ema(length=self.ema_period, close='close', append=True, col_names=(self.ema_col_name,))

            # 計算 ATR (不需要 shift)
            df.ta.atr(length=self.atr_period, append=True, col_names=(self.atr_col_name,))

            df = df.dropna();
            if df.empty: return None
            return df
        except Exception as e: print(f"DataManager: 處理 {stock_code} 數據時出錯: {e}"); traceback.print_exc(); return None

    def load_all_data(self, start_date, end_date):
        # (與之前版本類似)
        print(f"DataManager: 正在載入數據 ({start_date} to {end_date}) for {len(self.stock_codes_initial)} stocks...")
        temp_data_dict, temp_common_dates, successful_codes = {}, None, []
        try: start_dt = pd.to_datetime(start_date, format='%Y%m%d'); buffer_days = max(self.ema_period, self.atr_period) + 5; required_start_dt = start_dt - pd.Timedelta(days=buffer_days * 1.5); required_start_date_str = required_start_dt.strftime('%Y%m%d')
        except ValueError: print("錯誤：起始日期格式無效。"); return False
        for code in self.stock_codes_initial:
            df = self._load_and_preprocess_single_stock(code, required_start_date_str, end_date) # 調用包含 EMA, ATR 的版本
            if df is not None and not df.empty:
                df_filtered = df[df.index >= start_dt]
                if not df_filtered.empty and len(df_filtered) >= self.window_size : # window_size 確保 VecNormalize 有足夠數據
                    temp_data_dict[code] = df_filtered; successful_codes.append(code)
                    if temp_common_dates is None: temp_common_dates = df_filtered.index
                    else: temp_common_dates = temp_common_dates.intersection(df_filtered.index)
        if not temp_data_dict or not successful_codes: print("DataManager 錯誤：沒有任何股票數據成功載入。"); return False
        self.stock_codes = successful_codes; self.data_dict = temp_data_dict
        if temp_common_dates is None or len(temp_common_dates) == 0: print("DataManager 錯誤：找不到共同交易日期。"); return False
        self.common_dates = temp_common_dates.sort_values()
        if len(self.common_dates) < self.window_size + 1: print(f"DataManager 錯誤：共同交易日數據量不足"); return False
        print(f"DataManager: 數據載入完成，找到 {len(self.common_dates)} 個共同交易日。")
        return True

    # Getters
    def get_common_dates(self): return self.common_dates
    def get_stock_codes(self): return self.stock_codes
    def get_data_on_date(self, stock_code, date):
        if stock_code in self.data_dict and date in self.data_dict[stock_code].index:
            data_slice = self.data_dict[stock_code].loc[[date]]; return data_slice.iloc[0] if not data_slice.empty else None
        else: return None
    def get_indicator_periods(self): return {'ema': self.ema_period, 'atr': self.atr_period} # <<< 返回 EMA 週期
    def get_atr_col_name(self): return self.atr_col_name
    def get_ema_col_name(self): return self.ema_col_name # <<< 返回 EMA 列名


class PortfolioManager:
    """投資組合管理器 (用於 ATR 止盈止損策略)"""
    # (與之前的 ATR 止盈止損版本類似)
    def __init__(self, initial_capital, stock_codes):
        self.initial_capital = initial_capital; self.stock_codes = list(stock_codes);
        if len(self.stock_codes) != 1: print("警告：PortfolioManager 設計為單股票回測。")
        self.target_code = self.stock_codes[0] if self.stock_codes else None
        self.cash = initial_capital
        self.shares_held = defaultdict(int)
        self.entry_price = defaultdict(float)
        self.stop_loss_price = defaultdict(float)
        self.take_profit_price = defaultdict(float) # <<< 止盈價
        self.entry_units = defaultdict(int)        # <<< 進場張數
        self.portfolio_value = initial_capital

    def reset(self):
        self.cash = self.initial_capital; self.shares_held = defaultdict(int)
        self.entry_price = defaultdict(float); self.stop_loss_price = defaultdict(float)
        self.take_profit_price = defaultdict(float); self.entry_units = defaultdict(int)
        self.portfolio_value = self.initial_capital
        print(f"PortfolioManager ({self.target_code}): 狀態已重設。")

    def update_on_buy(self, stock_code, shares_bought, entry_units, cost, stop_loss, take_profit): # <<< 添加 entry_units
        if stock_code != self.target_code or shares_bought <= 0: return
        self.cash -= cost
        self.entry_price[stock_code] = cost / shares_bought if shares_bought > 0 else 0
        self.shares_held[stock_code] = shares_bought
        self.stop_loss_price[stock_code] = stop_loss
        self.take_profit_price[stock_code] = take_profit # <<<
        self.entry_units[stock_code] = entry_units       # <<<

    def update_on_sell(self, stock_code, shares_sold, proceeds): # <<< 賣出即清空
        if stock_code != self.target_code or shares_sold <= 0: return
        if shares_sold > self.shares_held[stock_code]: shares_sold = self.shares_held[stock_code]
        self.cash += proceeds
        self.shares_held[stock_code] = 0 # 直接清空
        self.entry_price[stock_code] = 0.0
        self.stop_loss_price[stock_code] = 0.0
        self.take_profit_price[stock_code] = 0.0
        self.entry_units[stock_code] = 0

    def calculate_and_update_portfolio_value(self, data_manager: DataManager, current_date):
        # (與之前版本一致)
        # ...
        total_stock_value = 0.0;
        if self.target_code:
             shares = self.shares_held[self.target_code]
             if shares > 0:
                data = data_manager.get_data_on_date(self.target_code, current_date)
                if data is not None and pd.notna(data['close']) and data['close'] > 0: total_stock_value += shares * data['close']
                else:
                    common_dates_list = data_manager.get_common_dates();
                    try:
                         if isinstance(common_dates_list, pd.DatetimeIndex):
                             current_idx = common_dates_list.get_loc(current_date); prev_idx = current_idx - 1
                             if prev_idx >= 0:
                                 prev_date = common_dates_list[prev_idx]
                                 prev_data = data_manager.get_data_on_date(self.target_code, prev_date)
                                 if prev_data is not None and pd.notna(prev_data['close']) and prev_data['close'] > 0: total_stock_value += shares * prev_data['close']
                    except Exception: pass
        self.portfolio_value = self.cash + total_stock_value;
        return self.portfolio_value


    # Getters
    def get_cash(self): return self.cash
    def get_shares(self, stock_code): return self.shares_held.get(stock_code, 0)
    def get_portfolio_value(self): return self.portfolio_value
    def get_entry_price(self, stock_code): return self.entry_price.get(stock_code, 0.0)
    def get_stop_loss_price(self, stock_code): return self.stop_loss_price.get(stock_code, 0.0)
    def get_take_profit_price(self, stock_code): return self.take_profit_price.get(stock_code, 0.0) # <<<
    def get_entry_units(self, stock_code): return self.entry_units.get(stock_code, 0) # <<<


class TradeExecutor:
    """交易執行器 (處理固定張數買單和完全賣單)"""
    def __init__(self, api: Stock_API, portfolio_manager: PortfolioManager, data_manager: DataManager,
                 buy_units=[1, 2], # <<< AI 決定的張數選項
                 stop_loss_atr_multiplier=1.0, take_profit_atr_multiplier=2.0,
                 shares_per_unit=1000):
        self.api = api; self.portfolio_manager = portfolio_manager; self.data_manager = data_manager
        self.buy_units = buy_units # <<<
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.take_profit_atr_multiplier = take_profit_atr_multiplier
        self.shares_per_unit = shares_per_unit # <<<

    def place_sell_orders(self, sell_requests): # sell_requests: (code, shares, price, sell_type)
        # (與之前版本類似，但只有完全賣出)
        simulated_success_map = {};
        if not sell_requests: return simulated_success_map
        print("TradeExecutor: [BACKTEST] 模擬提交賣單...")
        for code, shares_api, price, sell_type in sell_requests:
            sheets = shares_api / self.shares_per_unit
            print(f"  [BACKTEST] 模擬提交賣單 ({sell_type}): {sheets:.0f} 張 {code}")
            simulated_success_map[(code, sell_type)] = True # 假設總是成功
        return simulated_success_map

    def determine_and_place_buy_orders(self, buy_decision: dict, date_T): # buy_decision: {code: action_idx}
        simulated_success_map = {}; orders_to_submit_buy = [] # (code, shares_api, entry_units, price_T, stop_loss, take_profit)
        if not buy_decision: return orders_to_submit_buy, simulated_success_map

        available_cash = self.portfolio_manager.get_cash()
        print(f"TradeExecutor: [BACKTEST] 處理 AI 張數買入請求... (可用現金: {available_cash:.2f})")

        for code, action_idx in buy_decision.items():
             if code != self.portfolio_manager.target_code: continue
             if action_idx < 0 or action_idx >= len(self.buy_units):
                 print(f"  > {code}: AI 返回無效動作索引 {action_idx}，跳過。")
                 continue

             planned_buy_units = self.buy_units[action_idx]
             shares_to_buy_api = planned_buy_units * self.shares_per_unit

             data_T = self.data_manager.get_data_on_date(code, date_T)
             price_T = data_T['close'] if data_T is not None and pd.notna(data_T['close']) else 0
             atr_T = data_T.get(self.data_manager.get_atr_col_name(), 0.0) if data_T is not None else 0.0
             if price_T <= 0 or atr_T <= 0:
                 print(f"  > {code}: 無效價格或ATR ({price_T:.2f}, {atr_T:.2f})，無法計算止損止盈。")
                 continue

             # 計算止損止盈價
             stop_loss_price_T = price_T - self.stop_loss_atr_multiplier * atr_T
             take_profit_price_T = price_T + self.take_profit_atr_multiplier * atr_T
             if stop_loss_price_T >= price_T:
                 print(f"  > {code}: 止損價 ({stop_loss_price_T:.2f}) >= 進場價 ({price_T:.2f})，跳過。")
                 continue

             # 檢查現金
             estimated_cost = shares_to_buy_api * price_T # 估算成本 (用 T 日價)
             if available_cash < estimated_cost:
                 print(f"  > {code}: 現金不足以購買 {planned_buy_units} 張 (需~{estimated_cost:.0f})。")
                 # 在此簡化策略下，現金不足直接放棄訂單
                 continue

             print(f"  > {code}: AI 選擇買入 {planned_buy_units} 張，計劃 SL={stop_loss_price_T:.2f}, TP={take_profit_price_T:.2f}")
             orders_to_submit_buy.append((code, shares_to_buy_api, planned_buy_units, price_T, stop_loss_price_T, take_profit_price_T))

        print("TradeExecutor: [BACKTEST] 模擬提交買單...")
        for code, shares_api, entry_units, price, sl, tp in orders_to_submit_buy:
            print(f"  [BACKTEST] 模擬提交買單: {entry_units} 張 {code} (SL:{sl:.2f}, TP:{tp:.2f})")
            simulated_success_map[code] = True
        return orders_to_submit_buy, simulated_success_map


class SimulationEngine:
    """回測引擎 (EMA5觸及進場, AI張數, ATR止盈止損, 正規化Obs)"""
    def __init__(self, start_date, end_date, data_manager: DataManager,
                 portfolio_manager: PortfolioManager, trade_executor: TradeExecutor,
                 models: dict, # {stock_code: model}
                 vec_normalize_stats_path: str): # <<< 需要正規化路徑
        self.start_date_str = start_date; self.end_date_str = end_date
        self.data_manager = data_manager; self.portfolio_manager = portfolio_manager
        self.trade_executor = trade_executor; self.models = models
        self.vec_normalize_stats_path = vec_normalize_stats_path
        self.target_stock_code = self.portfolio_manager.target_code
        if not self.target_stock_code or self.target_stock_code not in self.models: raise ValueError("無效股票代碼或模型")

        self.stock_codes = [self.target_stock_code]
        indicator_params = data_manager.get_indicator_periods()
        self.ema_period = indicator_params['ema'] # <<<
        self.ema_col_name = data_manager.get_ema_col_name() # <<<
        self.atr_period = indicator_params['atr']
        self.atr_col_name = data_manager.get_atr_col_name()

        self.features_per_stock = 2 # <<< 觀察空間維度
        self.portfolio_history = []; self.dates_history = []
        self.vec_env = None

    def _get_single_stock_observation(self, stock_code, date_idx):
        # 計算 2 維原始特徵: NormATR, NormDistToEMA5
        common_dates = self.data_manager.get_common_dates();
        if date_idx < 0 or date_idx >= len(common_dates): return np.zeros(self.features_per_stock, dtype=np.float32)
        current_date = common_dates[date_idx]
        obs_data = self.data_manager.get_data_on_date(stock_code, current_date)
        if obs_data is None: return np.zeros(self.features_per_stock, dtype=np.float32)
        try:
            close_price = obs_data['close']
            atr_val = obs_data.get(self.atr_col_name, np.nan)
            ema_val = obs_data.get(self.ema_col_name, np.nan) # <<< 獲取 EMA 值
            if pd.isna(close_price) or pd.isna(atr_val) or pd.isna(ema_val) or \
               close_price <= 0 or atr_val <= 0 or ema_val <= 0:
                return np.zeros(self.features_per_stock, dtype=np.float32)

            norm_atr = atr_val / close_price
            norm_dist_to_ema5 = (close_price - ema_val) / atr_val if atr_val > 0 else 0.0

            features = [norm_atr, norm_dist_to_ema5]
            observation = np.array(features, dtype=np.float32)
            observation = np.nan_to_num(observation, nan=0.0, posinf=10.0, neginf=-10.0)
            observation = np.clip(observation, -10.0, 10.0)

            if observation.shape[0] != self.features_per_stock: print(f"錯誤: Obs 維度錯誤"); return np.zeros(self.features_per_stock, dtype=np.float32)
            return observation
        except Exception as e: print(f"錯誤 ({stock_code}): Obs 未知錯誤 @ {current_date}: {e}"); traceback.print_exc(); return np.zeros(self.features_per_stock, dtype=np.float32)

    def _check_entry_signal(self, current_step_idx): # <<< EMA5 觸及檢查
        if current_step_idx < 0: return False
        current_date = self.data_manager.common_dates[current_step_idx]
        data_T = self.data_manager.get_data_on_date(self.target_stock_code, current_date)
        if data_T is None: return False
        low_T = data_T.get('low', np.inf); close_T = data_T.get('close', np.nan); ema5_T = data_T.get(self.ema_col_name, np.nan)
        if pd.isna(low_T) or pd.isna(close_T) or pd.isna(ema5_T): return False
        return low_T <= ema5_T and close_T >= ema5_T

    # 不需要 _check_exit_signal

    def run_backtest(self):
        """回測主循環 (EMA5進場, AI張數, ATR止盈止損, 正規化Obs)"""
        if not self.data_manager.load_all_data(self.start_date_str, self.end_date_str): return
        if self.target_stock_code not in self.data_manager.get_stock_codes(): return

        # 初始化 VecNormalize
        print(f"--- 加載 VecNormalize 統計數據: {self.vec_normalize_stats_path} ---")
        if not os.path.exists(self.vec_normalize_stats_path):
            print(f"警告：找不到 VecNormalize 統計文件。評估將使用未正規化的觀察值！")
            self.vec_env = None
        else:
            try:
                dummy_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.features_per_stock,), dtype=np.float32)
                dummy_act_space = spaces.Discrete(len(self.trade_executor.buy_units)) # <<< 從 TE 獲取動作空間大小
                dummy_env = gym.Env(); dummy_env.observation_space=dummy_obs_space; dummy_env.action_space=dummy_act_space
                dummy_vec_env_load = DummyVecEnv([lambda: dummy_env])
                self.vec_env = VecNormalize.load(self.vec_normalize_stats_path, dummy_vec_env_load)
                self.vec_env.training = False; self.vec_env.norm_reward = False
                print("--- VecNormalize 環境已準備 (評估模式) ---")
            except Exception as e: print(f"加載 VecNormalize 失敗: {e}. 使用未正規化觀察值。"); self.vec_env = None

        self.stock_codes = [self.target_stock_code]; self.portfolio_manager.stock_codes = self.stock_codes
        self.portfolio_manager.reset(); common_dates = self.data_manager.get_common_dates()
        window_size = self.data_manager.window_size; start_idx = window_size; end_idx = len(common_dates) - 1
        self.portfolio_history = [self.portfolio_manager.initial_capital] * start_idx
        self.dates_history = list(common_dates[:start_idx])

        print(f"\n--- SimulationEngine: 開始回測 ({self.target_stock_code}, {common_dates[start_idx].strftime('%Y-%m-%d')} to {common_dates[end_idx].strftime('%Y-%m-%d')}) ---")
        print(f"    策略: EMA{self.ema_period}觸及進場, AI決定張數, ATR(1:{self.trade_executor.take_profit_atr_multiplier}) TP/SL") # <<< 更新策略描述
        if self.vec_env: print("    觀察值將使用正規化。") 
        else: print("    警告：觀察值未正規化。")

        # --- 主回測循環 ---
        for current_idx in range(start_idx, end_idx):
            date_T = common_dates[current_idx]; date_T1 = common_dates[current_idx + 1]
            print(f"\n====== Day T: {date_T.strftime('%Y-%m-%d')} (收盤後決策) ======")

            code = self.target_stock_code; buy_decision = {}; sell_requests_plan = []
            current_shares = self.portfolio_manager.get_shares(code)

            # --- T 日決策 ---
            if current_shares == 0: # 未持倉，檢查進場
                entry_signal = self._check_entry_signal(current_idx)
                if entry_signal:
                    print(f"  > EMA{self.ema_period} 觸及信號，調用 AI 決定張數...")
                    raw_obs = self._get_single_stock_observation(code, current_idx)
                    if self.vec_env:
                        try: normalized_obs = self.vec_env.normalize_obs(np.array([raw_obs]))
                        except Exception as e: print(f"警告：正規化錯誤: {e}"); normalized_obs = np.array([raw_obs])
                    else: normalized_obs = np.array([raw_obs])

                    if normalized_obs.shape[1] == self.features_per_stock:
                        action, _ = self.models[code].predict(normalized_obs[0], deterministic=True)
                        buy_decision[code] = action # 記錄 AI 的動作索引
                        print(f"    > AI 選擇動作索引: {action} (對應 {self.trade_executor.buy_units[action]} 張)")
                    else: print(f"錯誤: 觀察值維度錯誤，無法預測。")
            # else: 持倉，本策略下 T 日不做出場決策，等待 T+1 檢查止盈止損

            # --- 執行階段 (只需處理潛在買單) ---
            print("--- 調用 TradeExecutor (模擬買單) ---")
            # <<< 修改：TradeExecutor 返回更詳細的買單信息 >>>
            planned_buy_orders, api_buy_success = self.trade_executor.determine_and_place_buy_orders(buy_decision, date_T)
            # sell_requests_plan 在此策略下 T 日總為空
            api_sell_success = self.trade_executor.place_sell_orders(sell_requests_plan)
            print("--- TradeExecutor 調用完成 (模擬) ---")

            # --- T+1 日 檢查止盈止損 & 結算 ---
            print(f"\n====== Day T+1: {date_T1.strftime('%Y-%m-%d')} (開盤檢查/成交/結算) ======")
            executed_trades_info = []
            trade_executed_sell = False

            # 優先檢查 T+1 止盈止損 (如果持倉)
            if self.portfolio_manager.get_shares(code) > 0:
                 stop_loss_target = self.portfolio_manager.get_stop_loss_price(code)
                 take_profit_target = self.portfolio_manager.get_take_profit_price(code)
                 entry_units = self.portfolio_manager.get_entry_units(code) # 獲取進場張數用於扣成本

                 data_T1 = self.data_manager.get_data_on_date(code, date_T1)
                 sell_price = None; sell_type = None
                 if data_T1 is not None:
                      price_T1_open = data_T1['open'] if pd.notna(data_T1['open']) else np.nan
                      low_T1 = data_T1['low'] if pd.notna(data_T1['low']) else np.nan
                      high_T1 = data_T1['high'] if pd.notna(data_T1['high']) else np.nan

                      if pd.notna(price_T1_open) and price_T1_open <= stop_loss_target: sell_price = price_T1_open; sell_type = 'stop_loss_gap'
                      elif pd.notna(low_T1) and low_T1 <= stop_loss_target: sell_price = stop_loss_target; sell_type = 'stop_loss'
                      elif pd.notna(price_T1_open) and price_T1_open >= take_profit_target: sell_price = price_T1_open; sell_type = 'take_profit_gap'
                      elif pd.notna(high_T1) and high_T1 >= take_profit_target: sell_price = take_profit_target; sell_type = 'take_profit'

                      if sell_type is not None:
                           shares_to_sell = self.portfolio_manager.get_shares(code) # 賣出全部
                           simulated_sell_price = sell_price # 默認
                           if sell_type == 'stop_loss_gap': simulated_sell_price = price_T1_open
                           elif sell_type == 'take_profit_gap': simulated_sell_price = price_T1_open
                           # 盤中觸及按目標價 (簡化)

                           if pd.notna(simulated_sell_price) and simulated_sell_price > 0:
                               proceeds = shares_to_sell * simulated_sell_price
                               self.portfolio_manager.update_on_sell(code, shares_to_sell, proceeds) # 賣出即清空狀態
                               sheets = shares_to_sell / self.trade_executor.shares_per_unit
                               executed_trades_info.append(f"{code}:SELL_{sell_type.upper()}_{sheets:.0f}張(Sim)")
                               trade_executed_sell = True
                           else:
                                print(f"警告: T+1 賣出 ({sell_type}) 失敗，成交價無效({simulated_sell_price})，強制清空記錄。")
                                self.portfolio_manager.update_on_sell(code, shares_to_sell, 0) # 也要更新狀態
                                trade_executed_sell = True # 標記事件發生
                 else:
                      print(f"警告: 無法獲取 T+1 數據，無法檢查止盈止損。")


            # 處理 T+1 的買入 (如果 T 日計劃且 T+1 未執行賣出)
            if not trade_executed_sell:
                 # <<< 解包買單信息 >>>
                 for code_req, shares_api, entry_units, price_T, stop_loss_req, take_profit_req in planned_buy_orders:
                      if code == code_req:
                           data_T1 = self.data_manager.get_data_on_date(code, date_T1)
                           if data_T1 is not None:
                                price_T1_open = data_T1['open'] if pd.notna(data_T1['open']) else 0
                                if price_T1_open > 0:
                                     cost = shares_api * price_T1_open
                                     if self.portfolio_manager.get_cash() >= cost:
                                          # <<< 傳遞張數、止損、止盈 >>>
                                          self.portfolio_manager.update_on_buy(code, shares_api, entry_units, cost, stop_loss_req, take_profit_req)
                                          executed_trades_info.append(f"{code}:BUY_{entry_units}張(SL:{stop_loss_req:.2f},TP:{take_profit_req:.2f})(Sim)")
                                     else: print(f"[BACKTEST] 警告: 買入現金不足")
                                else: print(f"[BACKTEST] 警告: 買入開盤價無效")

            # 更新組合價值 (使用 T+1 收盤價)
            current_value = self.portfolio_manager.calculate_and_update_portfolio_value(self.data_manager, date_T1)
            self.portfolio_history.append(current_value); self.dates_history.append(date_T1)
            print(f"本日結算後組合價值: {current_value:.2f}")
            if executed_trades_info: print(f"  本日成交: {', '.join(executed_trades_info)}")

        # 循環結束
        self.report_results(); self.plot_performance()
        if self.vec_env is not None: self.vec_env.close()

    def report_results(self): # (與之前版本一致)
        # ...
        final_portfolio_value=self.portfolio_manager.get_portfolio_value(); initial_capital=self.portfolio_manager.initial_capital
        total_return_pct=((final_portfolio_value-initial_capital)/initial_capital)*100 if initial_capital else 0
        print("\n--- SimulationEngine: 最終回測結果 ---"); print(f"股票: {self.target_stock_code}"); print(f"評估期間: {self.start_date_str} to {self.end_date_str}")
        print(f"初始資金: {initial_capital:.2f}"); print(f"最終組合價值: {final_portfolio_value:.2f}"); print(f"總回報率: {total_return_pct:.2f}%")


    def plot_performance(self): # (更新標題和檔名)
        try:
            plt.style.use('seaborn-v0_8-darkgrid'); plt.figure(figsize=(14, 7))
            if len(self.dates_history) == len(self.portfolio_history) and len(self.dates_history) > 0 :
                portfolio_series = pd.Series(self.portfolio_history, index=self.dates_history)
                title = f"{self.target_stock_code} Backtest ({self.start_date_str} to {self.end_date_str}) - EMA{self.ema_period} Touch, AI Units ({'/'.join(map(str, self.trade_executor.buy_units))}), ATR RR(1:{self.trade_executor.take_profit_atr_multiplier})" # <<< 更新標題
                filename = f"portfolio_curve_{self.target_stock_code}_backtest_ema_touch_ai_units.png" # <<< 更新檔名
                plt.plot(portfolio_series.index, portfolio_series.values, label='Portfolio Value', linewidth=1.5); plt.title(title)
                plt.xlabel("Date"); plt.ylabel("Portfolio Value (TWD)"); plt.legend(); plt.grid(True); plt.tight_layout(); import matplotlib.ticker as mtick
                ax = plt.gca(); ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f')); plt.xticks(rotation=45)
                plt.savefig(filename, dpi=300); print(f"投資組合價值曲線圖已保存為 {filename}")
            else: print(f"警告: 歷史數據長度不匹配或為空。")
        except ImportError: print("未安裝 matplotlib。")
        except Exception as e: print(f"繪製圖表時出錯: {e}")


# --- 主程序入口 ---
if __name__ == '__main__':
     API_ACCOUNT = "N26132089"; API_PASSWORD = "joshua900905"
     TARGET_STOCK_CODE_EVAL = '2330'
     EVAL_START_DATE = '20230101'; EVAL_END_DATE = '20240331'
     TOTAL_INITIAL_CAPITAL = 50000000.0

     # --- 指向對應的模型目錄 ---
     experiment_name = "ai_ema5_touch_atr_rr_units_1or2_v1" # <<< 與訓練時一致 (假設是 1 或 2 張的版本)
     MODELS_BASE_DIR = f"tuned_models/{TARGET_STOCK_CODE_EVAL}/{experiment_name}"
     MODEL_FILE_NAME = f"ppo_agent_{TARGET_STOCK_CODE_EVAL}_final.zip"
     MODEL_LOAD_PATH = os.path.join(MODELS_BASE_DIR, MODEL_FILE_NAME)
     VEC_NORMALIZE_STATS_PATH_EVAL = os.path.join(MODELS_BASE_DIR, "vecnormalize.pkl")

     # --- 參數 (必須與訓練時完全一致) ---
     EMA_PERIOD_EVAL = 5; ATR_PERIOD_EVAL = 14
     STOP_LOSS_ATR_MULT_EVAL = 1.0; TAKE_PROFIT_ATR_MULT_EVAL = 2.0
     BUY_UNITS_EVAL = [1, 2] # <<< 與訓練時一致
     WINDOW_SIZE_EVAL = max(20, ATR_PERIOD_EVAL + 1, EMA_PERIOD_EVAL + 1)
     SHARES_PER_UNIT_EVAL = 1000
     # ---

     RUN_TRAINING = False; RUN_EVALUATION = True

     if RUN_TRAINING: print("錯誤：此腳本僅用於評估。")
     if RUN_EVALUATION:
        print(f"\n=============== 開始回測 ({TARGET_STOCK_CODE_EVAL} - {experiment_name}) ===============")
        print(f"模型路徑: {MODEL_LOAD_PATH}"); print(f"VecNormalize 路徑: {VEC_NORMALIZE_STATS_PATH_EVAL}")

        if not os.path.exists(MODEL_LOAD_PATH): print(f"錯誤：找不到模型文件"); exit()
        if not os.path.exists(VEC_NORMALIZE_STATS_PATH_EVAL): print(f"警告：找不到 VecNormalize 統計文件")

        print("--- 初始化評估組件 ---")
        api_eval = Stock_API(API_ACCOUNT, API_PASSWORD)
        data_manager_eval = DataManager( # <<< 傳遞 EMA 週期
            stock_codes_initial=[TARGET_STOCK_CODE_EVAL], api=api_eval, window_size=WINDOW_SIZE_EVAL,
            ema_period=EMA_PERIOD_EVAL, atr_period=ATR_PERIOD_EVAL )

        if data_manager_eval.load_all_data(EVAL_START_DATE, EVAL_END_DATE):
             portfolio_manager_eval = PortfolioManager(
                 initial_capital=TOTAL_INITIAL_CAPITAL, stock_codes=data_manager_eval.get_stock_codes())
             trade_executor_eval = TradeExecutor( # <<< 傳遞買入單位和止盈止損參數
                 api=api_eval, portfolio_manager=portfolio_manager_eval, data_manager=data_manager_eval,
                 buy_units=BUY_UNITS_EVAL,
                 stop_loss_atr_multiplier=STOP_LOSS_ATR_MULT_EVAL,
                 take_profit_atr_multiplier=TAKE_PROFIT_ATR_MULT_EVAL,
                 shares_per_unit=SHARES_PER_UNIT_EVAL)

             models_eval = {}; print(f"--- 加載 PPO 模型 ---");
             try: models_eval[TARGET_STOCK_CODE_EVAL] = PPO.load(MODEL_LOAD_PATH, device='cpu')
             except Exception as e: print(f"加載 PPO 模型失敗: {e}"); exit()

             simulation_engine = SimulationEngine( # <<< 移除不相關參數
                 start_date=EVAL_START_DATE, end_date=EVAL_END_DATE, data_manager=data_manager_eval,
                 portfolio_manager=portfolio_manager_eval, trade_executor=trade_executor_eval, models=models_eval,
                 vec_normalize_stats_path=VEC_NORMALIZE_STATS_PATH_EVAL)
             simulation_engine.run_backtest()
        else: print("數據管理器初始化失敗。")
        print("\n=============== 回測完成 ===============")

     if not RUN_TRAINING and not RUN_EVALUATION: print("\n請設置 RUN_EVALUATION=True。")
     print("\n--- 程序執行完畢 ---")