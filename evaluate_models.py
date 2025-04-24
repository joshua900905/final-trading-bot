# -*- coding: utf-8 -*-
# evaluate_multi_stock_shared_capital_v1.py - Evaluate Multi-Stock Strategy (Shared Capital, AI Units 1/2)

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
import matplotlib.pyplot as plt
import traceback
from datetime import datetime
from typing import List, Dict, Tuple

# --- Stable Baselines 3 Import ---
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError:
    print("錯誤：找不到 stable_baselines3 或其組件。請運行 'pip install stable_baselines3[extra]'")
    exit()

# --- 台灣 0050 成分股列表 ---
TAIWAN_0050_STOCKS = [ # <<< 您的 0050 列表
    "2330", "2454", "2317", "2412", "6505", "2881", "2308", "2882", "1303",
    "1301", "2886", "3045", "2891", "2002", "1101", "2382", "5880", "2884",
    "1216", "2207", "2303", "3711", "2892", "1102", "2912", "2885", "2408",
    "2880", "6669", "2379", "1326", "2474", "3008", "2395", "5871", "2887",
    "4904", "2357", "4938", "1402", "2883", "9904", "8046", "2105", "1590",
    "2603", "2609", "2615", "2801", "6415"
]

# --- Stock API Class (不變) ---
class Stock_API:
    # ... (同上) ...
    """Stock API Class"""
    def __init__(self, account, password): self.account = account; self.password = password; self.base_url = 'http://140.116.86.242:8081/stock/api/v1'
    def Get_Stock_Informations(self, stock_code, start_date, stop_date):
        information_url = (f"{self.base_url}/api_get_stock_info_from_date_json/{stock_code}/{start_date}/{stop_date}");
        max_retries = 3; delay = 1
        for attempt in range(max_retries):
            try:
                response = requests.get(information_url, timeout=20); response.raise_for_status(); result = response.json();
                if result.get('result') == 'success': data = result.get('data', []); return data if isinstance(data, list) else []
                else: print(f"API Err ({stock_code}): {result.get('status', '未知')}"); return []
            except requests.exceptions.Timeout: time.sleep(delay)
            except requests.exceptions.RequestException as e: print(f"API Req Err ({stock_code}): {e}"); return []
        print(f"API Max Retry ({stock_code})"); return []

# --- MultiStockDataManager (不變) ---
class MultiStockDataManager:
    # ... (同上一個回答中的 MultiStockDataManager) ...
    def __init__(self, stock_codes: List[str], api: Stock_API, window_size: int,
                 ema_period: int = 5, atr_period: int = 14):
        self.stock_codes_initial = stock_codes; self.api = api; self.window_size = window_size
        self.ema_period = ema_period; self.ema_col_name = f'EMA_{ema_period}'
        self.atr_period = atr_period; self.atr_col_name = f'ATR_{atr_period}'
        self.data_dict: Dict[str, pd.DataFrame] = {}; self.successful_codes: List[str] = []
        self.common_dates: pd.DatetimeIndex | None = None

    def _load_and_preprocess_single_stock(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
        print(f"  Loading {stock_code} ({start_date} to {end_date})")
        raw_data = self.api.Get_Stock_Informations(stock_code, start_date, end_date);
        if not raw_data: print(f"    > No data for {stock_code}"); return None
        try:
            df = pd.DataFrame(raw_data); df['date'] = pd.to_datetime(df['date'], unit='s'); df = df.sort_values('date'); df = df[~df['date'].duplicated(keep='first')]; df = df.set_index('date')
            required_cols = ['open', 'high', 'low', 'close'];
            if not all(col in df.columns for col in required_cols): return None
            for col in required_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=required_cols)
            if df.empty: return None
            df.ta.ema(length=self.ema_period, close='close', append=True, col_names=(self.ema_col_name,))
            df.ta.atr(length=self.atr_period, high='high', low='low', close='close', append=True, col_names=(self.atr_col_name,))
            df = df.dropna()
            if df.empty: return None
            return df
        except Exception as e: print(f"    Err processing {stock_code}: {e}"); return None

    def load_all_data(self, start_date_str: str, end_date_str: str) -> bool:
        print(f"MultiStockDataManager: Loading {len(self.stock_codes_initial)} stocks...")
        start_dt_obj = pd.to_datetime(start_date_str, format='%Y%m%d')
        buffer_days = max(self.ema_period, self.atr_period) + 5
        required_start_dt = start_dt_obj - pd.Timedelta(days=buffer_days * 2.0); required_start_date_str = required_start_dt.strftime('%Y%m%d')
        all_dates = None; temp_data_dict = {}

        for code in self.stock_codes_initial:
            df = self._load_and_preprocess_single_stock(code, required_start_date_str, end_date_str)
            if df is not None and not df.empty:
                df_filtered = df[df.index >= start_dt_obj]
                if len(df_filtered) >= self.window_size:
                    temp_data_dict[code] = df_filtered; self.successful_codes.append(code)
                    print(f"    > Loaded {code} ({len(df_filtered)} rows)")
                    if all_dates is None: all_dates = df_filtered.index
                    else: all_dates = all_dates.intersection(df_filtered.index)
                else: print(f"    > Insufficient data for {code}")
            time.sleep(0.1) # Shorter delay

        if not self.successful_codes: print("Error: No stock data loaded successfully."); return False
        if all_dates is None or len(all_dates) < self.window_size + 1: print("Error: Not enough common trading days found."); return False

        self.common_dates = all_dates.sort_values()
        final_data_dict = {}; final_successful_codes = []
        for code in self.successful_codes:
            if code in temp_data_dict:
                 df_common = temp_data_dict[code].loc[self.common_dates]
                 if not df_common.isnull().values.any():
                     final_data_dict[code] = df_common; final_successful_codes.append(code)
                 else: print(f"Warning: NaN values found in {code} after aligning. Skipping.")

        self.data_dict = final_data_dict; self.successful_codes = final_successful_codes
        if not self.successful_codes: print("Error: No stocks remaining after NaN check."); return False
        print(f"MultiStockDataManager: Loaded {len(self.successful_codes)} stocks with {len(self.common_dates)} common days.")
        return True

    # Getters (不變)
    def get_common_dates(self): return self.common_dates
    def get_stock_codes(self): return self.successful_codes
    def get_data_on_date(self, stock_code: str, date: pd.Timestamp) -> pd.Series | None: #...
        if stock_code in self.data_dict and date in self.data_dict[stock_code].index:
            # Use .at for faster access if index is unique and sorted
            try: return self.data_dict[stock_code].loc[date]
            except KeyError: return None # Handle date not found cleanly
        return None
    def get_indicator_periods(self): return {'ema': self.ema_period, 'atr': self.atr_period}
    def get_atr_col_name(self): return self.atr_col_name
    def get_ema_col_name(self): return self.ema_col_name

# --- Portfolio Manager (共享資金) ---
class MultiStockPortfolioManager:
    def __init__(self, initial_capital: float, stock_codes: List[str]): # <<< 移除 initial_capital_per_stock
        self.initial_capital = initial_capital
        self.stock_codes = stock_codes
        self.cash = initial_capital # <<< 使用總資金
        self.shares_held: Dict[str, int] = defaultdict(int)
        self.entry_price: Dict[str, float] = defaultdict(float)
        self.stop_loss_price: Dict[str, float] = defaultdict(float)
        self.take_profit_price: Dict[str, float] = defaultdict(float)
        self.entry_units: Dict[str, int] = defaultdict(int)
        self.portfolio_value = initial_capital

    def reset(self):
        self.cash = self.initial_capital # <<< 重置為總資金
        self.shares_held = defaultdict(int); self.entry_price = defaultdict(float)
        self.stop_loss_price = defaultdict(float); self.take_profit_price = defaultdict(float)
        self.entry_units = defaultdict(int); self.portfolio_value = self.initial_capital
        print(f"MultiStockPortfolioManager: 狀態已重設 (共享資金)。")

    def update_on_buy(self, stock_code: str, shares_bought: int, entry_units: int, cost: float, stop_loss: float, take_profit: float) -> bool: # <<< 返回是否成功
        if cost > self.cash:
             print(f"  > 買入 {stock_code} ({entry_units}張) 失敗 - 現金不足 ({self.cash:.0f} < {cost:.0f})")
             return False # <<< 現金不足，買入失敗
        self.cash -= cost
        self.entry_price[stock_code] = cost / shares_bought if shares_bought > 0 else 0
        # 允許多重持倉，直接增加股數
        self.shares_held[stock_code] += shares_bought
        # 如何處理 SL/TP/Units 當加倉時？ 這裡假設新買入覆蓋舊狀態 (簡化)
        self.stop_loss_price[stock_code] = stop_loss
        self.take_profit_price[stock_code] = take_profit
        self.entry_units[stock_code] = entry_units # 或者累加？ entry_units += units? 這裡假設覆蓋
        print(f"    > PM: 買入 {entry_units} 張 {stock_code} 成功。現金剩餘: {self.cash:.0f}")
        return True

    def update_on_sell(self, stock_code: str, shares_sold: int, proceeds: float):
        if stock_code not in self.shares_held or shares_sold <= 0: return
        actual_sold = min(shares_sold, self.shares_held[stock_code])
        self.cash += proceeds # 增加現金
        print(f"    > PM: 賣出 {actual_sold/1000:.0f} 張 {stock_code}。現金增加至: {self.cash:.0f}")
        # 清空該股票狀態
        del self.shares_held[stock_code]; del self.entry_price[stock_code]
        del self.stop_loss_price[stock_code]; del self.take_profit_price[stock_code]
        del self.entry_units[stock_code]


    def calculate_and_update_portfolio_value(self, data_manager: MultiStockDataManager, current_date: pd.Timestamp):
        # (與之前多股票版本一致)
        total_stock_value = 0.0
        for code in list(self.shares_held.keys()): # Iterate over a copy of keys
            shares = self.shares_held[code]
            if shares <= 0: continue
            data = data_manager.get_data_on_date(code, current_date)
            close_price = np.nan
            if data is not None and pd.notna(data['close']) and data['close'] > 0:
                close_price = data['close']
            else:
                 prev_date_loc = data_manager.common_dates.get_loc(current_date) - 1
                 if prev_date_loc >= 0:
                      prev_date = data_manager.common_dates[prev_date_loc]
                      prev_data = data_manager.get_data_on_date(code, prev_date)
                      if prev_data is not None and pd.notna(prev_data['close']) and prev_data['close'] > 0:
                           close_price = prev_data['close']
            if pd.notna(close_price): total_stock_value += shares * close_price
            else: print(f"警告: 無法獲取 {code} 有效價格 @ {current_date}")
        self.portfolio_value = self.cash + total_stock_value
        return self.portfolio_value

    # Getters (不變)
    def get_cash(self): return self.cash
    def get_shares(self, stock_code: str) -> int: return self.shares_held.get(stock_code, 0)
    def get_portfolio_value(self): return self.portfolio_value
    def get_entry_price(self, stock_code: str) -> float: return self.entry_price.get(stock_code, 0.0)
    def get_stop_loss_price(self, stock_code: str) -> float: return self.stop_loss_price.get(stock_code, 0.0)
    def get_take_profit_price(self, stock_code: str) -> float: return self.take_profit_price.get(stock_code, 0.0)
    def get_entry_units(self, stock_code: str) -> int: return self.entry_units.get(stock_code, 0)
    def get_holding_stocks(self) -> List[str]: return [code for code, shares in self.shares_held.items() if shares > 0]


# --- Trade Executor (共享資金) ---
class MultiStockTradeExecutor:
    def __init__(self, api: Stock_API, portfolio_manager: MultiStockPortfolioManager, data_manager: MultiStockDataManager,
                 buy_units: List[int], stop_loss_atr_multiplier: float, take_profit_atr_multiplier: float,
                 shares_per_unit: int, max_concurrent_trades: int = 5): # <<< 移除 capital_per_trade
        self.api = api; self.portfolio_manager = portfolio_manager; self.data_manager = data_manager
        self.buy_units = buy_units; self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.take_profit_atr_multiplier = take_profit_atr_multiplier; self.shares_per_unit = shares_per_unit
        self.max_concurrent_trades = max_concurrent_trades

    def place_sell_orders(self, sell_requests: List[Tuple[str, int, float, str]]):
        # (不變)
        print("TradeExecutor: [BACKTEST] 模擬提交賣單...")
        for code, shares_api, price, sell_type in sell_requests:
            sheets = shares_api / self.shares_per_unit
            print(f"  [BACKTEST] 計劃賣出 ({sell_type}): {sheets:.0f} 張 {code}")

    def determine_and_place_buy_orders(self, buy_decisions: Dict[str, int], date_T: pd.Timestamp):
        # (修改現金檢查邏輯)
        orders_to_submit_buy = []
        if not buy_decisions: return orders_to_submit_buy

        # <<< 使用共享的總可用現金 >>>
        available_cash = self.portfolio_manager.get_cash()
        current_holdings = self.portfolio_manager.get_holding_stocks()
        num_can_open = self.max_concurrent_trades - len(current_holdings) # 還能開多少新倉位

        print(f"TradeExecutor: [BACKTEST] 處理 {len(buy_decisions)} 個買入信號... (總現金: {available_cash:.0f}, 可開新倉: {num_can_open})")

        potential_orders = [] # 暫存潛在訂單 (code, shares, units, cost, sl, tp)

        # 1. 生成所有潛在訂單及其成本
        for code, action_idx in buy_decisions.items():
             if code in current_holdings: continue # 不加倉已持有股票
             if action_idx < 0 or action_idx >= len(self.buy_units): continue

             planned_buy_units = self.buy_units[action_idx]
             shares_to_buy_api = planned_buy_units * self.shares_per_unit
             data_T = self.data_manager.get_data_on_date(code, date_T)
             if data_T is None: continue
             price_T = data_T['close']; atr_T = data_T.get(self.data_manager.get_atr_col_name(), 0.0)
             if pd.isna(price_T) or price_T <= 0 or pd.isna(atr_T) or atr_T <= 0: continue

             stop_loss_price_T = price_T - self.stop_loss_atr_multiplier * atr_T
             take_profit_price_T = price_T + self.take_profit_atr_multiplier * atr_T
             if stop_loss_price_T >= price_T: continue

             estimated_cost = shares_to_buy_api * price_T
             potential_orders.append((code, shares_to_buy_api, planned_buy_units, estimated_cost, stop_loss_price_T, take_profit_price_T))

        # 2. 根據可用現金和持倉限制選擇訂單 (按遍歷順序，先到先得)
        print(f"  > 潛在買入訂單數量: {len(potential_orders)}")
        count_opened = 0
        for code, shares_api, entry_units, cost, sl, tp in potential_orders:
             if count_opened >= num_can_open:
                  print(f"  > 已達最大可開新倉數量 ({num_can_open})，停止處理買單。")
                  break
             if available_cash >= cost:
                  print(f"  > 計劃買入 {code}: {entry_units} 張 (成本~{cost:.0f})")
                  orders_to_submit_buy.append((code, shares_api, entry_units, price_T, sl, tp)) # price_T 用於記錄
                  available_cash -= cost # 預扣現金
                  count_opened += 1
             else:
                  print(f"  > {code}: 總現金不足以購買 {entry_units} 張 (需~{cost:.0f}，剩餘 {available_cash:.0f})。")

        # 模擬提交買單日誌 (不變)
        print("TradeExecutor: [BACKTEST] 模擬提交最終買單...")
        for code, shares_api, entry_units, price, sl, tp in orders_to_submit_buy:
             print(f"  [BACKTEST] 模擬提交買單: {entry_units} 張 {code} (SL:{sl:.2f}, TP:{tp:.2f})")

        return orders_to_submit_buy


# --- Simulation Engine (Multi-Stock, Shared Capital) ---
class MultiStockSimulationEngine:
     # (初始化和方法與上個版本基本一致，只需確保傳遞正確的 Manager 和 Executor)
    def __init__(self, start_date: str, end_date: str, data_manager: MultiStockDataManager,
                 portfolio_manager: MultiStockPortfolioManager, # <<< 使用共享資金 PM
                 trade_executor: MultiStockTradeExecutor,     # <<< 使用共享資金 TE
                 model: PPO, vec_normalize_stats_path: str):
        self.start_date_str = start_date; self.end_date_str = end_date
        self.data_manager = data_manager; self.portfolio_manager = portfolio_manager
        self.trade_executor = trade_executor; self.model = model
        self.vec_normalize_stats_path = vec_normalize_stats_path
        self.stock_codes = self.data_manager.get_available_stocks()
        # ... (獲取指標參數) ...
        indicator_params = data_manager.get_indicator_periods()
        self.ema_period = indicator_params['ema']; self.ema_col_name = data_manager.get_ema_col_name()
        self.atr_period = indicator_params['atr']; self.atr_col_name = data_manager.get_atr_col_name()
        self.features_per_stock = 2 # 觀察空間維度
        self.portfolio_history = []; self.dates_history = []
        self.vec_env = None

    def _get_observation(self, stock_code: str, date: pd.Timestamp) -> np.ndarray | None:
        # (與上版本一致)
        # ...
        data = self.data_manager.get_data_on_date(stock_code, date)
        if data is None: return None
        try:
            close_price = data['close']; atr_val = data.get(self.atr_col_name, np.nan); ema_val = data.get(self.ema_col_name, np.nan)
            if pd.isna(close_price) or pd.isna(atr_val) or pd.isna(ema_val) or close_price <= 0 or atr_val <= 0 or ema_val <= 0: return None
            norm_atr = atr_val / close_price
            norm_dist_to_ema5 = (close_price - ema_val) / atr_val if atr_val > 0 else 0.0
            features = [norm_atr, norm_dist_to_ema5]
            observation = np.array(features, dtype=np.float32)
            observation = np.nan_to_num(observation, nan=0.0, posinf=10.0, neginf=-10.0); observation = np.clip(observation, -10.0, 10.0)
            if observation.shape[0] != self.features_per_stock: return None
            return observation
        except Exception as e: print(f"Error getting obs for {stock_code} @ {date}: {e}"); return None


    def _check_entry_signal(self, stock_code: str, date: pd.Timestamp) -> bool:
        # (與上版本一致)
        # ...
        data_T = self.data_manager.get_data_on_date(stock_code, date)
        if data_T is None: return False
        low_T = data_T.get('low', np.inf); close_T = data_T.get('close', np.nan); ema5_T = data_T.get(self.ema_col_name, np.nan)
        if pd.isna(low_T) or pd.isna(close_T) or pd.isna(ema5_T): return False
        return low_T <= ema5_T and close_T >= ema5_T


    def run_backtest(self):
        # (回測主循環邏輯與上版本一致，但 PM 和 TE 已更新為共享資金邏輯)
        if not self.data_manager.common_dates is not None: print("錯誤：數據未加載"); return
        # ... (初始化 VecNormalize) ...
        print(f"--- 加載 VecNormalize 統計數據: {self.vec_normalize_stats_path} ---")
        if not os.path.exists(self.vec_normalize_stats_path): print(f"警告：找不到 VecNormalize 文件") ; self.vec_env = None
        else:
             try: # ... (加載 VecNormalize) ...
                 dummy_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.features_per_stock,), dtype=np.float32)
                 dummy_act_space = spaces.Discrete(len(self.trade_executor.buy_units))
                 dummy_env = gym.Env(); dummy_env.observation_space=dummy_obs_space; dummy_env.action_space=dummy_act_space
                 dummy_vec_env_load = DummyVecEnv([lambda: dummy_env])
                 self.vec_env = VecNormalize.load(self.vec_normalize_stats_path, dummy_vec_env_load)
                 self.vec_env.training = False; self.vec_env.norm_reward = False
                 print("--- VecNormalize 環境已準備 (評估模式) ---")
             except Exception as e: print(f"加載 VecNormalize 失敗: {e}."); self.vec_env = None

        self.portfolio_manager.reset(); common_dates = self.data_manager.common_dates
        start_idx = 0; end_idx = len(common_dates) - 1
        self.portfolio_history = []; self.dates_history = [] # 初始化為空

        print(f"\n--- SimulationEngine: 開始多股票回測 ({len(self.stock_codes)} stocks, 共享資金) ---")
        # ... (打印策略信息) ...
        print(f"    策略: EMA觸及進場, AI決定張數({self.trade_executor.buy_units}), ATR(1:{self.trade_executor.take_profit_atr_multiplier}) TP/SL")
        if self.vec_env: print("    觀察值將使用正規化。") 
        else: print("    警告：觀察值未正規化。")


        # --- 主回測循環 ---
        for current_idx in range(start_idx, end_idx):
            date_T = common_dates[current_idx]; date_T1 = common_dates[current_idx + 1]
            print(f"\n====== Day T: {date_T.strftime('%Y-%m-%d')} (收盤後決策) ======")

            buy_decisions_T: Dict[str, int] = {}
            sell_requests_T1_final: List[Tuple[str, int, float, str]] = []

            # --- T日決策循環 ---
            print("--- T日決策 ---")
            currently_holding = self.portfolio_manager.get_holding_stocks()
            for code in self.stock_codes:
                if code not in currently_holding: # 檢查買入
                    entry_signal = self._check_entry_signal(code, date_T)
                    if entry_signal:
                         print(f"  > {code}: EMA觸及信號，獲取 AI 動作...")
                         raw_obs = self._get_observation(code, date_T)
                         if raw_obs is not None:
                              normalized_obs = raw_obs # 默認
                              if self.vec_env:
                                   try: normalized_obs = self.vec_env.normalize_obs(np.array([raw_obs]))[0]
                                   except Exception as e: print(f"    > 正規化錯誤: {e}")
                              action, _ = self.model.predict(normalized_obs, deterministic=True)
                              buy_decisions_T[code] = action
                              print(f"    > AI 選擇買入 {self.trade_executor.buy_units[action]} 張")
                         else: print(f"    > 無法獲取觀察值")

            # --- T日結束，執行交易計劃 ---
            print("--- 調用 TradeExecutor ---")
            planned_buy_orders = self.trade_executor.determine_and_place_buy_orders(buy_decisions_T, date_T)
            # T日沒有賣單計劃
            self.trade_executor.place_sell_orders([])
            print("--- TradeExecutor 調用完成 ---")


            # --- T+1 日 檢查止盈止損 & 結算 ---
            print(f"\n====== Day T+1: {date_T1.strftime('%Y-%m-%d')} (開盤檢查/成交/結算) ======")
            executed_trades_info = []

            # 1. 檢查所有持倉股票的止盈止損
            print("--- T+1 檢查止盈止損 ---")
            holding_stocks_t1 = self.portfolio_manager.get_holding_stocks()
            for code in holding_stocks_t1:
                 # ... (檢查止盈止損邏輯，與上版本相同) ...
                 stop_loss_target = self.portfolio_manager.get_stop_loss_price(code)
                 take_profit_target = self.portfolio_manager.get_take_profit_price(code)
                 data_T1 = self.data_manager.get_data_on_date(code, date_T1)
                 sell_price = None; sell_type = None
                 if data_T1 is not None:
                      price_T1_open = data_T1.get('open', np.nan); low_T1 = data_T1.get('low', np.nan); high_T1 = data_T1.get('high', np.nan)
                      if pd.notna(price_T1_open) and pd.notna(low_T1) and pd.notna(high_T1):
                           if price_T1_open <= stop_loss_target: sell_price = price_T1_open; sell_type = 'stop_loss_gap'
                           elif low_T1 <= stop_loss_target: sell_price = stop_loss_target; sell_type = 'stop_loss'
                           elif price_T1_open >= take_profit_target: sell_price = price_T1_open; sell_type = 'take_profit_gap'
                           elif high_T1 >= take_profit_target: sell_price = take_profit_target; sell_type = 'take_profit'
                      else: print(f"  > {code}: T+1價格無效")
                 else: print(f"  > {code}: 無法獲取T+1數據")

                 if sell_type is not None:
                      shares_to_sell = self.portfolio_manager.get_shares(code)
                      simulated_sell_price = sell_price
                      if sell_type == 'stop_loss_gap': simulated_sell_price = price_T1_open
                      elif sell_type == 'take_profit_gap': simulated_sell_price = price_T1_open
                      if not (pd.notna(simulated_sell_price) and simulated_sell_price > 0):
                           simulated_sell_price = data_T1.get('close', 0) if data_T1 is not None else 0 # 嘗試收盤價
                      sell_requests_T1_final.append((code, shares_to_sell, simulated_sell_price, sell_type))
                      print(f"  > {code}: 觸發 {sell_type}，計劃以 {simulated_sell_price:.2f} 賣出")


            # 2. 執行 T+1 的賣單
            for code_sell, shares_sell, price_sell, type_sell in sell_requests_T1_final:
                 if price_sell > 0:
                     proceeds = shares_sell * price_sell
                     self.portfolio_manager.update_on_sell(code_sell, shares_sell, proceeds)
                     sheets = shares_sell / self.trade_executor.shares_per_unit
                     executed_trades_info.append(f"{code_sell}:SELL_{type_sell.upper()}_{sheets:.0f}@{price_sell:.2f}")
                 else:
                     self.portfolio_manager.update_on_sell(code_sell, shares_sell, 0)
                     sheets = shares_sell / self.trade_executor.shares_per_unit
                     executed_trades_info.append(f"{code_sell}:SELL_{type_sell.upper()}_{sheets:.0f}(Failed)")


            # 3. 執行 T+1 的買單 (來自 T 日的計劃)
            for code_buy, shares_buy, entry_units, price_T, stop_loss_req, take_profit_req in planned_buy_orders:
                 data_T1_buy = self.data_manager.get_data_on_date(code_buy, date_T1)
                 if data_T1_buy is not None:
                      price_T1_open_buy = data_T1_buy['open'] if pd.notna(data_T1_buy['open']) else 0
                      if price_T1_open_buy > 0:
                           cost = shares_buy * price_T1_open_buy
                           # PM 內部會檢查現金
                           buy_success = self.portfolio_manager.update_on_buy(code_buy, shares_buy, entry_units, cost, stop_loss_req, take_profit_req)
                           if buy_success: executed_trades_info.append(f"{code_buy}:BUY_{entry_units}張@{price_T1_open_buy:.2f}")
                           else: executed_trades_info.append(f"{code_buy}:BUY_{entry_units}張(Failed-Cash)")
                      else: executed_trades_info.append(f"{code_buy}:BUY_{entry_units}張(Failed-Price)")
                 else: executed_trades_info.append(f"{code_buy}:BUY_{entry_units}張(Failed-Data)")


            # --- T+1 收盤，更新組合價值 ---
            current_value = self.portfolio_manager.calculate_and_update_portfolio_value(self.data_manager, date_T1)
            # <<< 記錄歷史數據 >>>
            self.portfolio_history.append(current_value)
            self.dates_history.append(date_T1) # <<< 確保記錄日期
            holdings_summary = ",".join(f"{s}({int(self.portfolio_manager.get_shares(s)/1000)})" for s in self.portfolio_manager.get_holding_stocks())
            print(f"本日結算後價值: {current_value:.0f} | 現金: {self.portfolio_manager.get_cash():.0f} | 持倉: [{holdings_summary}]")
            if executed_trades_info: print(f"  本日成交: {', '.join(executed_trades_info)}")

        # 循環結束
        self.report_results(); self.plot_performance()
        if self.vec_env is not None: self.vec_env.close()

    # report_results, plot_performance (與上版本一致，標題可更新)
    def report_results(self): # ... (同上)
        final_portfolio_value=self.portfolio_manager.get_portfolio_value(); initial_capital=self.portfolio_manager.initial_capital
        total_return_pct=((final_portfolio_value-initial_capital)/initial_capital)*100 if initial_capital > 0 else 0
        num_stocks = len(self.stock_codes)
        print("\n--- SimulationEngine: 最終回測結果 ---"); print(f"股票池: 0050 成分股 ({num_stocks} 支有效)"); print(f"評估期間: {self.start_date_str} to {self.end_date_str}")
        print(f"初始資金: {initial_capital:.0f}"); print(f"最終組合價值: {final_portfolio_value:.0f}"); print(f"總回報率: {total_return_pct:.2f}%")

    def plot_performance(self): # ... (同上，可改標題檔名)
        try:
            plt.style.use('seaborn-v0_8-darkgrid'); plt.figure(figsize=(14, 7))
            if len(self.dates_history) == len(self.portfolio_history) and len(self.dates_history) > 0 :
                portfolio_series = pd.Series(self.portfolio_history, index=self.dates_history)
                title = f"Multi-Stock Backtest ({self.start_date_str} to {self.end_date_str}) - Shared Capital, EMA Touch, AI Units ({'/'.join(map(str, self.trade_executor.buy_units))}), ATR RR"
                filename = f"portfolio_curve_multi_stock_shared_capital.png"
                plt.plot(portfolio_series.index, portfolio_series.values, label='Portfolio Value', linewidth=1.5); plt.title(title)
                plt.xlabel("Date"); plt.ylabel("Portfolio Value (TWD)"); plt.legend(); plt.grid(True); plt.tight_layout(); import matplotlib.ticker as mtick
                ax = plt.gca(); ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f')); plt.xticks(rotation=45)
                plt.savefig(filename, dpi=300); print(f"曲線圖已保存為 {filename}")
            else: print(f"警告: 歷史數據長度不匹配或為空。")
        except ImportError: print("未安裝 matplotlib。")
        except Exception as e: print(f"繪製圖表時出錯: {e}")


# --- 主程序入口 ---
if __name__ == '__main__':
     API_ACCOUNT = "N26132089"; API_PASSWORD = "joshua900905"
     EVAL_START_DATE = '20180101'; EVAL_END_DATE = '20231231' # 使用更長的回測區間
     TOTAL_INITIAL_CAPITAL = 50000000.0 # <<< 總資金

     # --- 指向單股票訓練的模型 ---
     TRAINED_STOCK_CODE = '2330'
     experiment_name = "ai_ema5_touch_atr_rr_units_1or2_v1" # <<< 使用對應的模型
     MODELS_BASE_DIR = f"tuned_models/{TRAINED_STOCK_CODE}/{experiment_name}"
     MODEL_LOAD_PATH = os.path.join(MODELS_BASE_DIR, f"ppo_agent_{TRAINED_STOCK_CODE}_final.zip")
     VEC_NORMALIZE_STATS_PATH_EVAL = os.path.join(MODELS_BASE_DIR, "vecnormalize.pkl")

     # --- 策略參數 (與訓練時一致) ---
     EMA_PERIOD_EVAL = 5; ATR_PERIOD_EVAL = 14
     STOP_LOSS_ATR_MULT_EVAL = 1.0; TAKE_PROFIT_ATR_MULT_EVAL = 2.0
     BUY_UNITS_EVAL = [1, 2] # <<< AI 動作空間
     WINDOW_SIZE_EVAL = max(20, ATR_PERIOD_EVAL + 1, EMA_PERIOD_EVAL + 1)
     SHARES_PER_UNIT_EVAL = 1000
     # --- 多股票特定參數 ---
     MAX_CONCURRENT_TRADES_EVAL = 10 # <<< 最多同時持有幾檔股票 (可調整)

     RUN_EVALUATION = True

     if RUN_EVALUATION:
        print(f"\n=============== 開始多股票回測 (共享資金, 單一模型) ===============")
        # ... (打印路徑) ...
        print(f"模型路徑: {MODEL_LOAD_PATH}"); print(f"VecNormalize 路徑: {VEC_NORMALIZE_STATS_PATH_EVAL}")

        if not os.path.exists(MODEL_LOAD_PATH): print(f"錯誤：找不到模型文件"); exit()
        if not os.path.exists(VEC_NORMALIZE_STATS_PATH_EVAL): print(f"警告：找不到 VecNormalize 文件")

        print("--- 初始化評估組件 ---")
        api_eval = Stock_API(API_ACCOUNT, API_PASSWORD)
        data_manager_eval = MultiStockDataManager( # <<< 使用 Multi
            stock_codes=TAIWAN_0050_STOCKS, api=api_eval, window_size=WINDOW_SIZE_EVAL,
            ema_period=EMA_PERIOD_EVAL, atr_period=ATR_PERIOD_EVAL )

        if data_manager_eval.load_all_data(EVAL_START_DATE, EVAL_END_DATE):
             available_stocks_eval = data_manager_eval.get_available_stocks()
             if not available_stocks_eval: print("錯誤: 無有效股票數據"); exit()

             portfolio_manager_eval = MultiStockPortfolioManager( # <<< 使用 Multi
                 initial_capital=TOTAL_INITIAL_CAPITAL, stock_codes=available_stocks_eval)

             trade_executor_eval = MultiStockTradeExecutor( # <<< 使用 Multi
                 api=api_eval, portfolio_manager=portfolio_manager_eval, data_manager=data_manager_eval,
                 buy_units=BUY_UNITS_EVAL,
                 stop_loss_atr_multiplier=STOP_LOSS_ATR_MULT_EVAL,
                 take_profit_atr_multiplier=TAKE_PROFIT_ATR_MULT_EVAL,
                 shares_per_unit=SHARES_PER_UNIT_EVAL,
                 max_concurrent_trades=MAX_CONCURRENT_TRADES_EVAL) # <<< 傳遞並發限制

             model_eval = None; print(f"--- 加載 PPO 模型 ---");
             try: model_eval = PPO.load(MODEL_LOAD_PATH, device='cpu')
             except Exception as e: print(f"加載 PPO 模型失敗: {e}"); exit()

             simulation_engine = MultiStockSimulationEngine( # <<< 使用 Multi
                 start_date=EVAL_START_DATE, end_date=EVAL_END_DATE, data_manager=data_manager_eval,
                 portfolio_manager=portfolio_manager_eval, trade_executor=trade_executor_eval,
                 model=model_eval,
                 vec_normalize_stats_path=VEC_NORMALIZE_STATS_PATH_EVAL)

             simulation_engine.run_backtest()
        else: print("數據管理器初始化失敗。")
        print("\n=============== 多股票回測完成 ===============")

     else: print("\n請設置 RUN_EVALUATION=True。")
     print("\n--- 程序執行完畢 ---")