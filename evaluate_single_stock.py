# -*- coding: utf-8 -*-
# evaluate_single_stock.py - Evaluate model trained with TEMA features and Indicator Exit

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

# --- Stable Baselines 3 Import ---
try:
    from stable_baselines3 import PPO
except ImportError:
    print("錯誤：找不到 stable_baselines3。請運行 'pip install stable_baselines3'")
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

# --- Refactored Evaluation Classes ---

class DataManager:
    """負責加載、預處理和提供市場數據 (只計算 TEMA)。"""
    # --- 修改: __init__ 只接收 TEMA 參數 ---
    def __init__(self, stock_codes_initial, api, window_size,
                 tema_short=9, tema_long=18):
        self.stock_codes_initial = list(stock_codes_initial); self.api = api;
        self.tema_short_period = tema_short; self.tema_long_period = tema_long
        # --- 修改: window_size 基於 TEMA ---
        self.window_size = self.tema_long_period * 3 + 10 # 使用與 Env 相同的保守估計
        # ---
        self.data_dict = {}; self.common_dates = None; self.stock_codes = []

    # --- 修改: _load_and_preprocess_single_stock 只計算 TEMA 和變化率 ---
    def _load_and_preprocess_single_stock(self, stock_code, start_date, end_date):
        # print(f"  DataManager: 載入數據 {stock_code} ({start_date} to {end_date})")
        raw_data = self.api.Get_Stock_Informations(stock_code, start_date, end_date);
        if not raw_data: return None
        try:
            df = pd.DataFrame(raw_data); df['date'] = pd.to_datetime(df['date'], unit='s'); df = df.sort_values('date'); df = df[~df['date'].duplicated(keep='first')]; df = df.set_index('date')
            required_cols = ['open', 'high', 'low', 'close', 'turnover']; # 仍然需要 OHLC 用於退出模擬
            if not all(col in df.columns for col in required_cols): return None
            numeric_cols = ['open', 'high', 'low', 'close', 'turnover', 'capacity', 'transaction_volume']
            for col in numeric_cols:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
            # Volume calculation optional
            # if 'turnover' in df.columns and 'close' in df.columns: df['volume'] = np.where(df['close'].isna() | (df['close'] == 0), 0, df['turnover'] / df['close']); df['volume'] = df['volume'].fillna(0).replace([np.inf, -np.inf], 0).round().astype(np.int64)
            # else: df['volume'] = 0
            indicator_base_cols = ['open', 'high', 'low', 'close']; df = df.dropna(subset=indicator_base_cols) # Keep OHLC
            if df.empty: return None
            # --- 計算 TEMA ---
            tema9_col = f'TEMA_{self.tema_short_period}'
            tema18_col = f'TEMA_{self.tema_long_period}'
            df.ta.tema(length=self.tema_short_period, close='close', append=True, col_names=(tema9_col,))
            df.ta.tema(length=self.tema_long_period, close='close', append=True, col_names=(tema18_col,))
            # --- 計算 TEMA 變化率 ---
            df[f'{tema9_col}_slope'] = df[tema9_col].pct_change().fillna(0).replace([np.inf, -np.inf], 0)
            df[f'{tema18_col}_slope'] = df[tema18_col].pct_change().fillna(0).replace([np.inf, -np.inf], 0)
            # ---
            df = df.dropna(); # 移除 TEMA 計算產生的 NaN
            if df.empty: return None
            return df
        except Exception as e: print(f"DataManager: 處理 {stock_code} 數據時出錯: {e}"); traceback.print_exc(); return None

    def load_all_data(self, start_date, end_date):
        print(f"DataManager: 正在載入數據 ({start_date} to {end_date}) for {len(self.stock_codes_initial)} stocks...")
        temp_data_dict, temp_common_dates, successful_codes = {}, None, []
        try: start_dt = pd.to_datetime(start_date, format='%Y%m%d'); buffer_days = 30; required_start_dt = start_dt - pd.Timedelta(days=(self.window_size + buffer_days) * 1.5); required_start_date_str = required_start_dt.strftime('%Y%m%d')
        except ValueError: print("錯誤：起始日期格式無效。"); return False
        for code in self.stock_codes_initial:
            df = self._load_and_preprocess_single_stock(code, required_start_date_str, end_date)
            if df is not None and not df.empty:
                df_filtered = df[df.index >= start_dt]
                if not df_filtered.empty and len(df_filtered) >= self.window_size :
                    temp_data_dict[code] = df_filtered; successful_codes.append(code)
                    if temp_common_dates is None: temp_common_dates = df_filtered.index
                    else: temp_common_dates = temp_common_dates.intersection(df_filtered.index)
        if not temp_data_dict or not successful_codes: print("DataManager 錯誤：沒有任何股票數據成功載入。"); return False
        self.stock_codes = successful_codes; self.data_dict = temp_data_dict
        if temp_common_dates is None or len(temp_common_dates) == 0: print("DataManager 錯誤：找不到共同交易日期。"); return False
        self.common_dates = temp_common_dates.sort_values()
        if len(self.common_dates) < self.window_size + 1: print(f"DataManager 錯誤：共同交易日數據量不足 (需要 {self.window_size + 1}, 實際 {len(self.common_dates)})"); return False
        print(f"DataManager: 數據載入完成，找到 {len(self.common_dates)} 個共同交易日。")
        return True
    def get_common_dates(self): return self.common_dates;
    def get_stock_codes(self): return self.stock_codes;
    def get_data_on_date(self, stock_code, date):
        if stock_code in self.data_dict and date in self.data_dict[stock_code].index:
            data_slice = self.data_dict[stock_code].loc[[date]]; return data_slice.iloc[0] if not data_slice.empty else None
        else: return None
    # --- 修改: 返回 TEMA 週期 ---
    def get_indicator_periods(self): return {'tema_short': self.tema_short_period, 'tema_long': self.tema_long_period}

class PortfolioManager:
    """管理投資組合的狀態（單股票回測版本）。"""
    # (保持不變)
    def __init__(self, initial_capital, stock_codes):
        self.initial_capital = initial_capital; self.stock_codes = list(stock_codes);
        if len(self.stock_codes) != 1: print("警告：PortfolioManager 設計為單股票回測。")
        self.target_code = self.stock_codes[0] if self.stock_codes else None
        self.cash = initial_capital; self.shares_held = defaultdict(int); self.entry_price = defaultdict(float); self.entry_atr = defaultdict(float); self.portfolio_value = initial_capital
    def reset(self): self.cash = self.initial_capital; self.shares_held = defaultdict(int); self.entry_price = defaultdict(float); self.entry_atr = defaultdict(float); self.portfolio_value = self.initial_capital; print(f"PortfolioManager ({self.target_code}): 狀態已重設。")
    def update_on_buy(self, stock_code, shares_bought, cost, entry_atr=0): # entry_atr is optional now
        if stock_code != self.target_code or shares_bought <= 0: return; self.cash -= cost
        self.entry_price[stock_code] = cost / shares_bought if shares_bought > 0 else 0; self.entry_atr[stock_code] = entry_atr; self.shares_held[stock_code] += shares_bought
    def update_on_sell(self, stock_code, shares_sold, proceeds):
        if stock_code != self.target_code or shares_sold <= 0: return; self.cash += proceeds; self.shares_held[stock_code] -= shares_sold
        if self.shares_held[stock_code] <= 0: self.shares_held[stock_code] = 0; self.entry_price[stock_code] = 0.0; self.entry_atr[stock_code] = 0.0;
    def calculate_and_update_portfolio_value(self, data_manager: DataManager, current_date):
        total_stock_value = 0.0;
        if self.target_code:
             shares = self.shares_held[self.target_code]
             if shares > 0:
                data = data_manager.get_data_on_date(self.target_code, current_date)
                if data is not None and pd.notna(data['close']) and data['close'] > 0: total_stock_value += shares * data['close']
                else: # Fallback remains same
                    common_dates_list = data_manager.get_common_dates();
                    try:
                         if isinstance(common_dates_list, pd.DatetimeIndex):
                             current_idx = common_dates_list.get_loc(current_date); prev_idx = current_idx - 1
                             if prev_idx >= 0: prev_date = common_dates_list[prev_idx]; prev_data = data_manager.get_data_on_date(self.target_code, prev_date)
                             if prev_data is not None and pd.notna(prev_data['close']) and prev_data['close'] > 0: total_stock_value += shares * prev_data['close']
                    except Exception: pass
        self.portfolio_value = self.cash + total_stock_value; return self.portfolio_value
    def get_cash(self): return self.cash; 
    def get_shares(self, stock_code): return self.shares_held.get(stock_code, 0); 
    def get_portfolio_value(self): return self.portfolio_value; 
    def get_entry_price(self, stock_code): return self.entry_price.get(stock_code, 0.0); 
    def get_entry_atr(self, stock_code): return self.entry_atr.get(stock_code, 0.0)

class TradeExecutor:
    """負責應用資金管理規則和【模擬】提交訂單（單股票版本，AI 決定倉位比例）。"""
    # --- 修改: __init__ 接收 position_ratios ---
    def __init__(self, api: Stock_API, portfolio_manager: PortfolioManager, data_manager: DataManager,
                 position_ratios=[0.1, 0.2, 0.3], shares_per_level=1000):
        self.api = api; self.portfolio_manager = portfolio_manager; self.data_manager = data_manager
        self.position_ratios = position_ratios; self.shares_per_level = shares_per_level

    def place_sell_orders(self, sell_requests):
        # (保持不變)
        simulated_success_map = {};
        if not sell_requests: return simulated_success_map
        print("TradeExecutor: [BACKTEST] 模擬提交賣單...")
        for code, shares_api, price in sell_requests: sheets = shares_api/self.shares_per_level; print(f"  [BACKTEST] 模擬提交賣單: {sheets:.0f} 張 {code}"); simulated_success_map[code] = True
        return simulated_success_map

    def determine_and_place_buy_orders(self, buy_decision: dict, date_T):
        # (保持不變 - 已修正 NameError)
        simulated_success_map = {}; orders_to_submit_buy = []
        if not buy_decision: return orders_to_submit_buy, simulated_success_map
        current_total_value = self.portfolio_manager.get_portfolio_value(); available_cash = self.portfolio_manager.get_cash()
        if not isinstance(current_total_value, (int, float)): current_total_value = self.portfolio_manager.initial_capital
        if not isinstance(current_total_value, (int, float)): print("錯誤：無法獲取有效的 portfolio_value。"); return orders_to_submit_buy, simulated_success_map
        if current_total_value <= 0: print("TradeExecutor: 組合價值非正數，跳過買入。"); return orders_to_submit_buy, simulated_success_map
        max_capital_per_stock = current_total_value / 20.0
        print(f"TradeExecutor: [BACKTEST] 處理 AI 倉位決策... (可用現金: {available_cash:.2f})"); print(f"  資本限制: 單股上限={max_capital_per_stock:.2f}")
        for code, action_index in buy_decision.items():
             if code != self.portfolio_manager.target_code: continue
             data_T = self.data_manager.get_data_on_date(code, date_T); price_T = data_T['close'] if data_T is not None and pd.notna(data_T['close']) else 0
             if price_T <= 0: print(f"  > {code}: 無效價格，無法買入。"); continue
             if action_index >= 0 and action_index < len(self.position_ratios): target_ratio = self.position_ratios[action_index]; print(f"  > {code}: AI 選擇倉位比例 {target_ratio*100:.0f}%")
             else: print(f"  > {code}: AI 返回無效動作索引 {action_index}，跳過。"); continue
             target_capital = current_total_value * target_ratio; target_capital = min(target_capital, max_capital_per_stock)
             cost_per_sheet = self.shares_per_level * price_T;
             if cost_per_sheet <= 0: continue
             target_sheets = math.floor(target_capital / cost_per_sheet) if cost_per_sheet > 0 else 0
             if target_sheets <= 0: print(f"  > {code}: 計算目標張數為 0。"); continue
             final_cost = target_sheets * cost_per_sheet
             if available_cash < final_cost:
                 # print(f"  > {code}: 現金 ({available_cash:.2f}) 不足以購買 {target_sheets} 張 (需 {final_cost:.2f})。")
                 target_sheets = math.floor(available_cash / cost_per_sheet) if cost_per_sheet > 0 else 0
                 if target_sheets <= 0: print(f"    > 現金不足以購買任何張。"); continue
                 else: final_cost = target_sheets * cost_per_sheet; # print(f"    > 調整為購買 {target_sheets} 張。")
             if target_sheets > 0: shares_to_buy_api = target_sheets * self.shares_per_level; orders_to_submit_buy.append((code, shares_to_buy_api, price_T)); available_cash -= final_cost
        print("TradeExecutor: [BACKTEST] 模擬提交買單...")
        for code, shares_api, price in orders_to_submit_buy: sheets = shares_api/self.shares_per_level; print(f"  [BACKTEST] 模擬提交買單: {sheets:.0f} 張 {code}"); simulated_success_map[code] = True
        return orders_to_submit_buy, simulated_success_map

class SimulationEngine:
    """主回測引擎（單股票評估版本，AI決定倉位比例，TEMA 指標出場）。"""
    # --- 修改: 移除 SL/TP multiplier, 更新 features_per_stock ---
    def __init__(self, start_date, end_date, data_manager: DataManager,
                 portfolio_manager: PortfolioManager, trade_executor: TradeExecutor,
                 models: dict, # {stock_code: model}
                 position_ratios = [0.1, 0.2, 0.3]):
        self.start_date_str = start_date; self.end_date_str = end_date
        self.data_manager = data_manager; self.portfolio_manager = portfolio_manager
        self.trade_executor = trade_executor; self.models = models
        self.target_stock_code = self.portfolio_manager.target_code
        if not self.target_stock_code or self.target_stock_code not in self.models: raise ValueError("目標股票代碼無效或缺少對應模型。")
        self.stock_codes = [self.target_stock_code]
        indicator_params = data_manager.get_indicator_periods()
        self.tema_short_period = indicator_params['tema_short'] # <<<--- 獲取 TEMA 週期
        self.tema_long_period = indicator_params['tema_long']
        # --- 移除不再需要的週期 ---
        # self.ma_short_period = indicator_params['ma_short']; ...
        # --- 修改: 觀察空間維度為 6 ---
        self.features_per_stock = 6
        # ---
        self.position_ratios = position_ratios
        self.portfolio_history = []; self.dates_history = []

    # --- 修改: _get_single_stock_observation 計算 6 維 TEMA 特徵 ---
    def _get_single_stock_observation(self, stock_code, date_idx):
        common_dates = self.data_manager.get_common_dates();
        if date_idx < 0 or date_idx >= len(common_dates): return np.zeros(self.features_per_stock, dtype=np.float32)
        current_date = common_dates[date_idx]; obs_data = self.data_manager.get_data_on_date(stock_code, current_date)
        if obs_data is None: return np.zeros(self.features_per_stock, dtype=np.float32)
        try:
            close_price = obs_data['close']
            # --- 獲取 TEMA 和 斜率 ---
            tema9_val = obs_data.get(f'TEMA_{self.tema_short_period}', close_price)
            tema18_val = obs_data.get(f'TEMA_{self.tema_long_period}', close_price)
            tema9_slope = obs_data.get(f'TEMA_{self.tema_short_period}_slope', 0.0)
            tema18_slope = obs_data.get(f'TEMA_{self.tema_long_period}_slope', 0.0)
            # ---
            price_tema9_ratio = close_price / tema9_val if tema9_val != 0 else 1.0; price_tema18_ratio = close_price / tema18_val if tema18_val != 0 else 1.0; tema9_tema18_ratio = tema9_val / tema18_val if tema18_val != 0 else 1.0
            holding_position = 1.0 if self.portfolio_manager.get_shares(stock_code) > 0 else 0.0

            # --- 組裝 6 個特徵 ---
            stock_features = np.array([
                price_tema9_ratio, price_tema18_ratio, tema9_tema18_ratio, # TEMA Ratios (3)
                tema9_slope * 100, tema18_slope * 100, # TEMA Slopes (scaled) (2)
                holding_position                     # Holding (1)
            ], dtype=np.float32)
            # ---
            stock_features = np.nan_to_num(stock_features, nan=0.0, posinf=1e9, neginf=-1e9);
            if stock_features.shape[0] != self.features_per_stock: print(f"錯誤: 觀察值維度 ({stock_features.shape[0]}) 與預期 ({self.features_per_stock}) 不符！"); return np.zeros(self.observation_shape, dtype=np.float32)
            return stock_features
        except Exception as e: print(f"錯誤 ({stock_code}): Obs 未知錯誤 @ {current_date}: {e}"); traceback.print_exc(); return np.zeros(self.observation_shape, dtype=np.float32)

    # --- 入場和出場信號檢查 (使用 TEMA) ---
    def _check_entry_signal(self, current_step_idx):
        if current_step_idx < 1: return False
        current_date = self.data_manager.common_dates[current_step_idx]; yesterday_date = self.data_manager.common_dates[current_step_idx - 1]
        today_data = self.data_manager.get_data_on_date(self.target_stock_code, current_date); yesterday_data = self.data_manager.get_data_on_date(self.target_stock_code, yesterday_date)
        if today_data is None or yesterday_data is None: return False
        tema9_today = today_data.get(f'TEMA_{self.tema_short_period}', np.nan); tema18_today = today_data.get(f'TEMA_{self.tema_long_period}', np.nan)
        tema9_yesterday = yesterday_data.get(f'TEMA_{self.tema_short_period}', np.nan); tema18_yesterday = yesterday_data.get(f'TEMA_{self.tema_long_period}', np.nan)
        if pd.isna(tema9_today) or pd.isna(tema18_today) or pd.isna(tema9_yesterday) or pd.isna(tema18_yesterday): return False
        crossed_up = tema9_yesterday <= tema18_yesterday and tema9_today > tema18_today
        # 可以加入額外條件，例如價格也要在 TEMA 之上
        # price_above_tema = today_data['close'] > tema18_today
        # if crossed_up and price_above_tema: return True
        if crossed_up: return True
        return False
    def _check_exit_signal(self, current_step_idx):
        if current_step_idx < 1: return False
        current_date = self.data_manager.common_dates[current_step_idx]; yesterday_date = self.data_manager.common_dates[current_step_idx - 1]
        today_data = self.data_manager.get_data_on_date(self.target_stock_code, current_date); yesterday_data = self.data_manager.get_data_on_date(self.target_stock_code, yesterday_date)
        if today_data is None or yesterday_data is None: return False
        tema9_today = today_data.get(f'TEMA_{self.tema_short_period}', np.nan); tema18_today = today_data.get(f'TEMA_{self.tema_long_period}', np.nan)
        tema9_yesterday = yesterday_data.get(f'TEMA_{self.tema_short_period}', np.nan); tema18_yesterday = yesterday_data.get(f'TEMA_{self.tema_long_period}', np.nan)
        if pd.isna(tema9_today) or pd.isna(tema18_today) or pd.isna(tema9_yesterday) or pd.isna(tema18_yesterday): return False
        crossed_down = tema9_yesterday >= tema18_yesterday and tema9_today < tema18_today
        if crossed_down: return True
        return False

    def run_backtest(self):
        """執行單股票的回測循環 (AI決定倉位比例, TEMA 指標出場)。"""
        if not self.data_manager.load_all_data(self.start_date_str, self.end_date_str):
            print("SimulationEngine: 數據初始化失敗。")
            return
        if self.target_stock_code not in self.data_manager.get_stock_codes():
            print(f"錯誤：目標股票 {self.target_stock_code} 的數據未能成功加載。")
            return
        self.stock_codes = [self.target_stock_code]
        self.portfolio_manager.stock_codes = self.stock_codes # Ensure PM knows the target
        self.portfolio_manager.reset()
        common_dates = self.data_manager.get_common_dates()
        window_size = self.data_manager.window_size
        start_idx = window_size
        end_idx = len(common_dates) - 1
        self.portfolio_history = [self.portfolio_manager.initial_capital] * start_idx
        self.dates_history = list(common_dates[:start_idx])
        print(f"\n--- SimulationEngine: 開始回測 ({self.target_stock_code}, {common_dates[start_idx].strftime('%Y-%m-%d')} to {common_dates[end_idx].strftime('%Y-%m-%d')}) ---")

        # --- 主回測循環 ---
        for current_idx in range(start_idx, end_idx):
            date_T = common_dates[current_idx]
            date_T1 = common_dates[current_idx + 1]
            print(f"\n====== Day T: {date_T.strftime('%Y-%m-%d')} (收盤後決策) [BACKTEST MODE] ======")
            code = self.target_stock_code
            sell_requests_plan, buy_decision_dict = [], {}

            # --- 決策階段 ---
            entry_signal = self._check_entry_signal(current_idx)
            exit_signal = self._check_exit_signal(current_idx)
            current_shares = self.portfolio_manager.get_shares(code)

            if code in self.models:
                if entry_signal and current_shares == 0:
                    obs = self._get_single_stock_observation(code, current_idx)
                    if obs.shape[0] != self.features_per_stock:
                        print(f"錯誤: 觀察值維度 ({obs.shape[0]}) 與預期 ({self.features_per_stock}) 不符！")
                        continue
                    action, _states = self.models[code].predict(obs, deterministic=True)
                    buy_decision_dict[code] = action
                    print(f"  > 入場信號觸發，AI 決定倉位動作 (索引): {action}")
                elif exit_signal and current_shares > 0:
                    data_T = self.data_manager.get_data_on_date(code, date_T)
                    price_T = data_T['close'] if data_T is not None and pd.notna(data_T['close']) else 0
                    sheets_to_sell = math.floor(current_shares / 1000)
                    if sheets_to_sell > 0:
                        sell_requests_plan.append((code, sheets_to_sell * 1000, price_T))
                        print(f"  > 指標出場信號觸發，計劃賣出 {sheets_to_sell} 張")

            # --- 執行階段 (模擬提交訂單) ---
            print("--- 調用 TradeExecutor (模擬) ---")
            api_sell_success = self.trade_executor.place_sell_orders(sell_requests_plan)
            planned_buy_orders, api_buy_success = self.trade_executor.determine_and_place_buy_orders(buy_decision_dict, date_T)
            print("--- TradeExecutor 調用完成 (模擬) ---")

            # --- 結算階段 (T+1 收盤後 - 模擬成交) ---
            print(f"\n====== Day T+1: {date_T1.strftime('%Y-%m-%d')} (盤後結算) [BACKTEST MODE] ======")
            executed_trades_info = []

            # --- Backtest 模式: 模擬成交 ---
            # 模擬指標觸發的賣出
            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            # !!! 確保這個 for 循環及其內部正確縮排在 for current_idx 之下 !!!
            for code_req, shares_api, price_T in sell_requests_plan:
                if code == code_req: # Double check it's the target stock
                    data_T1 = self.data_manager.get_data_on_date(code, date_T1)
                    if data_T1 is not None:
                        price_T1_open = data_T1['open'] if pd.notna(data_T1['open']) else 0
                        # Indicator exit doesn't need price range check for execution
                        if price_T1_open > 0:
                            proceeds = shares_api * price_T1_open
                            self.portfolio_manager.update_on_sell(code, shares_api, proceeds)
                            sheets = shares_api / 1000
                            executed_trades_info.append(f"{code}:SELL_INDICATOR_{sheets:.0f}張(Sim)")
                        else:
                            print(f"[BACKTEST] 警告: {code} @ {date_T1} 開盤價無效，假設成交清空持倉。")
                            self.portfolio_manager.update_on_sell(code, shares_api, 0) # Update holding anyway
                    # else: print(f"[BACKTEST] 警告: 無法獲取 {code} @ {date_T1} 數據，無法模擬賣出。")

            # 模擬 AI 決定的買入
            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            # !!! 確保這個 for 循環及其內部正確縮排在 for current_idx 之下 !!!
            for code_req, shares_api, price_T in planned_buy_orders:
                 if code == code_req: # Double check it's the target stock
                     data_T1 = self.data_manager.get_data_on_date(code, date_T1)
                     if data_T1 is not None:
                         low_T1 = data_T1['low'] if pd.notna(data_T1['low']) else np.inf
                         price_T1_open = data_T1['open'] if pd.notna(data_T1['open']) else 0
                         atr_T1 = data_T1.get(f'ATR_{self.atr_period}', 0.0) # Get ATR for PortfolioManager update
                         # 買入成交條件：目標價 >= 最低價 (保留)
                         if price_T > 0 and low_T1 != np.inf and price_T >= low_T1:
                             if price_T1_open > 0:
                                 cost = shares_api * price_T1_open
                                 if self.portfolio_manager.get_cash() >= cost:
                                     self.portfolio_manager.update_on_buy(code, shares_api, cost, atr_T1) # Pass atr_T1
                                     sheets = shares_api / 1000
                                     executed_trades_info.append(f"{code}:BUY_{sheets:.0f}張 (Sim)")
                                 # else: print(f"[BACKTEST] 警告: 模擬買入 {code} 時現金不足。") # Optional log
                             # else: print(f"[BACKTEST] 警告: {code} @ {date_T1} 開盤價無效，無法模擬買入。") # Optional log
                         # else: print(f"[BACKTEST] 買單未觸發 (價格範圍): {code}") # Optional log
                     # else: print(f"[BACKTEST] 警告: 無法獲取 {code} @ {date_T1} 數據，無法模擬買入。") # Optional log
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

            # --- 共同邏輯: 計算並更新 T+1 收盤後的組合價值 ---
            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            # !!! 確保以下程式碼的縮排與 for current_idx... 對齊 !!!
            current_value = self.portfolio_manager.calculate_and_update_portfolio_value(self.data_manager, date_T1)
            self.portfolio_history.append(current_value)
            self.dates_history.append(date_T1)
            print(f"本日結算後組合價值: {current_value:.2f}")
            if executed_trades_info: print(f"  本日成交: {', '.join(executed_trades_info)}")
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        # --- for current_idx 循環結束 ---

        # --- 以下程式碼在循環之外 ---
        self.report_results()
        self.plot_performance()

    def report_results(self):
        final_portfolio_value = self.portfolio_manager.get_portfolio_value(); initial_capital = self.portfolio_manager.initial_capital
        total_return_pct = ((final_portfolio_value - initial_capital) / initial_capital) * 100 if initial_capital else 0
        print("\n--- SimulationEngine: 最終回測結果 ---"); print(f"股票: {self.target_stock_code}"); print(f"評估期間: {self.start_date_str} to {self.end_date_str}")
        print(f"初始資金: {initial_capital:.2f}"); print(f"最終組合價值: {final_portfolio_value:.2f}"); print(f"總回報率: {total_return_pct:.2f}%")
    def plot_performance(self):
        try:
            plt.style.use('seaborn-v0_8-darkgrid'); plt.figure(figsize=(14, 7))
            if len(self.dates_history) == len(self.portfolio_history) and len(self.dates_history) > 0 :
                portfolio_series = pd.Series(self.portfolio_history, index=self.dates_history)
                plt.plot(portfolio_series.index, portfolio_series.values, label='Portfolio Value', linewidth=1.5); plt.title(f"{self.target_stock_code} Backtest ({self.start_date_str} to {self.end_date_str}) - AI Pos Ratio / TEMA Exit") # Update title
                plt.xlabel("Date"); plt.ylabel("Portfolio Value (TWD)"); plt.legend(); plt.grid(True); plt.tight_layout(); import matplotlib.ticker as mtick
                ax = plt.gca(); ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f')); plt.xticks(rotation=45)
                plt.savefig(f"portfolio_curve_{self.target_stock_code}_backtest_tema_exit.png", dpi=300) # Update filename
                print(f"投資組合價值曲線圖已保存為 portfolio_curve_{self.target_stock_code}_backtest_tema_exit.png")
            else: print(f"警告: 日期歷史({len(self.dates_history)})與價值歷史({len(self.portfolio_history)})長度不匹配或為空，無法繪圖。")
        except ImportError: print("未安裝 matplotlib。")
        except Exception as e: print(f"繪製圖表時出錯: {e}")


# --- 主程序入口 ---
if __name__ == '__main__':
     API_ACCOUNT = "N26132089"
     API_PASSWORD = "joshua900905"
     TARGET_STOCK_CODE_EVAL = '2330'
     EVAL_START_DATE = '20230101'
     EVAL_END_DATE = '20231231'
     TOTAL_INITIAL_CAPITAL = 50000000.0
     # --- 修改: 指向 TEMA 模型目錄 ---
     EXPERIMENT_NAME = "ai_pos_ratio_tema_exit_tema_features_v1" # <<<--- 確保與訓練時一致
     MODELS_BASE_DIR = f"tuned_models/{TARGET_STOCK_CODE_EVAL}/{EXPERIMENT_NAME}"
     MODEL_FILE_NAME = f"ppo_agent_{TARGET_STOCK_CODE_EVAL}_final.zip"
     MODEL_LOAD_PATH = os.path.join(MODELS_BASE_DIR, MODEL_FILE_NAME)
     # ---

     # --- 指標和窗口參數 (必須與訓練時完全一致) ---
     TEMA_SHORT = 9; TEMA_LONG = 18
     # --- 提供其他週期給 DataManager 計算窗口大小 ---
     RSI_PERIOD_EVAL = 14; ATR_PERIOD_EVAL = 14; MACD_SLOW_EVAL = 26
     # --- 倉位比例定義 ---
     POSITION_RATIOS_EVAL = [0.1, 0.2, 0.3] # <<<--- 必須與訓練時一致
     SHARES_PER_LEVEL_EVAL = 1000
     # ---

     RUN_TRAINING = False; RUN_EVALUATION = True

     if RUN_TRAINING: print("錯誤：此腳本僅用於單股票評估。")
     if RUN_EVALUATION:
        print(f"\n=============== 開始單股票回測 ({TARGET_STOCK_CODE_EVAL} - TEMA Exit Model) ===============")
        if not os.path.exists(MODEL_LOAD_PATH): print(f"錯誤：找不到模型文件 '{MODEL_LOAD_PATH}'。"); exit()
        else:
             print("--- 初始化評估組件 ---")
             api_eval = Stock_API(API_ACCOUNT, API_PASSWORD)
             # --- 修改: DataManager 初始化使用 TEMA 參數 ---
             data_manager_eval = DataManager(
                 stock_codes_initial=[TARGET_STOCK_CODE_EVAL], api=api_eval,
                 # window_size 在內部計算
                 tema_short=TEMA_SHORT, tema_long=TEMA_LONG,
                 rsi_period=RSI_PERIOD_EVAL, atr_period=ATR_PERIOD_EVAL, # 仍然需要傳遞用於計算 window_size
                 macd_slow=MACD_SLOW_EVAL # 傳遞 macd_slow 用於計算 window_size
             )
             # ---
             if data_manager_eval.load_all_data(EVAL_START_DATE, EVAL_END_DATE):
                 portfolio_manager_eval = PortfolioManager( initial_capital=TOTAL_INITIAL_CAPITAL, stock_codes=data_manager_eval.get_stock_codes() )
                 # --- 修改: TradeExecutor 初始化傳遞倉位定義 ---
                 trade_executor_eval = TradeExecutor( api=api_eval, portfolio_manager=portfolio_manager_eval, data_manager=data_manager_eval, position_ratios=POSITION_RATIOS_EVAL, shares_per_level=SHARES_PER_LEVEL_EVAL )
                 models_eval = {}; print(f"--- 從 '{MODEL_LOAD_PATH}' 加載預訓練模型 ---");
                 try: models_eval[TARGET_STOCK_CODE_EVAL] = PPO.load(MODEL_LOAD_PATH); print(f"  > 已成功加載模型: {TARGET_STOCK_CODE_EVAL}")
                 except Exception as e: print(f"加載模型 {TARGET_STOCK_CODE_EVAL} 失敗: {e}"); exit()
                 # --- 修改: SimulationEngine 初始化傳遞倉位比例 ---
                 simulation_engine = SimulationEngine(
                     start_date=EVAL_START_DATE, end_date=EVAL_END_DATE, data_manager=data_manager_eval,
                     portfolio_manager=portfolio_manager_eval, trade_executor=trade_executor_eval, models=models_eval,
                     position_ratios=POSITION_RATIOS_EVAL
                 )
                 # ---
                 simulation_engine.run_backtest()
             else: print("數據管理器初始化失敗，無法進行評估。")
        print("\n=============== 單股票回測完成 ===============")

     if not RUN_TRAINING and not RUN_EVALUATION: print("\n請設置 RUN_EVALUATION=True 來執行回測。")
     print("\n--- 程序執行完畢 ---")