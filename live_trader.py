# -*- coding: utf-8 -*-
# live_trader.py - Script for placing live orders to the simulation platform

import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import time
import os
import math
from collections import defaultdict
import traceback
from datetime import datetime, timedelta # Import timedelta

# --- Stable Baselines 3 Import ---
try:
    from stable_baselines3 import PPO
except ImportError:
    print("錯誤：找不到 stable_baselines3。請運行 'pip install stable_baselines3'")
    exit()

# --- Stock API Class ---
class Stock_API:
    # ... (與 evaluate_models.py 中版本相同) ...
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
            if result.get('result') == 'success':
                holdings_data = result.get('data', []);
                if isinstance(holdings_data, list):
                    processed_holdings = []
                    for stock in holdings_data:
                         try: processed_holdings.append({'stock_code': str(stock.get('stock_code')), 'shares': int(stock.get('shares', 0)), 'price': float(stock.get('price', 0.0)), 'amount': float(stock.get('amount', 0.0))})
                         except (ValueError, TypeError) as e: print(f"處理持股數據類型轉換錯誤 for {stock.get('stock_code')}: {e}")
                    return processed_holdings
                else: return []
            else: print(f"API 錯誤 (Get_User_Stocks): {result.get('status', '未知狀態')}"); return []
        except Exception as e: print(f"處理用戶持股時出錯 (Get_User_Stocks): {e}"); return []
        return []
    def Buy_Stock(self, stock_code, stock_shares, stock_price):
        stock_shares = int(stock_shares);
        if stock_shares <= 0 or stock_shares % 1000 != 0: print(f"買單股數錯誤 ({stock_shares} 股 {stock_code})。不提交。"); return False
        sheets = stock_shares / 1000; print(f"[LIVE] 嘗試提交買單: {sheets:.0f} 張 ({stock_shares} 股) {stock_code} @ 目標價 {stock_price:.2f}")
        data = {'account': self.account, 'password': self.password, 'stock_code': str(stock_code),'stock_shares': stock_shares, 'stock_price': float(stock_price)}
        buy_url = f'{self.base_url}/buy';
        try:
            response = requests.post(buy_url, data=data, timeout=15); response.raise_for_status(); result = response.json()
            print(f"[LIVE] 買單提交響應: 結果={result.get('result', 'N/A')}, 狀態={result.get('status', 'N/A')}"); return result.get('result') == 'success'
        except Exception as e: print(f"[LIVE] 處理買單提交時出錯 ({stock_code}): {e}"); return False
    def Sell_Stock(self, stock_code, stock_shares, stock_price):
        stock_shares = int(stock_shares);
        if stock_shares <= 0 or stock_shares % 1000 != 0: print(f"賣單股數錯誤 ({stock_shares} 股 {stock_code})。不提交。"); return False
        sheets = stock_shares / 1000; print(f"[LIVE] 嘗試提交賣單: {sheets:.0f} 張 ({stock_shares} 股) {stock_code} @ 目標價 {stock_price:.2f}")
        data = {'account': self.account, 'password': self.password, 'stock_code': str(stock_code),'stock_shares': stock_shares, 'stock_price': float(stock_price)}
        sell_url = f'{self.base_url}/sell';
        try:
            response = requests.post(sell_url, data=data, timeout=15); response.raise_for_status(); result = response.json()
            print(f"[LIVE] 賣單提交響應: 結果={result.get('result', 'N/A')}, 狀態={result.get('status', 'N/A')}"); return result.get('result') == 'success'
        except Exception as e: print(f"[LIVE] 處理賣單提交時出錯 ({stock_code}): {e}"); return False

# --- DataManager Class (只需要獲取當天和近期數據計算指標) ---
class DataManagerForLive:
    """僅用於 Live Trading，獲取計算當前觀察值所需的數據。"""
    def __init__(self, stock_codes, api, window_size,
                 ma_short=10, rsi_period=14, atr_period=14):
        self.stock_codes = list(stock_codes); self.api = api; self.window_size = window_size
        self.ma_short_period = ma_short; self.rsi_period = rsi_period; self.atr_period = atr_period
        self.data_dict = {} # 只存儲計算指標所需的近期數據

    def _load_and_preprocess_recent_data(self, stock_code, today_dt):
        """獲取從足夠早的日期到今天的數據，用於計算指標。"""
        buffer_days = 30
        # 計算需要的數據起始日期
        required_start_dt = today_dt - pd.Timedelta(days=(self.window_size + buffer_days) * 1.5)
        start_date_str = required_start_dt.strftime('%Y%m%d')
        end_date_str = today_dt.strftime('%Y%m%d') # 獲取到今天為止的數據
        # print(f"  DataManagerLive: 載入 {stock_code} 數據 {start_date_str} to {end_date_str}")
        raw_data = self.api.Get_Stock_Informations(stock_code, start_date_str, end_date_str)
        if not raw_data: return None
        try:
            df = pd.DataFrame(raw_data); df['date'] = pd.to_datetime(df['date'], unit='s')
            df = df.sort_values('date').set_index('date')
            required_cols = ['open', 'high', 'low', 'close', 'turnover']
            if not all(col in df.columns for col in required_cols): return None
            numeric_cols = ['open', 'high', 'low', 'close', 'turnover', 'capacity', 'transaction_volume']
            for col in numeric_cols:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
            if 'turnover' in df.columns and 'close' in df.columns:
                 df['volume'] = np.where(df['close'].isna() | (df['close'] == 0), 0, df['turnover'] / df['close'])
                 df['volume'] = df['volume'].fillna(0).replace([np.inf, -np.inf], 0).round().astype(np.int64)
            else: df['volume'] = 0
            indicator_base_cols = ['open', 'high', 'low', 'close']; df = df.dropna(subset=indicator_base_cols)
            if df.empty: return None
            # 計算指標
            df.ta.sma(length=self.ma_short_period, close='close', append=True, col_names=(f'SMA_{self.ma_short_period}',))
            df.ta.rsi(length=self.rsi_period, close='close', append=True, col_names=(f'RSI_{self.rsi_period}',))
            df.ta.atr(length=self.atr_period, high='high', low='low', close='close', append=True, col_names=(f'ATR_{self.atr_period}',))
            if f'ATR_{self.atr_period}' not in df.columns: return None
            df[f'ATR_norm_{self.atr_period}'] = df[f'ATR_{self.atr_period}'] / df['close']; df[f'ATR_norm_{self.atr_period}'] = df[f'ATR_norm_{self.atr_period}'].replace([np.inf, -np.inf], 0)
            df = df.dropna(); # 移除 NaN
            if df.empty: return None
            # 只保留最後 window_size + buffer 天的數據就夠計算了，減少內存佔用（可選）
            # df = df.tail(self.window_size + buffer_days)
            return df
        except Exception as e: print(f"DataManagerLive: 處理 {stock_code} 數據時出錯: {e}"); traceback.print_exc(); return None

    def update_data_for_today(self, today_dt=None):
        """更新所有股票到指定日期的數據。"""
        if today_dt is None:
            today_dt = pd.Timestamp(datetime.now().date()) # 使用 Pandas Timestamp
        print(f"DataManagerLive: 更新數據至 {today_dt.strftime('%Y-%m-%d')}...")
        successful_codes = []
        for code in self.stock_codes:
            df_recent = self._load_and_preprocess_recent_data(code, today_dt)
            if df_recent is not None and not df_recent.empty:
                # 確保今天日期在數據中
                if today_dt in df_recent.index:
                    self.data_dict[code] = df_recent
                    successful_codes.append(code)
                else:
                    print(f"  > 警告: 股票 {code} 加載的最新數據不包含今天 ({today_dt.strftime('%Y-%m-%d')})。")
            else:
                print(f"  > 警告: 股票 {code} 無法加載或處理近期數據。")
        # 更新實際可用的股票列表
        self.stock_codes = successful_codes
        print(f"DataManagerLive: 數據更新完成，可用股票 {len(self.stock_codes)} 支。")
        return len(self.stock_codes) > 0

    def get_data_on_date(self, stock_code, date):
        """獲取特定日期的數據 Series。"""
        if stock_code in self.data_dict and date in self.data_dict[stock_code].index:
            # 直接用 .loc 獲取 Series
            return self.data_dict[stock_code].loc[date]
        else:
            return None
    def get_stock_codes(self): return self.stock_codes
    def get_indicator_periods(self): return {'ma_short': self.ma_short_period, 'rsi': self.rsi_period, 'atr': self.atr_period}

# --- Portfolio Manager Class (只需要查詢當前持股和現金) ---
class PortfolioManagerForLive:
    """管理實時交易中的投資組合狀態（主要用於查詢）。"""
    def __init__(self, api: Stock_API, initial_capital=0): # 初始資金可能不重要，以API為準
        self.api = api
        self.cash = initial_capital # 可以初始化，但實際應以 API 查詢為準 (如果API提供)
        self.shares_held = defaultdict(int)
        self.entry_price = defaultdict(float) # Live 模式下追蹤成本價比較複雜
        self.entry_atr = defaultdict(float)
        self.portfolio_value = initial_capital # 同上
        self.stock_codes = [] # 記錄當前持有的股票代碼
        self.update_holdings() # 初始化時查詢一次持股

    def update_holdings(self):
        """從 API 更新當前持股狀態。"""
        print("PortfolioManagerLive: 正在從 API 查詢當前持股...")
        holdings = self.api.Get_User_Stocks()
        # 清空舊狀態
        self.shares_held.clear()
        self.entry_price.clear() # 成本價需要更複雜的邏輯或依賴API提供
        current_stock_codes = []
        if holdings:
            for stock_info in holdings:
                code = stock_info['stock_code']
                shares = stock_info['shares']
                if shares > 0:
                    self.shares_held[code] = shares
                    current_stock_codes.append(code)
                    # 如果 API 提供成本價，可以在這裡記錄
                    # self.entry_price[code] = stock_info['price']
            print(f"  > 持股已更新: {dict(self.shares_held)}")
            self.stock_codes = current_stock_codes
        else:
            print("  > 未查詢到持股或 API 錯誤。")
            self.stock_codes = []
        # 注意：這裡沒有更新現金和總價值，因為 API 可能不直接提供
        # Live trading 的現金和價值計算可能需要更複雜的賬戶管理系統

    def get_shares(self, stock_code): return self.shares_held.get(stock_code, 0)
    # 在 Live 模式下，portfolio_value 和 cash 的準確性依賴於是否能查詢賬戶信息
    def get_portfolio_value(self): return self.portfolio_value # 可能不準確
    def get_cash(self): return self.cash # 可能不準確
    def get_entry_price(self, stock_code): return self.entry_price.get(stock_code, 0.0) # 可能不準確
    def get_entry_atr(self, stock_code): return self.entry_atr.get(stock_code, 0.0) # Live模式下可能不需要

# --- Trade Executor Class (用於 Live Trading) ---
class TradeExecutorForLive:
    """負責應用資金管理規則並實際提交 API 訂單。"""
    def __init__(self, api: Stock_API, portfolio_manager: PortfolioManagerForLive, data_manager: DataManagerForLive):
        self.api = api
        self.portfolio_manager = portfolio_manager
        self.data_manager = data_manager

    def place_sell_orders(self, sell_requests):
        """實際提交賣單。"""
        api_success_map = {}
        if not sell_requests: return api_success_map
        print("TradeExecutorLive: 提交賣單請求至 API...")
        for code, shares_api, price in sell_requests:
            api_success_map[code] = self.api.Sell_Stock(code, shares_api, price)
            time.sleep(0.1)
        return api_success_map

    def determine_and_place_buy_orders(self, buy_requests, date_T):
        """根據資金管理規則確定最終買單並實際提交。"""
        api_success_map = {}
        orders_to_submit_buy = []

        # --- 在 Live 模式下，需要獲取「當前」的帳戶價值和現金 ---
        # --- 這通常需要額外的 API 調用或手動設置 ---
        # --- 為了簡化，我們先用初始資金作為估算基礎，這很不準確！ ---
        # --- 在實際應用中，你需要一個方法來獲取真實的帳戶總值和可用現金 ---
        current_total_value = self.portfolio_manager.get_portfolio_value() # 獲取的值可能不準
        available_cash = self.portfolio_manager.get_cash()           # 獲取的值可能不準
        print(f"警告：Live 模式下的組合價值 ({current_total_value:.2f}) 和現金 ({available_cash:.2f}) 可能不準確，依賴於 PortfolioManager 的更新。")

        if not isinstance(current_total_value, (int, float)) or current_total_value <= 0:
             print("錯誤：無法獲取有效的帳戶價值，使用初始資本估算。")
             current_total_value = self.portfolio_manager.initial_capital # Fallback
             if not isinstance(current_total_value, (int, float)) or current_total_value <= 0:
                  print("錯誤：初始資本也無效，無法計算買單。")
                  return orders_to_submit_buy, api_success_map

        if not isinstance(available_cash, (int, float)):
            print("錯誤：無法獲取有效的可用現金，假設等於總價值（非常不安全）。")
            available_cash = current_total_value

        max_capital_per_stock = current_total_value / 20.0
        max_capital_per_trade = current_total_value / 100.0
        print(f"TradeExecutorLive: 處理買單... (可用現金估算: {available_cash:.2f})")
        print(f"  資本限制: 單股上限={max_capital_per_stock:.2f}, 單筆交易上限={max_capital_per_trade:.2f}")
        buy_requests.sort(key=lambda x: x[0])

        for code, price_T in buy_requests:
            if price_T <= 0: continue; cost_per_sheet = 1000 * price_T
            if cost_per_sheet <= 0: continue; max_sheets_trade = math.floor(max_capital_per_trade / cost_per_sheet) if cost_per_sheet > 0 else 0
            if max_sheets_trade <= 0: continue; target_sheets = max_sheets_trade; required_cash_for_target = target_sheets * cost_per_sheet
            if available_cash < required_cash_for_target:
                sheets_can_afford = math.floor(available_cash / cost_per_sheet) if cost_per_sheet > 0 else 0
                if sheets_can_afford <= 0: continue; 
                else: target_sheets = sheets_can_afford; potential_cost = target_sheets * cost_per_sheet;
            current_shares = self.portfolio_manager.get_shares(code) # 查詢當前持股
            current_stock_value = current_shares * price_T
            potential_cost = target_sheets * cost_per_sheet; potential_new_value = current_stock_value + potential_cost
            if potential_new_value > max_capital_per_stock:
                allowed_additional_capital = max_capital_per_stock - current_stock_value
                if allowed_additional_capital > 0:
                     allowed_additional_sheets = math.floor(allowed_additional_capital / cost_per_sheet) if cost_per_sheet > 0 else 0
                     final_sheets_to_buy = min(target_sheets, allowed_additional_sheets);
                     if final_sheets_to_buy > 0: target_sheets = final_sheets_to_buy; potential_cost = target_sheets * cost_per_sheet
                     else: continue
                else: continue
            if target_sheets <= 0: continue; final_cost = potential_cost
            if available_cash < final_cost: print(f"  > {code}: 最終現金檢查不足 ({available_cash:.2f} < {final_cost:.2f})"); continue;
            shares_to_buy_api = target_sheets * 1000
            orders_to_submit_buy.append((code, shares_to_buy_api, price_T)); available_cash -= final_cost # 更新估算的可用現金
            # print(f"  > {code}: 計劃買入 {target_sheets} 張。")

        # --- 實際提交 API 請求 ---
        print("TradeExecutorLive: [LIVE] 提交買單請求至 API...")
        for code, shares_api, price in orders_to_submit_buy:
            api_success_map[code] = self.api.Buy_Stock(code, shares_api, price); time.sleep(0.1)
        return orders_to_submit_buy, api_success_map # 返回計劃提交的訂單和結果

# --- Live Trader Main Logic ---
class LiveTrader:
    """執行每日下單邏輯。"""
    def __init__(self, stock_codes, api_account, api_password, models_dir,
                 ma_short=10, rsi_period=14, atr_period=14,
                 sl_multiplier=2.0, tp_multiplier=3.0, window_size=60):
        self.api = Stock_API(api_account, api_password)
        self.models_dir = models_dir
        # 注意：這裡的 PortfolioManager 實例主要用於查詢當前持股，資金計算依賴外部或估算
        self.portfolio_manager = PortfolioManagerForLive(self.api)
        self.data_manager = DataManagerForLive(stock_codes, self.api, window_size, ma_short, rsi_period, atr_period)
        self.trade_executor = TradeExecutorForLive(self.api, self.portfolio_manager, self.data_manager)
        self.models = self._load_models(stock_codes) # 加載模型時使用初始列表

        # 指標參數，用於觀察值計算
        self.ma_short_period = ma_short; self.rsi_period = rsi_period; self.atr_period = atr_period
        self.sl_multiplier = sl_multiplier; self.tp_multiplier = tp_multiplier
        self.features_per_stock = 7

    def _load_models(self, stock_codes_to_load):
        models = {}; print("LiveTrader: 正在加載預訓練模型...")
        loaded_codes = []
        for code in stock_codes_to_load:
            model_path = os.path.join(self.models_dir, f"ppo_agent_{code}")
            if os.path.exists(model_path + ".zip"):
                try: models[code] = PPO.load(model_path); print(f"  > 已加載模型: {code}"); loaded_codes.append(code)
                except Exception as e: print(f"加載模型 {code} 失敗: {e}")
            else: print(f"警告: 找不到模型文件 {model_path}.zip for stock {code}")
        if not models: raise ValueError("沒有成功加載任何模型。")
        # 更新實際使用的股票列表（基於成功加載的模型）
        self.data_manager.stock_codes = loaded_codes
        print(f"LiveTrader: 成功加載 {len(models)} 個模型。")
        return models

    def _get_single_stock_observation(self, stock_code, date_T):
        """計算單支股票在 T 日收盤後的觀察狀態。"""
        obs_data = self.data_manager.get_data_on_date(stock_code, date_T)
        if obs_data is None: return np.zeros(self.features_per_stock, dtype=np.float32)
        try:
            close_price = obs_data['close']; atr_val = obs_data.get(f'ATR_{self.atr_period}', 0.0); atr_norm_val = obs_data.get(f'ATR_norm_{self.atr_period}', 0.0)
            ma_short_val = obs_data.get(f'SMA_{self.ma_short_period}', close_price); rsi_val_raw = obs_data.get(f'RSI_{self.rsi_period}', 50.0)
            price_ma_ratio = close_price / ma_short_val if ma_short_val != 0 else 1.0; rsi_val = rsi_val_raw / 100.0
            # --- Live 模式下獲取持倉 ---
            holding_position = 1.0 if self.portfolio_manager.get_shares(stock_code) > 0 else 0.0
            # --- Live 模式下 SL/TP 特徵可能不準確或不需要 ---
            # --- 可以簡化或移除，因為決策主要基於指標 ---
            distance_to_sl_norm, distance_to_tp_norm, is_below_potential_sl = 0.0, 0.0, 0.0
            # 或者保留計算，但知道 entry_price/atr 可能不準
            entry_p, entry_a = self.portfolio_manager.get_entry_price(stock_code), self.portfolio_manager.get_entry_atr(stock_code)
            if holding_position > 0 and entry_p > 0 and entry_a > 0:
                 potential_sl = entry_p - self.sl_multiplier * entry_a; potential_tp = entry_p + self.tp_multiplier * entry_a
                 if close_price > 0: distance_to_sl_norm = (close_price - potential_sl) / close_price; distance_to_tp_norm = (potential_tp - close_price) / close_price
                 if close_price < potential_sl and potential_sl > 0: is_below_potential_sl = 1.0

            stock_features = np.array([ price_ma_ratio, rsi_val, atr_norm_val, holding_position, distance_to_sl_norm, distance_to_tp_norm, is_below_potential_sl ], dtype=np.float32)
            stock_features = np.nan_to_num(stock_features, nan=0.0, posinf=1e9, neginf=-1e9); return stock_features
        except Exception as e: print(f"錯誤 ({stock_code}): Live Obs 未知錯誤 @ {date_T}: {e}"); traceback.print_exc(); return np.zeros(self.features_per_stock, dtype=np.float32)

    def run_daily_order_placement(self):
        """執行每日的下單流程。"""
        today_dt = pd.Timestamp(datetime.now().date())
        print(f"\n====== Live Trader: {today_dt.strftime('%Y-%m-%d')} 收盤後 ======")

        # 1. 更新數據到今天
        if not self.data_manager.update_data_for_today(today_dt):
            print("錯誤：無法更新今日數據，終止下單流程。")
            return

        # 2. 更新當前持股狀態
        self.portfolio_manager.update_holdings()

        # 3. 收集決策
        sell_requests_plan, buy_requests_plan = [], []
        print("LiveTrader: 正在生成交易決策...")
        for code in self.data_manager.get_stock_codes(): # 使用更新後的列表
             if code in self.models:
                 # 使用今天的日期獲取觀察值
                 obs = self._get_single_stock_observation(code, today_dt)
                 if np.any(obs): # 確保觀察值有效 (非全零)
                     action, _states = self.models[code].predict(obs, deterministic=True)
                     data_T = self.data_manager.get_data_on_date(code, today_dt)
                     price_T = data_T['close'] if data_T is not None and pd.notna(data_T['close']) else 0

                     current_shares = self.portfolio_manager.get_shares(code) # 使用更新後的持股

                     if action == 2 and current_shares > 0:
                         sheets_to_sell = math.floor(current_shares / 1000)
                         if sheets_to_sell > 0:
                             shares_to_sell_api = sheets_to_sell * 1000
                             sell_requests_plan.append((code, shares_to_sell_api, price_T))
                             print(f"  > 決策: 賣出 {sheets_to_sell:.0f} 張 {code}")
                     elif action == 1 and current_shares == 0 and price_T > 0:
                         buy_requests_plan.append((code, price_T))
                         print(f"  > 決策: 買入 {code}")
                     # else: print(f"  > 決策: 持有 {code}") # 可以取消註釋以查看持有決策
                 else:
                      print(f"  > 警告: 無法為 {code} 生成有效的觀察值，跳過決策。")


        # 4. 執行下單 (始終使用 is_live=True)
        print("\n--- LiveTrader: 調用 TradeExecutor 執行下單 ---")
        api_sell_success = self.trade_executor.place_sell_orders(sell_requests_plan, is_live=True)
        # determine_and_place_buy_orders 內部會應用資金管理
        planned_buy_orders, api_buy_success = self.trade_executor.determine_and_place_buy_orders(buy_requests_plan, today_dt, is_live=True) # date_T is today
        print("--- LiveTrader: TradeExecutor 調用完成 ---")

        # 5. 打印總結 (可選)
        print("\n--- LiveTrader: 今日預約單提交總結 ---")
        for code, shares, _ in sell_requests_plan:
            success = api_sell_success.get(code, False)
            print(f"  賣單: {code} ({shares/1000:.0f}張) - 提交 {'成功' if success else '失敗'}")
        for code, shares, _ in planned_buy_orders: # 使用 planned_buy_orders
            success = api_buy_success.get(code, False)
            print(f"  買單: {code} ({shares/1000:.0f}張) - 提交 {'成功' if success else '失敗'}")
        print("--- LiveTrader: 下單流程結束 ---")


# --- 主程序入口 ---
if __name__ == '__main__':
     # --- !!! 重要：請替換為您的 API 憑證 !!! ---
     API_ACCOUNT = "N26132089"
     API_PASSWORD = "joshua900905"

     # --- 股票列表 ---
     STOCK_CODES_LIST = ['2330','2454','2317','2308','2881','2891','2382','2303','2882','2412',
                    '2886','3711','2884','2357','1216','2885','3034','3231','2892','2345']

     # --- 模型和參數設定 (應與訓練時一致) ---
     MODELS_LOAD_DIR = "trained_individual_models_ma10_enhanced_reward" # 指向最新的模型目錄
     MA_SHORT = 10
     RSI_PERIOD = 14
     ATR_PERIOD = 14
     SL_ATR_MULT = 2.0 # 用於觀察值計算
     TP_ATR_MULT = 3.0 # 用於觀察值計算
     WINDOW_SIZE = max(MA_SHORT, RSI_PERIOD, ATR_PERIOD) + 10

     print("=============== Live Trader 啟動 ===============")
     if not os.path.exists(MODELS_LOAD_DIR):
         print(f"錯誤：找不到模型目錄 '{MODELS_LOAD_DIR}'。請先運行訓練。")
         exit()

     try:
         # --- 創建 Live Trader 實例 ---
         live_trader = LiveTrader(
             stock_codes=STOCK_CODES_LIST,
             api_account=API_ACCOUNT,
             api_password=API_PASSWORD,
             models_dir=MODELS_LOAD_DIR,
             ma_short=MA_SHORT,
             rsi_period=RSI_PERIOD,
             atr_period=ATR_PERIOD,
             sl_multiplier=SL_ATR_MULT,
             tp_multiplier=TP_ATR_MULT,
             window_size=WINDOW_SIZE
         )

         # --- 執行每日下單 ---
         live_trader.run_daily_order_placement()

     except ValueError as e:
          print(f"初始化錯誤: {e}")
     except Exception as e:
          print(f"執行過程中發生未預期的錯誤: {e}")
          traceback.print_exc()

     print("\n=============== Live Trader 執行完畢 ===============")