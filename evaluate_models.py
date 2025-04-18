# -*- coding: utf-8 -*-
# evaluate_models.py - Script for Evaluating Coordinated Models

import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import time
import os
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import traceback # Import traceback for better error printing

# --- Stable Baselines 3 Import ---
try:
    from stable_baselines3 import PPO
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
                     # print(f"API 警告 (Get_Stock_Informations - {stock_code}): 返回成功但數據列表為空。") # Reduced verbosity
                     pass
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
                                   'shares': int(stock.get('shares', 0)), # 假設 API 返回股數
                                   'price': float(stock.get('price', 0.0)),
                                   'amount': float(stock.get('amount', 0.0))
                              }
                              processed_holdings.append(processed_stock)
                         except (ValueError, TypeError) as e:
                              print(f"處理持股數據類型轉換錯誤 for {stock.get('stock_code')}: {e}")
                    return processed_holdings
                else:
                    # print(f"警告: Get_User_Stocks 返回非列表數據: {holdings_data}") # Reduced verbosity
                    return []
            else:
                print(f"API 錯誤 (Get_User_Stocks): {result.get('status', '未知狀態')}")
                return []
        except requests.exceptions.Timeout:
            print("請求超時 (Get_User_Stocks)")
            return []
        except requests.exceptions.RequestException as e:
            print(f"網路錯誤 (Get_User_Stocks): {e}")
            return []
        except Exception as e:
            print(f"處理用戶持股時出錯 (Get_User_Stocks): {e}")
            return []
        return [] # Return empty list on failure

    def Buy_Stock(self, stock_code, stock_shares, stock_price):
        """提交購買股票預約單 (單位: 股，但必須是 1000 的倍數)"""
        stock_shares = int(stock_shares)
        if stock_shares <= 0 or stock_shares % 1000 != 0:
            print(f"買單股數錯誤 ({stock_shares} 股 {stock_code}), 必須是 1000 的正整數倍。不提交。")
            return False
        sheets = stock_shares / 1000
        print(f"嘗試提交買單: {sheets:.0f} 張 ({stock_shares} 股) {stock_code} @ 目標價 {stock_price:.2f}")
        data = {'account': self.account, 'password': self.password, 'stock_code': str(stock_code),
                'stock_shares': stock_shares, 'stock_price': float(stock_price)}
        buy_url = f'{self.base_url}/buy'
        try:
            response = requests.post(buy_url, data=data, timeout=15)
            response.raise_for_status()
            result = response.json()
            print(f"買單提交響應: 結果={result.get('result', 'N/A')}, 狀態={result.get('status', 'N/A')}")
            return result.get('result') == 'success'
        except requests.exceptions.Timeout: print(f"提交買單請求超時 ({stock_code})"); return False
        except requests.exceptions.RequestException as e: print(f"提交買單時網路錯誤 ({stock_code}): {e}"); return False
        except Exception as e: print(f"處理買單提交時出錯 ({stock_code}): {e}"); return False

    def Sell_Stock(self, stock_code, stock_shares, stock_price):
        """提交賣出股票預約單 (單位: 股，但應基於持有的完整張數)"""
        stock_shares = int(stock_shares)
        if stock_shares <= 0 or stock_shares % 1000 != 0:
            print(f"賣單股數錯誤 ({stock_shares} 股 {stock_code}), 必須是 1000 的正整數倍。不提交。")
            return False
        sheets = stock_shares / 1000
        print(f"嘗試提交賣單: {sheets:.0f} 張 ({stock_shares} 股) {stock_code} @ 目標價 {stock_price:.2f}")
        data = {'account': self.account, 'password': self.password, 'stock_code': str(stock_code),
                'stock_shares': stock_shares, 'stock_price': float(stock_price)}
        sell_url = f'{self.base_url}/sell'
        try:
            response = requests.post(sell_url, data=data, timeout=15)
            response.raise_for_status()
            result = response.json()
            print(f"賣單提交響應: 結果={result.get('result', 'N/A')}, 狀態={result.get('status', 'N/A')}")
            return result.get('result') == 'success'
        except requests.exceptions.Timeout: print(f"提交賣單請求超時 ({stock_code})"); return False
        except requests.exceptions.RequestException as e: print(f"提交賣單時網路錯誤 ({stock_code}): {e}"); return False
        except Exception as e: print(f"處理賣單提交時出錯 ({stock_code}): {e}"); return False


# --- Refactored Evaluation Classes ---

class DataManager:
    """負責加載、預處理和提供市場數據 (已適應新 API 格式)。"""
    def __init__(self, stock_codes_initial, api, window_size,
                 ma_long=50, rsi_period=14, atr_period=14):
        self.stock_codes_initial = list(stock_codes_initial)
        self.api = api
        self.window_size = window_size
        self.ma_long_period = ma_long
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.data_dict = {}
        self.common_dates = None
        self.stock_codes = [] # 實際成功加載數據的股票列表

    def _load_and_preprocess_single_stock(self, stock_code, start_date, end_date):
        """載入、清洗數據並計算指標 (已適應新的 API 格式)。"""
        # print(f"  DataManager: 載入數據 {stock_code} ({start_date} to {end_date})") # Reduced verbosity
        raw_data = self.api.Get_Stock_Informations(stock_code, start_date, end_date)
        if not raw_data: return None
        try:
            df = pd.DataFrame(raw_data)
            if df.empty: return None

            # --- 修改: 處理 Unix Timestamp 日期 ---
            df['date'] = pd.to_datetime(df['date'], unit='s')
            df = df.sort_values('date').set_index('date')

            # --- 修改: 處理欄位名稱和成交量 ---
            required_cols = ['open', 'high', 'low', 'close', 'turnover']
            if not all(col in df.columns for col in required_cols):
                 print(f"    錯誤：{stock_code} 缺少必要的 OHLC 或 Turnover 欄位。")
                 return None

            numeric_cols = ['open', 'high', 'low', 'close', 'turnover', 'capacity', 'transaction_volume']
            for col in numeric_cols:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

            # 計算近似成交量 volume = turnover / close
            if 'turnover' in df.columns and 'close' in df.columns:
                 df['volume'] = np.where(df['close'].isna() | (df['close'] == 0), 0, df['turnover'] / df['close'])
                 df['volume'] = df['volume'].fillna(0).replace([np.inf, -np.inf], 0)
                 df['volume'] = df['volume'].round().astype(np.int64)
            else: df['volume'] = 0

            # --- 後續處理 ---
            indicator_base_cols = ['open', 'high', 'low', 'close']
            df = df.dropna(subset=indicator_base_cols)
            if df.empty: return None

            df.ta.sma(length=self.ma_long_period, close='close', append=True, col_names=(f'SMA_{self.ma_long_period}',))
            df.ta.rsi(length=self.rsi_period, close='close', append=True, col_names=(f'RSI_{self.rsi_period}',))
            df.ta.atr(length=self.atr_period, high='high', low='low', close='close', append=True, col_names=(f'ATR_{self.atr_period}',))

            if f'ATR_{self.atr_period}' not in df.columns: return None
            df[f'ATR_norm_{self.atr_period}'] = df[f'ATR_{self.atr_period}'] / df['close']
            df[f'ATR_norm_{self.atr_period}'] = df[f'ATR_norm_{self.atr_period}'].replace([np.inf, -np.inf], 0)
            df = df.dropna()
            if df.empty: return None

            return df
        except Exception as e:
            print(f"DataManager: 處理 {stock_code} 數據時出錯: {e}")
            traceback.print_exc()
            return None

    def load_all_data(self, start_date, end_date):
        print(f"DataManager: 正在載入所有股票數據 ({start_date} to {end_date})...")
        temp_data_dict = {}
        temp_common_dates = None
        successful_codes = []
        try:
            start_dt = pd.to_datetime(start_date, format='%Y%m%d')
            buffer_days = 30
            required_start_dt = start_dt - pd.Timedelta(days=(self.window_size + buffer_days) * 1.5)
            required_start_date_str = required_start_dt.strftime('%Y%m%d')
            # print(f"  為滿足窗口需求，實際請求數據起始日期: {required_start_date_str}")
        except ValueError:
             print("錯誤：起始日期格式無效。")
             return False

        for code in self.stock_codes_initial:
            df = self._load_and_preprocess_single_stock(code, required_start_date_str, end_date)
            if df is not None and not df.empty:
                df_filtered = df[df.index >= start_dt]
                if not df_filtered.empty:
                    temp_data_dict[code] = df_filtered
                    successful_codes.append(code)
                    if temp_common_dates is None: temp_common_dates = df_filtered.index
                    else: temp_common_dates = temp_common_dates.intersection(df_filtered.index)
                # else: print(f"警告: 股票 {code} 在評估期間 ({start_date}之後) 沒有有效數據。")
            # else: print(f"警告: 股票 {code} 數據載入或處理失敗。") # Reduced verbosity

        if not temp_data_dict:
            print("DataManager 錯誤：沒有任何股票數據成功載入。")
            return False
        self.stock_codes = successful_codes
        self.data_dict = temp_data_dict
        if temp_common_dates is None or len(temp_common_dates) == 0:
             print("DataManager 錯誤：找不到所有股票的共同交易日期。")
             return False
        self.common_dates = temp_common_dates.sort_values()
        if len(self.common_dates) < self.window_size + 1:
            print(f"DataManager 錯誤：共同交易日數據量不足 (需要 {self.window_size + 1}, 實際 {len(self.common_dates)})")
            return False
        print(f"DataManager: 數據載入完成，共 {len(self.stock_codes)} 支股票，找到 {len(self.common_dates)} 個共同交易日。")
        return True

    def get_common_dates(self): return self.common_dates
    def get_stock_codes(self): return self.stock_codes
    def get_data_on_date(self, stock_code, date):
        if stock_code in self.data_dict and date in self.data_dict[stock_code].index:
            return self.data_dict[stock_code].loc[date]
        else: return None
    def get_indicator_periods(self):
        return {'ma_long': self.ma_long_period, 'rsi': self.rsi_period, 'atr': self.atr_period}


class PortfolioManager:
    """管理投資組合的狀態：現金、持股、成本、價值。"""
    def __init__(self, initial_capital, stock_codes):
        self.initial_capital = initial_capital
        self.stock_codes = list(stock_codes)
        self.cash = initial_capital
        self.shares_held = defaultdict(int)
        self.entry_price = defaultdict(float)
        self.entry_atr = defaultdict(float)
        self.portfolio_value = initial_capital

    def reset(self):
        self.cash = self.initial_capital
        self.shares_held = defaultdict(int)
        self.entry_price = defaultdict(float)
        self.entry_atr = defaultdict(float)
        self.portfolio_value = self.initial_capital
        print("PortfolioManager: 狀態已重設。")

    def update_on_buy(self, stock_code, shares_bought, cost, entry_atr):
        if stock_code not in self.stock_codes or shares_bought <= 0: return
        self.cash -= cost
        self.entry_price[stock_code] = cost / shares_bought if shares_bought > 0 else 0
        self.entry_atr[stock_code] = entry_atr
        self.shares_held[stock_code] += shares_bought
        # sheets = shares_bought / 1000
        # print(f"PortfolioManager: 更新買入 {stock_code} ({sheets:.0f}張), Cost={cost:.2f}")

    def update_on_sell(self, stock_code, shares_sold, proceeds):
        if stock_code not in self.stock_codes or shares_sold <= 0: return
        self.cash += proceeds
        self.shares_held[stock_code] -= shares_sold
        if self.shares_held[stock_code] <= 0:
            self.shares_held[stock_code] = 0
            self.entry_price[stock_code] = 0.0
            self.entry_atr[stock_code] = 0.0
        # sheets = shares_sold / 1000
        # print(f"PortfolioManager: 更新賣出 {stock_code} ({sheets:.0f}張), Proceeds={proceeds:.2f}")

    def calculate_and_update_portfolio_value(self, data_manager: DataManager, current_date):
        total_stock_value = 0.0
        missing_data_codes = []
        common_dates_list = data_manager.get_common_dates()

        for code in self.stock_codes:
            shares = self.shares_held[code]
            if shares > 0:
                data = data_manager.get_data_on_date(code, current_date)
                if data is not None and pd.notna(data['close']) and data['close'] > 0:
                    total_stock_value += shares * data['close']
                else:
                    print(f"PortfolioManager 警告: 計算價值時 {code} 在 {current_date.strftime('%Y-%m-%d')} 缺少有效收盤價。")
                    missing_data_codes.append(code)
                    try:
                        current_idx_in_common = common_dates_list.get_loc(current_date)
                        prev_date_idx = current_idx_in_common - 1
                        if prev_date_idx >= 0:
                             prev_date = common_dates_list[prev_date_idx]
                             prev_data = data_manager.get_data_on_date(code, prev_date)
                             if prev_data is not None and pd.notna(prev_data['close']) and prev_data['close'] > 0:
                                  total_stock_value += shares * prev_data['close']
                                  # print(f"  > 使用前一日 ({prev_date.strftime('%Y-%m-%d')}) 收盤價 {prev_data['close']:.2f} 估算 {code} 價值。")
                             # else: print(f"  > 無法找到前一日有效價格，{code} 價值暫計為 0。")
                        # else: print(f"  > 無法找到前一日有效價格，{code} 價值暫計為 0。")
                    except KeyError: pass
                         # print(f"  > 日期 {current_date} 不在共同日期列表中，無法找前一日價格，{code} 價值暫計為 0。")
                    except Exception as e: pass
                         # print(f"  > 尋找前一日價格時出錯: {e}")


        self.portfolio_value = self.cash + total_stock_value
        return self.portfolio_value

    def get_cash(self): return self.cash
    def get_shares(self, stock_code): return self.shares_held[stock_code]
    def get_portfolio_value(self): return self.portfolio_value
    def get_entry_price(self, stock_code): return self.entry_price[stock_code]
    def get_entry_atr(self, stock_code): return self.entry_atr[stock_code]

class TradeExecutor:
    """負責應用資金管理規則和提交 API 訂單。"""
    def __init__(self, api: Stock_API, portfolio_manager: PortfolioManager, data_manager: DataManager):
        self.api = api
        self.portfolio_manager = portfolio_manager
        self.data_manager = data_manager

    def place_sell_orders(self, sell_requests):
        api_success_map = {}
        if not sell_requests: return api_success_map
        print("TradeExecutor: 提交賣單請求至 API...")
        for code, shares_api, price in sell_requests:
            api_success_map[code] = self.api.Sell_Stock(code, shares_api, price)
            time.sleep(0.1)
        return api_success_map

    def determine_and_place_buy_orders(self, buy_requests, date_T):
        api_success_map = {}
        orders_to_submit_buy = []
        current_total_value = self.portfolio_manager.get_portfolio_value()
        available_cash = self.portfolio_manager.get_cash()

        if current_total_value <= 0:
            print("TradeExecutor: 組合價值非正數，無法計算資本限制。跳過買入。")
            return api_success_map

        max_capital_per_stock = current_total_value / 20.0
        max_capital_per_trade = current_total_value / 100.0
        print(f"TradeExecutor: 處理買單... (可用現金: {available_cash:.2f})")
        print(f"  資本限制: 單股上限={max_capital_per_stock:.2f}, 單筆交易上限={max_capital_per_trade:.2f}")
        buy_requests.sort(key=lambda x: x[0])

        for code, price_T in buy_requests:
            if price_T <= 0: continue
            cost_per_sheet = 1000 * price_T
            if cost_per_sheet <= 0: continue

            max_sheets_trade = math.floor(max_capital_per_trade / cost_per_sheet) if cost_per_sheet > 0 else 0
            if max_sheets_trade <= 0: continue

            target_sheets = max_sheets_trade
            required_cash_for_target = target_sheets * cost_per_sheet

            if available_cash < required_cash_for_target:
                sheets_can_afford = math.floor(available_cash / cost_per_sheet) if cost_per_sheet > 0 else 0
                if sheets_can_afford <= 0: continue
                else: target_sheets = sheets_can_afford

            current_shares = self.portfolio_manager.get_shares(code)
            current_stock_value = current_shares * price_T
            potential_cost = target_sheets * cost_per_sheet
            potential_new_value = current_stock_value + potential_cost

            if potential_new_value > max_capital_per_stock:
                allowed_additional_capital = max_capital_per_stock - current_stock_value
                if allowed_additional_capital > 0:
                     allowed_additional_sheets = math.floor(allowed_additional_capital / cost_per_sheet) if cost_per_sheet > 0 else 0
                     final_sheets_to_buy = min(target_sheets, allowed_additional_sheets)
                     if final_sheets_to_buy > 0:
                          target_sheets = final_sheets_to_buy
                          potential_cost = target_sheets * cost_per_sheet
                     else: continue
                else: continue

            if target_sheets <= 0: continue

            # Final check if cash is sufficient for the potentially reduced target_sheets
            final_cost = target_sheets * cost_per_sheet
            if available_cash < final_cost:
                # This might happen if the stock limit drastically reduced the sheets
                print(f"  > {code}: 最終現金檢查不足以購買 {target_sheets} 張。")
                continue

            shares_to_buy_api = target_sheets * 1000
            orders_to_submit_buy.append((code, shares_to_buy_api, price_T))
            available_cash -= final_cost # Use final calculated cost
            print(f"  > {code}: 計劃買入 {target_sheets} 張 ({shares_to_buy_api} 股)。(剩餘現金估算: {available_cash:.2f})")

        print("TradeExecutor: 提交買單請求至 API...")
        for code, shares_api, price in orders_to_submit_buy:
            api_success_map[code] = self.api.Buy_Stock(code, shares_api, price)
            time.sleep(0.1)
        return api_success_map

class SimulationEngine:
    """主回測引擎，協調各組件運行日循環。"""
    def __init__(self, start_date, end_date, data_manager: DataManager,
                 portfolio_manager: PortfolioManager, trade_executor: TradeExecutor,
                 models: dict, # {stock_code: model}
                 sl_multiplier=2.0, tp_multiplier=3.0): # These are for observation calculation
        self.start_date_str = start_date
        self.end_date_str = end_date
        self.data_manager = data_manager
        self.portfolio_manager = portfolio_manager
        self.trade_executor = trade_executor
        self.models = models
        self.stock_codes = data_manager.get_stock_codes() # Use DataManager's list
        indicator_params = data_manager.get_indicator_periods()
        self.ma_long_period = indicator_params['ma_long']
        self.rsi_period = indicator_params['rsi']
        self.atr_period = indicator_params['atr']
        self.sl_multiplier = sl_multiplier
        self.tp_multiplier = tp_multiplier
        self.features_per_stock = 7
        self.portfolio_history = []
        self.dates_history = []

    def _get_single_stock_observation(self, stock_code, date_idx):
        common_dates = self.data_manager.get_common_dates()
        if date_idx < 0 or date_idx >= len(common_dates): return np.zeros(self.features_per_stock, dtype=np.float32)
        current_date = common_dates[date_idx]
        obs_data = self.data_manager.get_data_on_date(stock_code, current_date)
        if obs_data is None: return np.zeros(self.features_per_stock, dtype=np.float32)

        close_price = obs_data['close']
        atr_val = obs_data.get(f'ATR_{self.atr_period}', 0.0)
        atr_norm_val = obs_data.get(f'ATR_norm_{self.atr_period}', 0.0)
        ma_long_val = obs_data.get(f'SMA_{self.ma_long_period}', close_price)
        rsi_val = obs_data.get(f'RSI_{self.rsi_period}', 50.0) / 100.0
        holding_position = 1.0 if self.portfolio_manager.get_shares(stock_code) > 0 else 0.0
        potential_sl, potential_tp, distance_to_sl_norm, distance_to_tp_norm, is_below_potential_sl = 0.0, 0.0, 0.0, 0.0, 0.0
        entry_p = self.portfolio_manager.get_entry_price(stock_code)
        entry_a = self.portfolio_manager.get_entry_atr(stock_code)
        if holding_position > 0 and entry_p > 0 and entry_a > 0:
            potential_sl = entry_p - self.sl_multiplier * entry_a
            potential_tp = entry_p + self.tp_multiplier * entry_a
            if close_price > 0:
                distance_to_sl_norm = (close_price - potential_sl) / close_price
                distance_to_tp_norm = (potential_tp - close_price) / close_price
            if close_price < potential_sl: is_below_potential_sl = 1.0

        stock_features = np.array([price_ma_ratio, rsi_val, atr_norm_val, holding_position,
                                   distance_to_sl_norm, distance_to_tp_norm, is_below_potential_sl], dtype=np.float32)
        stock_features = np.nan_to_num(stock_features, nan=0.0, posinf=1e9, neginf=-1e9)
        return stock_features

    def run_backtest(self):
        if not self.data_manager.load_all_data(self.start_date_str, self.end_date_str):
            print("SimulationEngine: 數據初始化失敗，無法開始回測。")
            return
        self.portfolio_manager.reset()
        common_dates = self.data_manager.get_common_dates()
        window_size = self.data_manager.window_size
        start_idx = window_size
        end_idx = len(common_dates) - 1
        self.portfolio_history = [self.portfolio_manager.initial_capital] * start_idx
        self.dates_history = list(common_dates[:start_idx])
        print(f"\n--- SimulationEngine: 開始回測 ({common_dates[start_idx].strftime('%Y-%m-%d')} to {common_dates[end_idx].strftime('%Y-%m-%d')}) ---")

        for current_idx in range(start_idx, end_idx):
            date_T = common_dates[current_idx]
            date_T1 = common_dates[current_idx + 1]
            print(f"\n====== Day T: {date_T.strftime('%Y-%m-%d')} (收盤後決策) ======")
            sell_requests_plan, buy_requests_plan = [], []
            for code in self.stock_codes:
                if code in self.models:
                    obs = self._get_single_stock_observation(code, current_idx)
                    action, _ = self.models[code].predict(obs, deterministic=True)
                    data_T = self.data_manager.get_data_on_date(code, date_T)
                    price_T = data_T['close'] if data_T is not None and pd.notna(data_T['close']) else 0

                    if action == 2 and self.portfolio_manager.get_shares(code) > 0:
                        sheets_to_sell = math.floor(self.portfolio_manager.get_shares(code) / 1000)
                        if sheets_to_sell > 0: sell_requests_plan.append((code, sheets_to_sell * 1000, price_T))
                    elif action == 1 and self.portfolio_manager.get_shares(code) == 0 and price_T > 0:
                        buy_requests_plan.append((code, price_T))

            # print("--- 調用 TradeExecutor ---")
            api_sell_success = self.trade_executor.place_sell_orders(sell_requests_plan)
            api_buy_success = self.trade_executor.determine_and_place_buy_orders(buy_requests_plan, date_T)
            # print("--- TradeExecutor 調用完成 ---")

            print(f"\n====== Day T+1: {date_T1.strftime('%Y-%m-%d')} (盤後結算) ======")
            # print("SimulationEngine: 查詢實際持股...")
            time.sleep(0.1)
            actual_holdings_list_T1 = self.trade_executor.api.Get_User_Stocks()
            actual_holdings_map_T1 = {h['stock_code']: h['shares'] for h in actual_holdings_list_T1}
            previous_holdings_state_T1 = {code: self.portfolio_manager.get_shares(code) for code in self.stock_codes}
            executed_trades_info = []

            for code in self.stock_codes:
                 previous_shares = previous_holdings_state_T1.get(code, 0)
                 actual_shares = actual_holdings_map_T1.get(code, 0)
                 data_T1 = self.data_manager.get_data_on_date(code, date_T1)
                 price_T1_open, atr_T1 = 0.0, 0.0
                 if data_T1 is not None:
                      price_T1_open = data_T1['open'] if pd.notna(data_T1['open']) else 0
                      atr_T1 = data_T1.get(f'ATR_{self.atr_period}', 0.0)
                 else: actual_shares = previous_shares

                 shares_changed = actual_shares - previous_shares
                 if shares_changed > 0:
                      if price_T1_open > 0:
                           cost = shares_changed * price_T1_open
                           self.portfolio_manager.update_on_buy(code, shares_changed, cost, atr_T1)
                           sheets = shares_changed / 1000; executed_trades_info.append(f"{code}:BUY_{sheets:.0f}張")
                      else:
                           print(f"警告: {code} 在 {date_T1} 的開盤價無效，無法更新買入成本。持股已根據API更新。")
                           self.portfolio_manager.shares_held[code] = actual_shares
                 elif shares_changed < 0:
                      if price_T1_open > 0:
                           proceeds = -shares_changed * price_T1_open
                           self.portfolio_manager.update_on_sell(code, -shares_changed, proceeds)
                           sheets = -shares_changed / 1000; executed_trades_info.append(f"{code}:SELL_{sheets:.0f}張")
                      else:
                           print(f"警告: {code} 在 {date_T1} 的開盤價無效，無法更新賣出收益。持股已根據API更新。")
                           self.portfolio_manager.shares_held[code] = actual_shares
                           if self.portfolio_manager.shares_held[code] == 0:
                                self.portfolio_manager.entry_price[code] = 0.0
                                self.portfolio_manager.entry_atr[code] = 0.0

            current_value = self.portfolio_manager.calculate_and_update_portfolio_value(self.data_manager, date_T1)
            self.portfolio_history.append(current_value)
            self.dates_history.append(date_T1)
            print(f"本日結算後組合價值: {current_value:.2f}")
            if executed_trades_info: print(f"  本日成交: {', '.join(executed_trades_info)}")

        self.report_results()
        self.plot_performance()

    def report_results(self):
        final_portfolio_value = self.portfolio_manager.get_portfolio_value()
        initial_capital = self.portfolio_manager.initial_capital
        total_return_pct = ((final_portfolio_value - initial_capital) / initial_capital) * 100 if initial_capital else 0
        print("\n--- SimulationEngine: 最終回測結果 ---")
        print(f"評估期間: {self.start_date_str} to {self.end_date_str}")
        print(f"初始資金: {initial_capital:.2f}")
        print(f"最終組合價值: {final_portfolio_value:.2f}")
        print(f"總回報率: {total_return_pct:.2f}%")

    def plot_performance(self):
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
            plt.figure(figsize=(14, 7))
            if len(self.dates_history) == len(self.portfolio_history) and len(self.dates_history) > 0 :
                portfolio_series = pd.Series(self.portfolio_history, index=self.dates_history)
                plt.plot(portfolio_series.index, portfolio_series.values, label='Portfolio Value', linewidth=1.5)
                plt.title(f"Portfolio Value Over Time ({self.start_date_str} to {self.end_date_str}) - Refactored")
                plt.xlabel("Date"); plt.ylabel("Portfolio Value (TWD)"); plt.legend(); plt.grid(True); plt.tight_layout()
                import matplotlib.ticker as mtick
                ax = plt.gca()
                ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
                plt.xticks(rotation=45)
                plt.savefig("portfolio_curve_refactored.png", dpi=300)
                print("投資組合價值曲線圖已保存為 portfolio_curve_refactored.png")
            else: print(f"警告: 日期歷史({len(self.dates_history)})與價值歷史({len(self.portfolio_history)})長度不匹配或為空，無法繪圖。")
        except ImportError: print("未安裝 matplotlib。請運行 'pip install matplotlib'")
        except Exception as e: print(f"繪製圖表時出錯: {e}")


# --- 主程序入口 ---
if __name__ == '__main__':
     # --- !!! 重要：請替換為您的 API 憑證 !!! ---
     API_ACCOUNT = "YOUR_API_ACCOUNT"
     API_PASSWORD = "YOUR_API_PASSWORD"

     # --- 股票列表 ---
     STOCK_CODES_LIST = ['2330','2454','2317','2308','2881','2891','2382','2303','2882','2412',
                    '2886','3711','2884','2357','1216','2885','3034','3231','2892','2345']

     # --- 評估參數 ---
     EVAL_START_DATE = '20200419'
     EVAL_END_DATE = '20250417'
     TOTAL_INITIAL_CAPITAL = 100,000,000.0 # 總初始資金
     MODELS_LOAD_DIR = "trained_individual_models" # 獨立模型儲存目錄

     # --- 指標和窗口參數 (應與訓練時一致) ---
     MA_LONG = 50
     RSI_PERIOD = 14
     ATR_PERIOD = 14
     SL_ATR_MULT = 2.0 # 用於觀察值計算
     TP_ATR_MULT = 3.0 # 用於觀察值計算
     WINDOW_SIZE = MA_LONG + 10 # 與訓練時保持一致

     # --- Phase Selection (Only run evaluation here) ---
     RUN_TRAINING = False
     RUN_EVALUATION = True

     if RUN_TRAINING:
         print("錯誤：此腳本僅用於評估。請運行 train_models.py 進行訓練。")

     if RUN_EVALUATION:
        print("\n=============== 開始模組化評估階段 ===============")
        if not os.path.exists(MODELS_LOAD_DIR):
             print(f"錯誤：找不到模型目錄 '{MODELS_LOAD_DIR}'。請先運行訓練階段。")
        else:
             # --- 創建組件 ---
             print("--- 初始化評估組件 ---")
             api_eval = Stock_API(API_ACCOUNT, API_PASSWORD)
             data_manager_eval = DataManager(
                 stock_codes_initial=STOCK_CODES_LIST, api=api_eval, window_size=WINDOW_SIZE,
                 ma_long=MA_LONG, rsi_period=RSI_PERIOD, atr_period=ATR_PERIOD
             )

             # 必須先成功加載數據才能繼續
             if data_manager_eval.load_all_data(EVAL_START_DATE, EVAL_END_DATE):
                 portfolio_manager_eval = PortfolioManager(
                     initial_capital=TOTAL_INITIAL_CAPITAL,
                     stock_codes=data_manager_eval.get_stock_codes() # 使用 DataManager 提供的列表
                 )
                 trade_executor_eval = TradeExecutor(
                     api=api_eval,
                     portfolio_manager=portfolio_manager_eval,
                     data_manager=data_manager_eval
                 )

                 # --- 加載模型 ---
                 models_eval = {}
                 print("--- 加載預訓練模型 ---")
                 loaded_codes_final = []
                 for code in data_manager_eval.get_stock_codes(): # 只加載有數據的股票的模型
                     model_path = os.path.join(MODELS_LOAD_DIR, f"ppo_agent_{code}")
                     if os.path.exists(model_path + ".zip"):
                         try:
                             models_eval[code] = PPO.load(model_path)
                             print(f"  > 已加載模型: {code}")
                             loaded_codes_final.append(code)
                         except Exception as e: print(f"加載模型 {code} 失敗: {e}")
                     else: print(f"警告: 找不到模型文件 {model_path}.zip for stock {code}")

                 if not models_eval:
                     print("錯誤：沒有成功加載任何用於評估的模型。")
                 else:
                     # 更新組件的股票列表以防模型加載失敗
                     # (確保 SimulationEngine 使用的列表與實際加載的模型一致)
                     final_stock_codes_for_sim = loaded_codes_final
                     # Important: Update lists in existing instances
                     data_manager_eval.stock_codes = final_stock_codes_for_sim
                     portfolio_manager_eval.stock_codes = final_stock_codes_for_sim

                     # --- 創建並運行引擎 ---
                     simulation_engine = SimulationEngine(
                         start_date=EVAL_START_DATE, end_date=EVAL_END_DATE,
                         data_manager=data_manager_eval,
                         portfolio_manager=portfolio_manager_eval,
                         trade_executor=trade_executor_eval,
                         models=models_eval, # 傳入加載好的模型
                         sl_multiplier=SL_ATR_MULT, # 傳入一致的參數
                         tp_multiplier=TP_ATR_MULT
                     )
                     simulation_engine.run_backtest()
             else:
                  print("數據管理器初始化失敗，無法進行評估。")

        print("\n=============== 模組化評估階段完成 ===============")

     if not RUN_TRAINING and not RUN_EVALUATION:
        print("\n請設置 RUN_TRAINING=True (在 train_models.py 中) 或 RUN_EVALUATION=True (在此腳本中) 來執行。")

     print("\n--- 程序執行完畢 ---")