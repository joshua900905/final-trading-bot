# -*- coding: utf-8 -*-
# evaluate_base_strategy.py - Evaluate Base Strategy (Fixed Entry/Stop, TEMA Exit)

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

# --- Refactored Evaluation Classes (Simplified) ---

class DataManager:
    """數據管理器 (計算 TEMA 和 ATR)"""
    # (與之前的評估版本一致)
    def __init__(self, stock_codes_initial, api, window_size,
                 tema_short=9, tema_long=18, atr_period=14):
        self.stock_codes_initial = list(stock_codes_initial); self.api = api;
        self.tema_short_period = tema_short; self.tema_long_period = tema_long
        self.atr_period = atr_period
        self.atr_col_name = f'ATR_{self.atr_period}'
        self.window_size = window_size
        self.data_dict = {}; self.common_dates = None; self.stock_codes = []

    def _load_and_preprocess_single_stock(self, stock_code, start_date, end_date):
        raw_data = self.api.Get_Stock_Informations(stock_code, start_date, end_date);
        if not raw_data: return None
        try:
            df = pd.DataFrame(raw_data); df['date'] = pd.to_datetime(df['date'], unit='s'); df = df.sort_values('date'); df = df[~df['date'].duplicated(keep='first')]; df = df.set_index('date')
            required_cols = ['open', 'high', 'low', 'close', 'turnover'];
            if not all(col in df.columns for col in required_cols): return None
            numeric_cols = ['open', 'high', 'low', 'close', 'turnover', 'capacity', 'transaction_volume']
            for col in numeric_cols:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
            indicator_base_cols = ['open', 'high', 'low', 'close']; df = df.dropna(subset=indicator_base_cols)
            if df.empty: return None
            tema9_col = f'TEMA_{self.tema_short_period}'; tema18_col = f'TEMA_{self.tema_long_period}'
            df.ta.tema(length=self.tema_short_period, close='close', append=True, col_names=(tema9_col,))
            df.ta.tema(length=self.tema_long_period, close='close', append=True, col_names=(tema18_col,))
            # 不需要 Slope
            # df[f'{tema9_col}_slope'] = df[tema9_col].pct_change().fillna(0).replace([np.inf, -np.inf], 0)
            # df[f'{tema18_col}_slope'] = df[tema18_col].pct_change().fillna(0).replace([np.inf, -np.inf], 0)
            df.ta.atr(length=self.atr_period, append=True, col_names=(self.atr_col_name,))
            df = df.dropna();
            if df.empty: return None
            return df
        except Exception as e: print(f"DataManager: 處理 {stock_code} 數據時出錯: {e}"); traceback.print_exc(); return None

    def load_all_data(self, start_date, end_date):
        # (與之前版本一致)
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
        if len(self.common_dates) < self.window_size + 1: print(f"DataManager 錯誤：共同交易日數據量不足"); return False
        print(f"DataManager: 數據載入完成，找到 {len(self.common_dates)} 個共同交易日。")
        return True

    # Getters (保持不變)
    def get_common_dates(self): return self.common_dates
    def get_stock_codes(self): return self.stock_codes
    def get_data_on_date(self, stock_code, date):
        if stock_code in self.data_dict and date in self.data_dict[stock_code].index:
            data_slice = self.data_dict[stock_code].loc[[date]]; return data_slice.iloc[0] if not data_slice.empty else None
        else: return None
    def get_indicator_periods(self): return {'tema_short': self.tema_short_period, 'tema_long': self.tema_long_period, 'atr': self.atr_period}
    def get_atr_col_name(self): return self.atr_col_name


class PortfolioManager:
    """投資組合管理器 (簡化版，只追蹤基本持倉和止損)"""
    def __init__(self, initial_capital, stock_codes):
        self.initial_capital = initial_capital; self.stock_codes = list(stock_codes);
        if len(self.stock_codes) != 1: print("警告：PortfolioManager 設計為單股票回測。")
        self.target_code = self.stock_codes[0] if self.stock_codes else None
        self.cash = initial_capital
        self.shares_held = defaultdict(int)
        self.entry_price = defaultdict(float)
        self.stop_loss_price = defaultdict(float) # <<< 追蹤止損價格
        self.portfolio_value = initial_capital

    def reset(self):
        self.cash = self.initial_capital; self.shares_held = defaultdict(int)
        self.entry_price = defaultdict(float); self.stop_loss_price = defaultdict(float)
        self.portfolio_value = self.initial_capital
        print(f"PortfolioManager ({self.target_code}): 狀態已重設。")

    def update_on_buy(self, stock_code, shares_bought, cost, stop_loss): # <<< 簡化參數
        if stock_code != self.target_code or shares_bought <= 0: return
        self.cash -= cost
        self.entry_price[stock_code] = cost / shares_bought if shares_bought > 0 else 0
        self.shares_held[stock_code] = shares_bought
        self.stop_loss_price[stock_code] = stop_loss

    def update_on_sell(self, stock_code, shares_sold, proceeds): # <<< 簡化參數
        if stock_code != self.target_code or shares_sold <= 0: return
        if shares_sold > self.shares_held[stock_code]: shares_sold = self.shares_held[stock_code]
        self.cash += proceeds
        self.shares_held[stock_code] -= shares_sold
        # 賣出後直接清空狀態 (因為只有完全賣出)
        if self.shares_held[stock_code] <= 0:
            self.shares_held[stock_code] = 0
            self.entry_price[stock_code] = 0.0
            self.stop_loss_price[stock_code] = 0.0

    def calculate_and_update_portfolio_value(self, data_manager: DataManager, current_date):
        # (與之前版本一致)
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

    # Getters (移除減倉相關)
    def get_cash(self): return self.cash
    def get_shares(self, stock_code): return self.shares_held.get(stock_code, 0)
    def get_portfolio_value(self): return self.portfolio_value
    def get_entry_price(self, stock_code): return self.entry_price.get(stock_code, 0.0)
    def get_stop_loss_price(self, stock_code): return self.stop_loss_price.get(stock_code, 0.0)


class TradeExecutor:
    """交易執行器 (只處理買單和完全賣單)"""
    def __init__(self, api: Stock_API, portfolio_manager: PortfolioManager, data_manager: DataManager,
                 initial_position_ratio=0.2, stop_loss_atr_multiplier=2.0):
        self.api = api; self.portfolio_manager = portfolio_manager; self.data_manager = data_manager
        self.initial_position_ratio = initial_position_ratio
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.shares_per_level = 1000

    def place_sell_orders(self, sell_requests): # sell_requests: (code, shares, price, sell_type)
        # (與之前版本一致)
        simulated_success_map = {};
        if not sell_requests: return simulated_success_map
        print("TradeExecutor: [BACKTEST] 模擬提交賣單...")
        for code, shares_api, price, sell_type in sell_requests:
            sheets = shares_api / self.shares_per_level
            print(f"  [BACKTEST] 模擬提交賣單 ({sell_type}): {sheets:.0f} 張 {code}")
            simulated_success_map[(code, sell_type)] = True
        return simulated_success_map

    def determine_and_place_buy_orders(self, buy_request: dict, date_T):
        # (與之前版本一致)
        simulated_success_map = {}; orders_to_submit_buy = []
        if not buy_request: return orders_to_submit_buy, simulated_success_map
        current_total_value = self.portfolio_manager.get_portfolio_value()
        available_cash = self.portfolio_manager.get_cash()
        if not isinstance(current_total_value, (int, float)) or current_total_value <= 0: return orders_to_submit_buy, simulated_success_map

        print(f"TradeExecutor: [BACKTEST] 處理固定比例買入請求... (可用現金: {available_cash:.2f})")
        for code, entry_signal_flag in buy_request.items():
             if code != self.portfolio_manager.target_code or not entry_signal_flag: continue
             data_T = self.data_manager.get_data_on_date(code, date_T)
             price_T = data_T['close'] if data_T is not None and pd.notna(data_T['close']) else 0
             atr_T = data_T.get(self.data_manager.get_atr_col_name(), 0.0) if data_T is not None else 0.0
             if price_T <= 0: print(f"  > {code}: 無效價格，無法買入。"); continue
             stop_loss_price_T = price_T - self.stop_loss_atr_multiplier * atr_T if atr_T > 0 else price_T * 0.95
             if stop_loss_price_T >= price_T: print(f"  > {code}: 止損價 ({stop_loss_price_T:.2f}) >= 進場價 ({price_T:.2f})，跳過。"); continue

             print(f"  > {code}: 使用固定初始倉位 {self.initial_position_ratio*100:.0f}%, 止損價: {stop_loss_price_T:.2f}")
             target_capital = current_total_value * self.initial_position_ratio
             cost_per_sheet = self.shares_per_level * price_T
             if cost_per_sheet <= 0: continue
             target_sheets = math.floor(target_capital / cost_per_sheet) if cost_per_sheet > 0 else 0
             if target_sheets <= 0: print(f"  > {code}: 計算目標張數為 0。"); continue
             final_cost = target_sheets * cost_per_sheet
             if available_cash < final_cost:
                 target_sheets = math.floor(available_cash / cost_per_sheet) if cost_per_sheet > 0 else 0
                 if target_sheets <= 0: print(f"    > 現金不足以購買任何張。"); continue
                 else: final_cost = target_sheets * cost_per_sheet; print(f"    > 現金不足，調整為購買 {target_sheets} 張。")

             if target_sheets > 0:
                 shares_to_buy_api = target_sheets * self.shares_per_level
                 orders_to_submit_buy.append((code, shares_to_buy_api, price_T, stop_loss_price_T)) # <<< 包含止損價

        print("TradeExecutor: [BACKTEST] 模擬提交買單...")
        for code, shares_api, price, sl_price in orders_to_submit_buy:
            sheets = shares_api / self.shares_per_level
            print(f"  [BACKTEST] 模擬提交買單: {sheets:.0f} 張 {code} (SL: {sl_price:.2f})")
            simulated_success_map[code] = True
        return orders_to_submit_buy, simulated_success_map


class SimulationEngine:
    """回測引擎 (基礎策略：固定進場/止損, TEMA出場)"""
    def __init__(self, start_date, end_date, data_manager: DataManager,
                 portfolio_manager: PortfolioManager, trade_executor: TradeExecutor): # <<< 移除 AI 相關參數
        self.start_date_str = start_date; self.end_date_str = end_date
        self.data_manager = data_manager; self.portfolio_manager = portfolio_manager
        self.trade_executor = trade_executor # <<<
        self.target_stock_code = self.portfolio_manager.target_code
        if not self.target_stock_code: raise ValueError("目標股票代碼無效。")

        self.stock_codes = [self.target_stock_code]
        indicator_params = data_manager.get_indicator_periods()
        self.tema_short_period = indicator_params['tema_short']
        self.tema_long_period = indicator_params['tema_long']
        self.atr_period = indicator_params['atr'] # 雖然不用於觀察，但需要檢查止損
        self.atr_col_name = data_manager.get_atr_col_name()

        self.portfolio_history = []; self.dates_history = []

    # 不需要 _get_single_stock_observation

    def _check_entry_signal(self, current_step_idx):
            if current_step_idx < 1:
                return False
            # <<< 將日期定義移出 if 判斷 >>>
            common_dates = self.data_manager.get_common_dates()
            # 檢查索引是否有效
            if current_step_idx >= len(common_dates) or current_step_idx - 1 < 0:
                print(f"警告: _check_entry_signal 索引無效 {current_step_idx}")
                return False
            current_date = common_dates[current_step_idx]
            yesterday_date = common_dates[current_step_idx - 1]

            today_data = self.data_manager.get_data_on_date(self.target_stock_code, current_date)
            yesterday_data = self.data_manager.get_data_on_date(self.target_stock_code, yesterday_date)
            if today_data is None or yesterday_data is None: return False
            tema9_today=today_data.get(f'TEMA_{self.tema_short_period}',np.nan); tema18_today=today_data.get(f'TEMA_{self.tema_long_period}',np.nan)
            tema9_yesterday=yesterday_data.get(f'TEMA_{self.tema_short_period}',np.nan); tema18_yesterday=yesterday_data.get(f'TEMA_{self.tema_long_period}',np.nan)
            if pd.isna(tema9_today) or pd.isna(tema18_today) or pd.isna(tema9_yesterday) or pd.isna(tema18_yesterday): return False
            crossed_up = tema9_yesterday <= tema18_yesterday and tema9_today > tema18_today
            return crossed_up

    def _check_exit_signal(self, current_step_idx):
        if current_step_idx < 1:
            return False
        # <<< 將日期定義移出 if 判斷 >>>
        common_dates = self.data_manager.get_common_dates()
        # 檢查索引是否有效
        if current_step_idx >= len(common_dates) or current_step_idx - 1 < 0:
             print(f"警告: _check_exit_signal 索引無效 {current_step_idx}")
             return False
        current_date = common_dates[current_step_idx]
        yesterday_date = common_dates[current_step_idx - 1]

        today_data = self.data_manager.get_data_on_date(self.target_stock_code, current_date)
        yesterday_data = self.data_manager.get_data_on_date(self.target_stock_code, yesterday_date)
        if today_data is None or yesterday_data is None: return False
        tema9_today=today_data.get(f'TEMA_{self.tema_short_period}',np.nan); tema18_today=today_data.get(f'TEMA_{self.tema_long_period}',np.nan)
        tema9_yesterday=yesterday_data.get(f'TEMA_{self.tema_short_period}',np.nan); tema18_yesterday=yesterday_data.get(f'TEMA_{self.tema_long_period}',np.nan)
        if pd.isna(tema9_today) or pd.isna(tema18_today) or pd.isna(tema9_yesterday) or pd.isna(tema18_yesterday): return False
        crossed_down = tema9_yesterday >= tema18_yesterday and tema9_today < tema18_today
        return crossed_down

    def run_backtest(self):
        """回測主循環 (基礎策略)"""
        if not self.data_manager.load_all_data(self.start_date_str, self.end_date_str): return
        if self.target_stock_code not in self.data_manager.get_stock_codes(): return

        self.stock_codes = [self.target_stock_code]; self.portfolio_manager.stock_codes = self.stock_codes
        self.portfolio_manager.reset(); common_dates = self.data_manager.get_common_dates()
        window_size = self.data_manager.window_size; start_idx = window_size; end_idx = len(common_dates) - 1
        self.portfolio_history = [self.portfolio_manager.initial_capital] * start_idx
        self.dates_history = list(common_dates[:start_idx])

        print(f"\n--- SimulationEngine: 開始回測 ({self.target_stock_code}, {common_dates[start_idx].strftime('%Y-%m-%d')} to {common_dates[end_idx].strftime('%Y-%m-%d')}) ---")
        print(f"    策略: 固定進場/止損, TEMA出場 (無 AI)") # <<< 更新策略描述

        # --- 主回測循環 ---
        for current_idx in range(start_idx, end_idx):
            date_T = common_dates[current_idx]; date_T1 = common_dates[current_idx + 1]
            print(f"\n====== Day T: {date_T.strftime('%Y-%m-%d')} (收盤後決策) ======")

            code = self.target_stock_code; buy_requests_plan = {}; sell_requests_plan = []
            current_shares = self.portfolio_manager.get_shares(code)
            stop_loss_price = self.portfolio_manager.get_stop_loss_price(code)

            # --- 決策階段 (規則化) ---
            data_T = self.data_manager.get_data_on_date(code, date_T)
            if data_T is None: print(f"警告: 無法獲取數據，跳過。"); continue
            low_price_T = data_T['low'] if pd.notna(data_T['low']) else np.inf
            close_price_T = data_T['close'] if pd.notna(data_T['close']) else np.nan

            stop_loss_triggered = (current_shares > 0 and low_price_T <= stop_loss_price)
            tema_exit_signal = self._check_exit_signal(current_idx)
            entry_signal = self._check_entry_signal(current_idx)

            if current_shares > 0: # 持倉
                if stop_loss_triggered:
                    shares_to_sell = current_shares
                    sell_requests_plan.append((code, shares_to_sell, close_price_T, 'stop_loss'))
                    print(f"  > 觸發止損 @ {stop_loss_price:.2f} (Low: {low_price_T:.2f})，計劃賣出 {shares_to_sell / 1000:.0f} 張")
                elif tema_exit_signal:
                    shares_to_sell = current_shares
                    sell_requests_plan.append((code, shares_to_sell, close_price_T, 'tema_exit'))
                    print(f"  > TEMA 出場信號觸發，計劃賣出 {shares_to_sell / 1000:.0f} 張")
                # else: 持有，不做任何事 (無 AI 減倉)
            else: # 未持倉
                if entry_signal:
                    print(f"  > TEMA 入場信號觸發，計劃使用固定比例買入。")
                    buy_requests_plan[code] = True

            # --- 執行階段 ---
            print("--- 調用 TradeExecutor (模擬) ---")
            api_sell_success = self.trade_executor.place_sell_orders(sell_requests_plan)
            planned_buy_orders, api_buy_success = self.trade_executor.determine_and_place_buy_orders(buy_requests_plan, date_T)
            print("--- TradeExecutor 調用完成 (模擬) ---")

            # --- 結算階段 (T+1 開盤) ---
            print(f"\n====== Day T+1: {date_T1.strftime('%Y-%m-%d')} (開盤結算) ======")
            executed_trades_info = []

            # 模擬賣出
            for code_req, shares_api, price_T, sell_type in sell_requests_plan:
                if code == code_req:
                    data_T1 = self.data_manager.get_data_on_date(code, date_T1)
                    if data_T1 is not None:
                        price_T1_open = data_T1['open'] if pd.notna(data_T1['open']) else 0
                        if price_T1_open > 0:
                            proceeds = shares_api * price_T1_open
                            self.portfolio_manager.update_on_sell(code, shares_api, proceeds) # <<< 調用簡化版
                            sheets = shares_api / 1000
                            executed_trades_info.append(f"{code}:SELL_{sell_type.upper()}_{sheets:.0f}張(Sim)")
                        else: print(f"[BACKTEST] 警告: 賣單 ({sell_type}) 開盤價無效"); self.portfolio_manager.update_on_sell(code, shares_api, 0)

            # 模擬買入
            for code_req, shares_api, price_T, stop_loss_price_req in planned_buy_orders:
                 if code == code_req:
                     data_T1 = self.data_manager.get_data_on_date(code, date_T1)
                     if data_T1 is not None:
                         price_T1_open = data_T1['open'] if pd.notna(data_T1['open']) else 0
                         if price_T1_open > 0:
                             cost = shares_api * price_T1_open
                             if self.portfolio_manager.get_cash() >= cost:
                                 self.portfolio_manager.update_on_buy(code, shares_api, cost, stop_loss_price_req) # <<< 調用簡化版
                                 sheets = shares_api / 1000
                                 executed_trades_info.append(f"{code}:BUY_{sheets:.0f}張_Fixed(SL:{stop_loss_price_req:.2f})(Sim)")
                             else: print(f"[BACKTEST] 警告: 買入現金不足")
                         else: print(f"[BACKTEST] 警告: 買入開盤價無效")

            # 更新組合價值
            current_value = self.portfolio_manager.calculate_and_update_portfolio_value(self.data_manager, date_T1)
            self.portfolio_history.append(current_value); self.dates_history.append(date_T1)
            print(f"本日結算後組合價值: {current_value:.2f}")
            if executed_trades_info: print(f"  本日成交: {', '.join(executed_trades_info)}")

        # 循環結束
        self.report_results(); self.plot_performance()

    def report_results(self): # (與之前版本一致)
        final_portfolio_value=self.portfolio_manager.get_portfolio_value(); initial_capital=self.portfolio_manager.initial_capital
        total_return_pct=((final_portfolio_value-initial_capital)/initial_capital)*100 if initial_capital else 0
        print("\n--- SimulationEngine: 最終回測結果 ---"); print(f"股票: {self.target_stock_code}"); print(f"評估期間: {self.start_date_str} to {self.end_date_str}")
        print(f"初始資金: {initial_capital:.2f}"); print(f"最終組合價值: {final_portfolio_value:.2f}"); print(f"總回報率: {total_return_pct:.2f}%")

    def plot_performance(self): # (更新標題和檔名)
        try:
            plt.style.use('seaborn-v0_8-darkgrid'); plt.figure(figsize=(14, 7))
            if len(self.dates_history) == len(self.portfolio_history) and len(self.dates_history) > 0 :
                portfolio_series = pd.Series(self.portfolio_history, index=self.dates_history)
                title = f"{self.target_stock_code} Backtest ({self.start_date_str} to {self.end_date_str}) - Base Strategy (Fixed Entry/SL, TEMA Exit)" # <<< 更新標題
                filename = f"portfolio_curve_{self.target_stock_code}_backtest_base_strategy.png" # <<< 更新檔名
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
     EVAL_START_DATE = '20230101'; EVAL_END_DATE = '20240331' # 與 AI 版本比較時使用相同區間
     TOTAL_INITIAL_CAPITAL = 50000000.0

     # --- 基礎策略參數 ---
     TEMA_SHORT_EVAL = 9; TEMA_LONG_EVAL = 18; ATR_PERIOD_EVAL = 14
     WINDOW_SIZE_EVAL = TEMA_LONG_EVAL * 3 + 10 # 需要足夠數據計算指標
     INITIAL_POS_RATIO_EVAL = 0.20 # 與 AI 版本一致的初始倉位
     STOP_LOSS_ATR_MULT_EVAL = 2.0 # <<< 關鍵參數，需要測試不同值

     RUN_EVALUATION = True

     if RUN_EVALUATION:
        print(f"\n=============== 開始基礎策略回測 ({TARGET_STOCK_CODE_EVAL}) ===============")
        print(f"    止損參數: {STOP_LOSS_ATR_MULT_EVAL} * ATR({ATR_PERIOD_EVAL})")

        print("--- 初始化評估組件 ---")
        api_eval = Stock_API(API_ACCOUNT, API_PASSWORD)
        data_manager_eval = DataManager(
            stock_codes_initial=[TARGET_STOCK_CODE_EVAL], api=api_eval, window_size=WINDOW_SIZE_EVAL,
            tema_short=TEMA_SHORT_EVAL, tema_long=TEMA_LONG_EVAL, atr_period=ATR_PERIOD_EVAL )

        if data_manager_eval.load_all_data(EVAL_START_DATE, EVAL_END_DATE):
             portfolio_manager_eval = PortfolioManager( # 使用簡化版 PM
                 initial_capital=TOTAL_INITIAL_CAPITAL, stock_codes=data_manager_eval.get_stock_codes())
             trade_executor_eval = TradeExecutor( # 傳遞基礎策略參數
                 api=api_eval, portfolio_manager=portfolio_manager_eval, data_manager=data_manager_eval,
                 initial_position_ratio=INITIAL_POS_RATIO_EVAL, stop_loss_atr_multiplier=STOP_LOSS_ATR_MULT_EVAL)

             # 不需要加載 AI 模型

             simulation_engine = SimulationEngine( # 使用基礎版 SE
                 start_date=EVAL_START_DATE, end_date=EVAL_END_DATE, data_manager=data_manager_eval,
                 portfolio_manager=portfolio_manager_eval, trade_executor=trade_executor_eval)
             simulation_engine.run_backtest()
        else: print("數據管理器初始化失敗。")
        print("\n=============== 基礎策略回測完成 ===============")

     else: print("\n請設置 RUN_EVALUATION=True 來執行回測。")
     print("\n--- 程序執行完畢 ---")