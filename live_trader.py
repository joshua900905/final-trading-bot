# -*- coding: utf-8 -*-
# live_trading_bot_ema5_ai_units_random_entry_price_final.py

import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import time
import os
import math
import random # Needed for random entry price
from collections import defaultdict
import traceback
from datetime import datetime, timedelta
import sys
from typing import List, Dict, Any

# --- Stable Baselines 3 Import ---
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import gymnasium as gym # Needed for dummy env when loading VecNormalize
    from gymnasium import spaces # Needed for dummy env
except ImportError:
    print("錯誤：找不到 stable_baselines3 或 gymnasium。請運行 'pip install stable_baselines3[extra] gymnasium'")
    sys.exit(1)

# --- 台灣 0050 成分股列表 ---
TAIWAN_0050_STOCKS = [ # <<< 請務必使用最新、準確的列表替換
    "2330", "2454", "2317", "2412", "6505", "2881", "2308", "2882", "1303",
    "1301", "2886", "3045", "2891", "2002", "1101", "2382", "5880", "2884",
    "1216", "2207", "2303", "3711", "2892", "1102", "2912", "2885", "2408",
    "2880", "6669", "2379", "1326", "2474", "3008", "2395", "5871", "2887",
    "4904", "2357", "4938", "1402", "2883", "9904", "8046", "2105", "1590",
    "2603", "2609", "2615", "2801", "6415"
]
# 如果有已知數據質量差或無法交易的股票，可以在此移除
# BAD_STOCKS = ["SOME_CODE"]
# TAIWAN_0050_STOCKS = [code for code in TAIWAN_0050_STOCKS if code not in BAD_STOCKS]


# --- 真實交易 API 類 (根據範例調整) ---
class Stock_API:
    def __init__(self, account, password):
        self.account = account; self.password = password
        self.base_url = 'http://140.116.86.242:8081/stock/api/v1'
        print(f"Stock_API 初始化 (Account: ***)") # 不打印密碼

    def Get_Stock_Informations(self, stock_code, start_date, stop_date) -> List[Dict]:
        information_url = (f"{self.base_url}/api_get_stock_info_from_date_json/{stock_code}/{start_date}/{stop_date}")
        max_retries = 3; delay = 2; timeout = 20
        for attempt in range(max_retries):
            try:
                response = requests.get(information_url, timeout=timeout); response.raise_for_status(); result = response.json()
                if result.get('result') == 'success': data = result.get('data', []); return data if isinstance(data, list) else []
                else: print(f"API 錯誤 (GetInfo - {stock_code}): {result.get('status', '未知狀態')}"); return []
            except requests.exceptions.Timeout: print(f"API 超時 ({stock_code}) 重試 {attempt + 1}"); time.sleep(delay)
            except requests.exceptions.RequestException as e: print(f"API 請求錯誤 ({stock_code}): {e}"); return []
            except Exception as e: print(f"Get_Stock_Informations 未知錯誤 ({stock_code}): {e}"); return []
        print(f"API 達到最大重試 ({stock_code})"); return []

    def Get_User_Stocks(self) -> List[Dict[str, Any]]:
        data = {'account': self.account, 'password': self.password}; search_url = f'{self.base_url}/get_user_stocks'
        timeout = 15
        try:
            response = requests.post(search_url, data=data, timeout=timeout); response.raise_for_status(); result = response.json()
            if result.get('result') == 'success':
                holdings_data = result.get('data', [])
                if not isinstance(holdings_data, list): print(f"API 返回持股格式非列表"); return []
                processed_holdings = []
                for stock in holdings_data:
                     code = stock.get('stock_code'); shares_str = stock.get('shares'); price_str = stock.get('price'); amount_str = stock.get('amount')
                     if code is None or shares_str is None: print(f"處理持股數據時發現缺失值 for {code}"); continue # 必要欄位檢查
                     try: processed_holdings.append({'stock_code': str(code), 'shares': int(shares_str),
                                                   'price': float(price_str) if price_str is not None else 0.0, # 處理 None
                                                   'amount': float(amount_str) if amount_str is not None else 0.0}) # 處理 None
                     except (ValueError, TypeError) as e: print(f"處理持股數據類型轉換錯誤 for {code}: {e}")
                return processed_holdings
            else: print(f"API 錯誤 (GetStocks): {result.get('status', '未知狀態')}"); return []
        except requests.exceptions.Timeout: print(f"API 超時 (GetStocks)"); return []
        except requests.exceptions.RequestException as e: print(f"API 請求錯誤 (GetStocks): {e}"); return []
        except Exception as e: print(f"處理用戶持股時出錯 (GetStocks): {e}"); return []

    def Buy_Stock(self, stock_code, stock_sheets, stock_price): # API 接收張數
        stock_sheets = int(stock_sheets);
        if stock_sheets <= 0: print(f"買單張數錯誤 ({stock_sheets})。"); return False

        print(f"\n[LIVE] === 提交買單 ===")
        print(f"  股票: {stock_code}")
        print(f"  數量: {stock_sheets} 張")
        print(f"  目標價: {stock_price:.2f} (限價單)")
        print(f"===================")
        # input("按 Enter 確認提交買單，按 Ctrl+C 取消...") # 正式部署時應移除

        data = {'account': self.account, 'password': self.password, 'stock_code': str(stock_code),
                'stock_shares': stock_sheets, 'stock_price': float(stock_price)} # API key 是 shares 但值是張數
        buy_url = f'{self.base_url}/buy'; timeout = 20
        try:
            response = requests.post(buy_url, data=data, timeout=timeout); response.raise_for_status(); result = response.json()
            print(f"[LIVE] 買單響應: Result={result.get('result', 'N/A')}, Status={result.get('status', 'N/A')}")
            return result.get('result') == 'success'
        except requests.exceptions.Timeout: print(f"[LIVE] 買單提交超時 ({stock_code})"); return False
        except requests.exceptions.RequestException as e: print(f"[LIVE] 買單請求錯誤 ({stock_code}): {e}"); return False
        except Exception as e: print(f"[LIVE] 買單未知錯誤 ({stock_code}): {e}"); return False

    def Sell_Stock(self, stock_code, stock_sheets, stock_price): # API 接收張數
        stock_sheets = int(stock_sheets);
        if stock_sheets <= 0: print(f"賣單張數錯誤 ({stock_sheets})。"); return False
        print(f"警告：Sell_Stock 被調用 ({stock_code} {stock_sheets} 張 @ {stock_price:.2f})，當前策略不應觸發賣出。")
        return False


# --- LiveDataManager (與上版本一致) ---
class LiveDataManager:
    def __init__(self, api: Stock_API, ema_period=5, atr_period=14, history_days=30):
        self.api = api; self.ema_period = ema_period; self.atr_period = atr_period
        self.history_days = max(history_days, ema_period + 5, atr_period + 5)
        self.ema_col_name = f'EMA_{self.ema_period}'; self.atr_col_name = f'ATR_{self.atr_period}'

    def get_latest_data_and_indicators(self, stock_code: str) -> pd.Series | None:
        today = datetime.now(); start_date = today - timedelta(days=self.history_days * 2)
        end_date = today; start_date_str = start_date.strftime('%Y%m%d'); end_date_str = end_date.strftime('%Y%m%d')
        print(f"  獲取 {stock_code} 數據: {start_date_str} to {end_date_str}")
        raw_data = self.api.Get_Stock_Informations(stock_code, start_date_str, end_date_str)
        if not raw_data: return None
        try:
            df = pd.DataFrame(raw_data); df['date'] = pd.to_datetime(df['date'], unit='s'); df = df.sort_values('date'); df = df[~df['date'].duplicated(keep='last')]; df = df.set_index('date')
            required_cols = ['open', 'high', 'low', 'close'];
            if not all(col in df.columns for col in required_cols): return None
            for col in required_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=required_cols)
            if df.empty or len(df) < max(self.ema_period, self.atr_period): return None
            df.ta.ema(length=self.ema_period, close='close', append=True, col_names=(self.ema_col_name,))
            df.ta.atr(length=self.atr_period, high='high', low='low', close='close', append=True, col_names=(self.atr_col_name,))
            df = df.dropna()
            if df.empty: return None
            return df.iloc[-1]
        except Exception as e: print(f"  處理 {stock_code} 數據時出錯: {e}"); traceback.print_exc(); return None

    def check_entry_signal(self, latest_data: pd.Series) -> bool:
        if latest_data is None: return False
        low_T = latest_data.get('low', np.inf); close_T = latest_data.get('close', np.nan); ema5_T = latest_data.get(self.ema_col_name, np.nan)
        if pd.isna(low_T) or pd.isna(close_T) or pd.isna(ema5_T): return False
        return low_T <= ema5_T and close_T >= ema5_T

    def get_observation(self, latest_data: pd.Series) -> np.ndarray | None:
        if latest_data is None: return None
        try:
            close_price = latest_data['close']; atr_val = latest_data.get(self.atr_col_name, np.nan); ema_val = latest_data.get(self.ema_col_name, np.nan)
            if pd.isna(close_price) or pd.isna(atr_val) or pd.isna(ema_val) or close_price <= 0 or atr_val <= 0 or ema_val <= 0: return None
            norm_atr = atr_val / close_price
            norm_dist_to_ema5 = (close_price - ema_val) / atr_val if atr_val > 0 else 0.0
            features = [norm_atr, norm_dist_to_ema5]
            observation = np.array(features, dtype=np.float32)
            observation = np.nan_to_num(observation, nan=0.0, posinf=10.0, neginf=-10.0); observation = np.clip(observation, -10.0, 10.0)
            return observation
        except Exception as e: print(f"  計算觀察值時出錯: {e}"); return None


# --- 主交易邏輯 ---
if __name__ == '__main__':
    # --- 配置 ---
    API_ACCOUNT = "N26132089"; API_PASSWORD = "joshua900905" # <<< 您的帳密
    STOCK_CODES_TO_MONITOR = TAIWAN_0050_STOCKS
    MODEL_STOCK_CODE = '2330' # 假設用 2330 訓練的模型
    experiment_name = "ai_ema5_touch_atr_rr_units_1or2_v1" # <<< 確認模型名稱
    MODELS_BASE_DIR = f"tuned_models/{MODEL_STOCK_CODE}/{experiment_name}"
    MODEL_LOAD_PATH = os.path.join(MODELS_BASE_DIR, f"ppo_agent_{MODEL_STOCK_CODE}_final.zip")
    VEC_NORMALIZE_STATS_PATH = os.path.join(MODELS_BASE_DIR, "vecnormalize.pkl")

    EMA_PERIOD = 5; ATR_PERIOD = 14
    BUY_UNITS = [1, 2] # AI 決策的張數 (1 或 2)
    # SHARES_PER_UNIT 不再需要，直接使用張數

    print("--- 初始化 Live Trading Bot (API範例版, 隨機價買入, 無止損止盈, 整張交易) ---")
    print(f"警告：本策略【沒有】內建的止損和止盈退出機制！")
    print(f"警告：將使用【限價單】嘗試買入，僅限整張！不保證成交！")
    print(f"警告：不限制持股數量，請確保帳戶資金充足！")

    # --- 加載模型和正規化環境 ---
    print(f"加載模型: {MODEL_LOAD_PATH}")
    if not os.path.exists(MODEL_LOAD_PATH): print(f"錯誤：找不到模型文件！"); sys.exit(1)
    try: model = PPO.load(MODEL_LOAD_PATH, device='cpu'); print("模型加載成功。")
    except Exception as e: print(f"加載模型失敗: {e}"); sys.exit(1)

    vec_env = None
    if not os.path.exists(VEC_NORMALIZE_STATS_PATH): print(f"警告：找不到 VecNormalize 文件。")
    else:
        try:
            dummy_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
            dummy_act_space = spaces.Discrete(len(BUY_UNITS))
            dummy_env = gym.Env(); dummy_env.observation_space=dummy_obs_space; dummy_env.action_space=dummy_act_space
            dummy_vec_env_load = DummyVecEnv([lambda: dummy_env])
            vec_env = VecNormalize.load(VEC_NORMALIZE_STATS_PATH, dummy_vec_env_load)
            vec_env.training = False; vec_env.norm_reward = False; print("VecNormalize 加載成功。")
        except Exception as e: print(f"加載 VecNormalize 失敗: {e}.")


    # --- 初始化 API 和數據管理器 ---
    api = Stock_API(API_ACCOUNT, API_PASSWORD)
    data_manager = LiveDataManager(api, ema_period=EMA_PERIOD, atr_period=ATR_PERIOD)

    # --- 記錄成功提交的訂單 ---
    submitted_buy_orders = []

    # --- 主要決策循環 ---
    print("\n--- 開始檢查交易信號 (執行一次) ---")
    try:
        # 1. 獲取當前持倉 (檢查是否有數據)
        print("正在獲取當前持倉...")
        current_holdings_list = api.Get_User_Stocks()
        current_holdings_shares = {} # {code: shares}
        if isinstance(current_holdings_list, list):
             for item in current_holdings_list:
                  code = item.get('stock_code'); shares = item.get('shares')
                  if code and shares is not None:
                       try: current_holdings_shares[str(code)] = int(shares)
                       except (ValueError, TypeError): pass
        print(f"當前持倉 ({len(current_holdings_shares)}): { {k:v for k,v in current_holdings_shares.items() if v>0} }")

        # 2. 遍歷監控列表，尋找買入機會
        buy_orders_to_plan = [] # (code, units, target_price)

        for stock_code in STOCK_CODES_TO_MONITOR:
            print(f"\n檢查 {stock_code}...")
            # 注意：不再檢查是否已持有 current_holdings_shares[stock_code] > 0

            latest_data = data_manager.get_latest_data_and_indicators(stock_code)
            if latest_data is None: print("  無法獲取或處理數據。"); continue

            low_T = latest_data.get('low', np.nan); high_T = latest_data.get('high', np.nan)
            if pd.isna(low_T) or pd.isna(high_T) or high_T < low_T: print("  > 無法獲取有效的 T 日高低價。"); continue

            entry_signal = data_manager.check_entry_signal(latest_data)
            if entry_signal:
                print(f"  > {stock_code}: 檢測到 EMA{EMA_PERIOD} 觸及信號！")
                observation = data_manager.get_observation(latest_data)
                if observation is None: print("  > 無法生成觀察值。"); continue

                normalized_obs = observation
                if vec_env:
                    try: normalized_obs = vec_env.normalize_obs(np.array([observation]))[0]
                    except Exception as e: print(f"  > 正規化錯誤: {e}")

                action, _ = model.predict(normalized_obs, deterministic=True)
                chosen_units = BUY_UNITS[action] # 1 或 2
                print(f"  > AI 建議買入: {chosen_units} 張")

                target_buy_price = round(random.uniform(low_T, high_T), 2)
                print(f"    > 隨機目標買入價: {target_buy_price:.2f} (範圍: {low_T:.2f} - {high_T:.2f})")
                buy_orders_to_plan.append((stock_code, chosen_units, target_buy_price)) # <<< 記錄計劃
            else:
                print(f"  > {stock_code}: 無進場信號。")
            time.sleep(0.1) # 稍微減慢 API 請求速度

        # 3. 執行買入訂單
        print(f"\n--- 計劃執行的買入訂單 ({len(buy_orders_to_plan)}): ---")
        if not buy_orders_to_plan:
            print("今日無買入操作。")
        else:
            executed_count = 0
            # 在這裡可以添加總體資金檢查邏輯
            # current_cash = api.get_cash_balance() # 假設有獲取現金的API
            # total_planned_cost = 0
            # for code, units, price in buy_orders_to_plan: total_planned_cost += units * 1000 * price
            # if total_planned_cost > current_cash * 0.8: # 例如最多使用80%現金
            #     print("警告：計劃總成本過高，可能部分訂單無法執行！")

            for code, units, target_price in buy_orders_to_plan:
                print(f"\n  準備提交限價買單: {code} {units} 張 @ {target_price:.2f}")
                # <<< 調用 Buy_Stock，傳入張數 >>>
                buy_success = api.Buy_Stock(code, units, target_price)
                if buy_success:
                    print(f"    > {code} 限價買單提交成功。")
                    submitted_buy_orders.append({ # 記錄成功提交的
                        "code": code, "units": units, "target_price": target_price, "status": "提交成功"
                    })
                    executed_count += 1
                else:
                    print(f"    > {code} 限價買單提交失敗。")
                time.sleep(1.5) # 增加下單間隔
            print(f"\n總共成功提交了 {executed_count} 筆買入訂單。")

    except Exception as e:
        print("\n--- 發生未預期錯誤 ---")
        traceback.print_exc()
    finally:
         # --- 打印今日下單總結 ---
         print("\n--- 今日下單總結 ---")
         if submitted_buy_orders:
             print(f"成功提交 {len(submitted_buy_orders)} 筆買單:")
             for order in submitted_buy_orders:
                 print(f"  - 股票: {order['code']}, 張數: {order['units']}, 目標價: {order['target_price']:.2f}")
         else:
             print("今日未成功提交任何買單。")

         # --- 清理 ---
         if vec_env is not None: vec_env.close(); print("VecNormalize 環境已關閉。")

    print("\n--- Live Trading Bot 單次執行完畢 ---")