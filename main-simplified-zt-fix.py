# import tushare as ts
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import os
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

TS_TOKEN = '填入 Tushare Token2682' 
INDEX_CODE = 2682
# START_DATE = '20150101'
# END_DATE = '20240101'
TEST_END_DATE = '20260101'

CSV_PATH = 'sinfocsv'


BATCH_SIZE = 1024
TRAIN_ITERATIONS = 400     
MAX_SEQ_LEN = 15            # 限制公式长度，防止过拟合
COST_RATE = 0.0005         

DATA_CACHE_PATH = 'data_cache_final.parquet'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

@torch.jit.script
def _ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    if d == 0: return x
    pad = torch.zeros((x.shape[0], d), device=x.device)
    return torch.cat([pad, x[:, :-d]], dim=1)

@torch.jit.script
def _ts_delta(x: torch.Tensor, d: int) -> torch.Tensor:
    return x - _ts_delay(x, d)

@torch.jit.script
def _ts_zscore(x: torch.Tensor, d: int) -> torch.Tensor:
    if d <= 1: return torch.zeros_like(x)
    B, T = x.shape
    pad = torch.zeros((B, d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    mean = windows.mean(dim=-1)
    std = windows.std(dim=-1) + 1e-6
    return (x - mean) / std

@torch.jit.script
def _ts_decay_linear(x: torch.Tensor, d: int) -> torch.Tensor:
    if d <= 1: return x
    B, T = x.shape
    pad = torch.zeros((B, d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1) # [B, T, d]
    w = torch.arange(1, d + 1, device=x.device, dtype=x.dtype)
    w = w / w.sum()
    return (windows * w).sum(dim=-1)

OPS_CONFIG = [
    ('ADD', lambda x, y: x + y, 2),
    ('SUB', lambda x, y: x - y, 2),
    ('MUL', lambda x, y: x * y, 2),
    ('DIV', lambda x, y: x / (y + 1e-6 * torch.sign(y)), 2),
    ('NEG', lambda x: -x, 1),
    ('ABS', lambda x: torch.abs(x), 1),
    ('SIGN', lambda x: torch.sign(x), 1),
    ('DELTA5', lambda x: _ts_delta(x, 5), 1),
    ('MA20',   lambda x: _ts_decay_linear(x, 20), 1),
    ('STD20',  lambda x: _ts_zscore(x, 20), 1),
    ('TS_RANK20', lambda x: _ts_zscore(x, 20), 1),
]

FEATURES = ['RET', 'RET5', 'VOL_CHG', 'V_RET', 'TREND','FANS','SBBR','ORG','EM','TH','TURNOVER','TDXGZ'] 

VOCAB = FEATURES + [cfg[0] for cfg in OPS_CONFIG]
VOCAB_SIZE = len(VOCAB)
OP_FUNC_MAP = {i + len(FEATURES): cfg[1] for i, cfg in enumerate(OPS_CONFIG)}
OP_ARITY_MAP = {i + len(FEATURES): cfg[2] for i, cfg in enumerate(OPS_CONFIG)}

class AlphaGPT(nn.Module):
    def __init__(self, d_model=64, n_head=4, n_layer=2):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, MAX_SEQ_LEN + 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=128, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head_actor = nn.Linear(d_model, VOCAB_SIZE)
        self.head_critic = nn.Linear(d_model, 1)

    def forward(self, idx):
        B, T = idx.size()
        x = self.token_emb(idx) + self.pos_emb[:, :T, :]
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(idx.device)
        x = self.blocks(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        last = x[:, -1, :]
        return self.head_actor(last), self.head_critic(last)

class DataEngine:
    def __init__(self,input_pd):
        # self.pro = ts.pro_api(TS_TOKEN)
        # self.df_tmp = pd.read_csv(CSV_PATH)
        self.df_tmp = input_pd
    def load(self):
        # if os.path.exists(DATA_CACHE_PATH):
        #     df = pd.read_parquet(DATA_CACHE_PATH)
        # else:
        #     print(f"🌐 Fetching {INDEX_CODE}...")
        #     if INDEX_CODE.endswith(".SZ") or INDEX_CODE.endswith(".SH"):
        #         try:
        #             df = self.pro.fund_daily(ts_code=INDEX_CODE, start_date=START_DATE, end_date=TEST_END_DATE)
        #         except:
        #             df = self.pro.index_daily(ts_code=INDEX_CODE, start_date=START_DATE, end_date=TEST_END_DATE)
            
        #     if df is None or df.empty:
        #         raise ValueError("未获取到数据，请检查Token或代码是否正确")
                
        #     df = df.sort_values('trade_date').reset_index(drop=True)
        #     df.to_parquet(DATA_CACHE_PATH)
        # df = self.df_tmp[self.df_tmp['codelist'] == INDEX_CODE]
        df = self.df_tmp
            
        
        for col in ['open', 'high', 'low', 'close', 'vol','fans','sbbr','org','em','th','tdxgz','turnover']:
            df[col] = pd.to_numeric(df[col], errors='coerce').ffill().bfill()
        # from pprint import pprint
        # pprint(df)
        self.dates = pd.to_datetime(df['trade_date'])
        
        close = df['close'].values.astype(np.float32)
        open_ = df['open'].values.astype(np.float32)
        fans = df['fans'].values.astype(np.float32)
        sbbr = df['sbbr'].values.astype(np.float32)
        org = df['org'].values.astype(np.float32)
        em = df['em'].values.astype(np.float32)
        th = df['th'].values.astype(np.float32)
        tdxgz = df['tdxgz'].values.astype(np.float32)
        turnover = df['turnover'].values.astype(np.float32)

        vol = df['vol'].values.astype(np.float32)
        
        ret = np.zeros_like(close)
        ret[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-6)
        
        ret5 = pd.Series(close).pct_change(5).fillna(0).values.astype(np.float32)
        
        vol_ma = pd.Series(vol).rolling(20).mean().values
        vol_chg = np.zeros_like(vol)
        mask = vol_ma > 0
        vol_chg[mask] = vol[mask] / vol_ma[mask] - 1
        vol_chg = np.nan_to_num(vol_chg).astype(np.float32)
        
        v_ret = (ret * (vol_chg + 1)).astype(np.float32)
        
        ma60 = pd.Series(close).rolling(60).mean().values
        trend = np.zeros_like(close)
        mask = ma60 > 0
        trend[mask] = close[mask] / ma60[mask] - 1
        trend = np.nan_to_num(trend).astype(np.float32)
        
        # Robust Normalization
        def robust_norm(x):
            x = x.astype(np.float32) # 强制转类型
            median = np.nanmedian(x)
            mad = np.nanmedian(np.abs(x - median)) + 1e-6
            res = (x - median) / mad
            return np.clip(res, -5, 5).astype(np.float32)

        self.feat_data = torch.stack([
            torch.from_numpy(robust_norm(ret)).to(DEVICE),
            torch.from_numpy(robust_norm(ret5)).to(DEVICE),
            torch.from_numpy(robust_norm(vol_chg)).to(DEVICE),
            torch.from_numpy(robust_norm(v_ret)).to(DEVICE),
            torch.from_numpy(robust_norm(trend)).to(DEVICE),
            torch.from_numpy(robust_norm(fans)).to(DEVICE),
            torch.from_numpy(robust_norm(sbbr)).to(DEVICE),
            torch.from_numpy(robust_norm(org)).to(DEVICE),
            torch.from_numpy(robust_norm(em)).to(DEVICE),
            torch.from_numpy(robust_norm(th)).to(DEVICE),
            torch.from_numpy(robust_norm(turnover)).to(DEVICE),
            torch.from_numpy(robust_norm(tdxgz)).to(DEVICE)
        ])
        
        open_tensor = torch.from_numpy(open_).to(DEVICE)
        open_t1 = torch.roll(open_tensor, -1)
        open_t2 = torch.roll(open_tensor, -2)
        
        self.target_oto_ret = (open_t2 - open_t1) / (open_t1 + 1e-6)
        self.target_oto_ret[-2:] = 0.0


        # 在 DataEngine.load() 中添加  
        close_tensor = torch.from_numpy(close).to(DEVICE)  
        
        close_t1 = torch.roll(close_tensor, -1)  
        daily_return = (close_t1 - close_tensor) / (close_tensor + 1e-6)  
         
        LIMIT_UP_THRESHOLD = 0.095  
        self.target_limit_up = (daily_return >= LIMIT_UP_THRESHOLD).float()  
        self.target_limit_up[-1] = 0.0  # 最后一天无标签
        
        
        self.raw_open = open_tensor
        self.raw_close = torch.from_numpy(close).to(DEVICE)
        
        self.split_idx = int(len(df) * 0.8)
        print(f"{INDEX_CODE} Data Ready. Normalization Fixed.")
        return self

class DeepQuantMiner:
    def __init__(self, engine):
        self.engine = engine
        self.model = AlphaGPT().to(DEVICE)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-5)
        self.best_sharpe = -10.0
        self.best_formula_tokens = None
        
    def get_strict_mask(self, open_slots, step):
        B = open_slots.shape[0]
        mask = torch.full((B, VOCAB_SIZE), float('-inf'), device=DEVICE)
        remaining_steps = MAX_SEQ_LEN - step
        
        done_mask = (open_slots == 0)
        mask[done_mask, 0] = 0.0 # Pad with first feature
        
        active_mask = ~done_mask
        # 如果剩余步数不够填坑了，必须选 Feature (arity=0)
        must_pick_feat = (open_slots >= remaining_steps)
        
        mask[active_mask, :len(FEATURES)] = 0.0
        can_pick_op_mask = active_mask & (~must_pick_feat)
        if can_pick_op_mask.any():
            mask[can_pick_op_mask, len(FEATURES):] = 0.0
        return mask

    def solve_one(self, tokens):
        stack = []
        try:
            for t in reversed(tokens):
                if t < len(FEATURES):
                    stack.append(self.engine.feat_data[t])
                else:
                    arity = OP_ARITY_MAP[t]
                    if len(stack) < arity: raise ValueError
                    args = [stack.pop() for _ in range(arity)]
                    func = OP_FUNC_MAP[t]
                    if arity == 2: res = func(args[0], args[1])
                    else: res = func(args[0])
                    
                    if torch.isnan(res).any(): res = torch.nan_to_num(res)
                    stack.append(res)
            
            if len(stack) >= 1:
                final = stack[-1]
                # 过滤掉常数因子
                if final.std() < 1e-4: return None
                return final
        except:
            return None
        return None

    def solve_batch(self, token_seqs):
        B = token_seqs.shape[0]
        results = torch.zeros((B, self.engine.feat_data.shape[1]), device=DEVICE)
        valid_mask = torch.zeros(B, dtype=torch.bool, device=DEVICE)
        
        for i in range(B):
            res = self.solve_one(token_seqs[i].cpu().tolist())
            if res is not None:
                results[i] = res
                valid_mask[i] = True
        return results, valid_mask
    def backtest(self, factors):
        if factors.shape[0] == 0: return torch.tensor([], device=DEVICE)
        
        split = self.engine.split_idx  
        target = self.engine.target_limit_up[:split]  # 改为二分类标签 
        
        rewards = torch.zeros(factors.shape[0], device=DEVICE)
        
        for i in range(factors.shape[0]):
            f = factors[i, :split]
            
            if torch.isnan(f).all() or (f == 0).all() or f.numel() == 0:
                rewards[i] = -2.0
                continue
            # 生成预测：因子值大于0预测涨停，小于0预测不涨停  
            pred = (f > 0).float()  
          
            if pred.numel() < 10:  
                rewards[i] = -2.0  
                continue  
          
            # 计算分类指标  
            correct = (pred == target).float()  
            accuracy = correct.mean()  
            # 计算精确率和召回率  
            true_positives = ((pred == 1) & (target == 1)).sum()  
            false_positives = ((pred == 1) & (target == 0)).sum()  
            false_negatives = ((pred == 0) & (target == 1)).sum()  
              
            precision = true_positives / (true_positives + false_positives + 1e-6)  
            recall = true_positives / (true_positives + false_negatives + 1e-6)  
              
            # 使用F1-score作为奖励（也可以根据业务需求调整权重）  
            f1_score = 2 * precision * recall / (precision + recall + 1e-6)  
              
            # 归一化到合理范围  
            reward = f1_score * 10 - 5  # 将[0,1]映射到[-5,5]  
              
            # 惩罚项  
            if accuracy < 0.5: reward = -2.0  # 准确率低于随机  
            if recall < 0.1: reward -= 1.0     # 召回率太低  
            if (pred == 0).all(): reward = -2.0  #
              
            rewards[i] = reward  
          
        return torch.clamp(rewards, -3, 5)

            
            
    def train(self):
        print(f"Training for Stable Profit... MAX_LEN={MAX_SEQ_LEN}")
        pbar = tqdm(range(TRAIN_ITERATIONS))
        
        for _ in pbar:
            # 1. Generate
            B = BATCH_SIZE
            open_slots = torch.ones(B, dtype=torch.long, device=DEVICE)
            log_probs, tokens = [], []
            curr_inp = torch.zeros((B, 1), dtype=torch.long, device=DEVICE)
            
            for step in range(MAX_SEQ_LEN):
                logits, val = self.model(curr_inp)
                mask = self.get_strict_mask(open_slots, step)
                dist = Categorical(logits=(logits + mask))
                action = dist.sample()
                
                log_probs.append(dist.log_prob(action))
                tokens.append(action)
                curr_inp = torch.cat([curr_inp, action.unsqueeze(1)], dim=1)
                
                is_op = action >= len(FEATURES)
                delta = torch.full((B,), -1, device=DEVICE)
                arity_tens = torch.zeros(VOCAB_SIZE, dtype=torch.long, device=DEVICE)
                for k,v in OP_ARITY_MAP.items(): arity_tens[k] = v
                op_delta = arity_tens[action] - 1
                delta = torch.where(is_op, op_delta, delta)
                delta[open_slots==0] = 0
                open_slots += delta
            
            seqs = torch.stack(tokens, dim=1)
            
            # 2. Evaluate
            with torch.no_grad():
                f_vals, valid_mask = self.solve_batch(seqs)
                valid_idx = torch.where(valid_mask)[0]
                rewards = torch.full((B,), -1.0, device=DEVICE) # 默认惩罚
                
                if len(valid_idx) > 0:
                    bt_scores = self.backtest(f_vals[valid_idx])
                    rewards[valid_idx] = bt_scores
                    
                    best_sub_idx = torch.argmax(bt_scores)
                    current_best_score = bt_scores[best_sub_idx].item()
                    
                    if current_best_score > self.best_sharpe:
                        self.best_sharpe = current_best_score
                        self.best_formula_tokens = seqs[valid_idx[best_sub_idx]].cpu().tolist()
            
            # 3. Update
            adv = rewards - rewards.mean()
            loss = -(torch.stack(log_probs, 1).sum(1) * adv).mean()
            
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
            pbar.set_postfix({'Valid': f"{len(valid_idx)/B:.1%}", 'BestSortino': f"{self.best_sharpe:.2f}"})

    def decode(self, tokens=None):
        if tokens is None: tokens = self.best_formula_tokens
        if tokens is None: return "N/A"
        stream = list(tokens)
        def _parse():
            if not stream: return ""
            t = stream.pop(0)
            if t < len(FEATURES): return FEATURES[t]
            args = [_parse() for _ in range(OP_ARITY_MAP[t])]
            return f"{VOCAB[t]}({','.join(args)})"
        try: return _parse()
        except: return "Invalid"

def final_reality_check(miner, engine):
    print("\n" + "="*60)
    print("FINAL CHECK (Out-of-Sample)")
    print("="*60)
    
    formula_str = miner.decode()
    if miner.best_formula_tokens is None: return
    print(f"Strategy Formula: {formula_str}")
    
    # 1. 获取全量因子值
    factor_all = miner.solve_one(miner.best_formula_tokens)
    if factor_all is None: return
    
    # 2. 提取测试集数据 (Strict OOS)
    split = engine.split_idx
    test_dates = engine.dates[split:]
    test_factors = factor_all[split:].cpu().numpy()
    
    # 改为使用涨停标签  
    test_labels = engine.target_limit_up[split:].cpu().numpy()  
      
    # 预测  
    predictions = (test_factors > 0).astype(int)  
    
    # 计算分类指标  
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix  
      
    accuracy = accuracy_score(test_labels, predictions)  
    precision = precision_score(test_labels, predictions, zero_division=0)  
    recall = recall_score(test_labels, predictions, zero_division=0)  
    f1 = f1_score(test_labels, predictions, zero_division=0)  
    cm = confusion_matrix(test_labels, predictions)  
    seq_dict.update({int(ele):{'formula':str(formula_str),'cm':cm.tolist()}})
        
    print(f"Test Period    : {test_dates.iloc[0].date()} ~ {test_dates.iloc[-1].date()}")
    print(f"Accuracy    : {accuracy:.2%}")  
    print(f"Precision   : {precision:.2%}")  
    print(f"Recall      : {recall:.2%}")  
    print(f"F1-Score    : {f1:.2f}")  
    print(f"Confusion Matrix:\n{cm}")
    print("-" * 60)  
    
    

if __name__ == "__main__":
    pd_data = pd.read_csv(CSV_PATH)
    import json
    
    # unique_values = pd_data['codelist'].unique()
    unique_values=[
1331,
2931,
2788,
600783,
2718,
2202,
600266,
600337,
2324,
2519,
2774,
600775,
2462,
301363,
300986,
2969,
2703,
2151,
601698,
2757,
1231,
2465,
600759,
2766,
603163,
2571,
603458,
2361,
600118,
1266,
605003,
603608,
2009,
2591,
2652,
603601,
2163,
2853,
600179,
603069,
2201,
36,
2865,
600330,
523,
2363,
605007,
600755,
600865,
603698,
2228,
551,
603696,
601231,
2596,
1336,
2149,
603023,
2639,
600936,
603585,
2188,
2565,
600250,
1330,
600280,
950,
2329,
601366,
3031,
1317,
600828,
2792,
2300,
890,
1208,
2413,
600151,
2682,
603709,
600343,
601399,
601106,
2235,
905,
3016,
603386,
605299,
603122,
2474,
600734,
2589,
2585,
547,
2702,
859,
78,
600981,
600868,
603933,
2348,
2632,
603327,
2353,
592,
3018,
603216,
2976,
603778,
2114,
2307,
2083,
952,
600829,
70,
2264,
620,
798,
663,
300589,
600097,
1216,
603829,
17,
601566,
601279,
2877,
600756,
546,
2192,
407,
633,
600408,
632,
1203,
600815,
2805,
2326,
688,
603026,
2451,
572,
2578,
600429,
603978,
2309,
2759,
2374,
993,
300437,
2255,
603169,
603759,
603067,
605060,
605255,
2150,
2885,
2512,
593,
603823,
2679,
555,
605178,
2790,
300300,
603378,
603232,
2160,
2213,
599,
2748,
1267,
601608,
600121,
2084,
605318,
2678,
637,
600403,
852,
2054,
2513,
600262,
2208,
600748,
626,
601011,
603938,
600617,
601212,
600078,
600382,
2825,
600629,
2942,
601126,
600960,
601611,
981,
969,
603011,
600635,
600167,
603156,
2295,
603286,
21,
300948,
2298,
1309,
559,
300250,
2925,
603239,
2453,
605358,
600400,
300111,
600159,
1234,
603618,
603686,
2249,
2674,
600996,
600735,
600376,
2059,
603297,
600699,
600170,
600162,
2905,
600651,
2067,
605289,
601619,
605298,
603516,
603959,
1202,
601116,
600977,
600173,
603127,
600103,
718,
603228,
980,
603301,
603222,
2384,
603101,
2975,
603285,
601069,
2846,
603257,
603839,
600021,
1296,
601929,
601869,
603626,
605288,
2256,
603032,
600246,
605188,
603177,
2217,
603948,
2053,
2600,
603083,
2212,
600410,
605303,
603158,
2158,
605028,
2871,
600658,
700,
2046,
600619,
2052,
2873,
601606,
603757,
2536,
561,
603221,
2939,
601609,
600737,
2879,
605169,
603256,
600208,
2782,
2173,
605598,
603767,
2148,
901,
603800,
2941,
600366,
600053,
600581,
603080,
2097,
2313,
603811,
601089,
534,
300289,
603059,
2775,
300486,
2287,
600749,
603280,
600117,
603616,
603367,
300528,
300877,
2644,
600326,
301038,
2370,
1226,
2297,
2205,
603176,
3023,
601669,
2883,
600528,
2827,
2037,
603916,
796,
1360,
300564,
600774,
2266,
600801,
3002,
600875,
2096,
3001,
603966,
1239,
601003,
600826,
605122,
3037,
2246,
600230,
2645,
603115,
600620,
600822,
2394,
2393,
2401,
2107,
605500,
600513,
2209,
1896,
514,
605259,
301388,
605399,
2121,
2623,
600962,
607,
2356,
600475,
603716,
2314,
655,
2218,
600744,
2636,
605162,
600537,
600571,
2263,
605277,
603676,
603936,
600192,
2721,
301024,
2579,
603316,
1298,
2658,
601218,
600113,
26,
605151,
600319,
600643,
601718,
2210
]
    seq_dict={}
    for ele in unique_values:
        subset = pd_data[pd_data['codelist'] == ele]    
        eng = DataEngine(subset)
        eng.load()
        miner = DeepQuantMiner(eng)
        miner.train()
        final_reality_check(miner, eng)
        with open('data.json', 'w', encoding='utf-8') as f:
            json.dump(seq_dict, f, indent=2, ensure_ascii=False)
    
    