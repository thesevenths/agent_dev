import requests
from typing import List, Dict, Optional

def get_nasdaq_top_gainers(top_n: int = 5) -> List[Dict]:
    """
    fetch NASDAQ top gainers from Yahoo Finance, sorted by percentage gain descending.
    
    Args:
        top_n (int): default 5 No. of top gainers to return.
    
    Returns:
        List[Dict]: top n stock list，each include symbol, name, change_pct
    """
    url = "https://query2.finance.yahoo.com/v1/finance/screener/predefined/saved"
    
    # fetch more to ensure enough NASDAQ stocks
    fetch_count = max(top_n * 3, 100)  
    
    params = {
        "scrIds": "day_gainers",
        "count": fetch_count
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://finance.yahoo.com/screener/predefined/day_gainers"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        quotes = data.get('finance', {}).get('result', [{}])[0].get('quotes', [])
        if not quotes:
            return []
        
        nasdaq_stocks = []
        for q in quotes:
            if q.get('exchange') == 'NMS' and q.get('regularMarketChangePercent') is not None:
                nasdaq_stocks.append({
                    'symbol': q.get('symbol', 'N/A'),
                    'name': q.get('shortName', 'N/A'),
                    'change_pct': float(q['regularMarketChangePercent'])
                })
        
        # sort by percentage gain descending
        nasdaq_stocks.sort(key=lambda x: x['change_pct'], reverse=True)
        
        # return top nq
        return nasdaq_stocks[:top_n]
        
    except Exception as e:
        print(f"⚠️ Warning: Failed to fetch data: {e}")
        return []

if __name__ == "__main__":
    TOP_N = 5  # <<< 可在此处配置你需要的 Top N
    
    top_gainers = get_nasdaq_top_gainers(top_n=TOP_N)
    
    if top_gainers:
        print(f"🏆 Top {len(top_gainers)} NASDAQ Gainers Today:")
        for i, stock in enumerate(top_gainers, 1):
            print(f"{i}. {stock['symbol']} ({stock['name']}) +{stock['change_pct']:.2f}%")
        
        # 如果你需要 JSON 输出（适合 agent 调用），可以这样：
        import json
        print("\n--- JSON Output ---")
        print(json.dumps(top_gainers, indent=2, ensure_ascii=False))
    else:
        print("❌ No NASDAQ gainers found.")