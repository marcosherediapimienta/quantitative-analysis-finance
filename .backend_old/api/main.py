import os
import time
from typing import Dict, List, Optional

import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from datetime import date, datetime as dt


app = FastAPI(title="Quant Finance API", version="1.0.0")

# CORS configuration via env vars to support Vercel
_env_origins = os.getenv("FRONTEND_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
origins = [o.strip() for o in _env_origins.split(",") if o.strip()]
allow_origin_regex = os.getenv("ALLOW_ORIGIN_REGEX", r"https://.*\.vercel\.app")
app.add_middleware(
	CORSMiddleware,
	allow_origins=origins,
	allow_origin_regex=allow_origin_regex,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Simple in-memory cache with TTL to avoid hammering Yahoo
_CACHE: Dict[str, Dict[str, object]] = {}
_DEFAULT_TTL_SECONDS = 60


def _cache_get(key: str) -> Optional[object]:
	entry = _CACHE.get(key)
	if not entry:
		return None
	if time.time() - entry["ts"] > entry.get("ttl", _DEFAULT_TTL_SECONDS):
		_CACHE.pop(key, None)
		return None
	return entry["data"]


def _cache_set(key: str, data: object, ttl: int = _DEFAULT_TTL_SECONDS) -> None:
	_CACHE[key] = {"data": data, "ts": time.time(), "ttl": ttl}


def _serialize_frame(df):
	if df is None or df.empty:
		return []
	out = []
	df = df.copy()
	if df.index.name is None:
		df.index.name = "date"
	df = df.reset_index()
	for row in df.to_dict(orient="records"):
		obj = {}
		for k, v in row.items():
			if hasattr(v, "isoformat"):
				obj[k] = v.isoformat()
			elif isinstance(v, float):
				obj[k] = float(round(v, 6))
			elif v is None:
				obj[k] = None
			else:
				obj[k] = v
		out.append(obj)
	return out


def _serialize_options_df(df):
	if df is None or df.empty:
		return []
	out: List[Dict[str, object]] = []
	for row in df.to_dict(orient="records"):
		obj: Dict[str, object] = {}
		for k, v in row.items():
			if hasattr(v, "isoformat"):
				obj[k] = v.isoformat()
			elif isinstance(v, float):
				obj[k] = float(round(v, 6))
			elif v is None:
				obj[k] = None
			else:
				obj[k] = v
		out.append(obj)
	return out


@app.get("/health")
def health():
	return {"status": "ok"}


@app.get("/yahoo/quote")
def yahoo_quote(symbol: str = Query(..., min_length=1, description="Ticker como AAPL, MSFT")):
	key = f"quote:{symbol.upper()}"
	cached = _cache_get(key)
	if cached is not None:
		return cached
	try:
		t = yf.Ticker(symbol)
		fi = getattr(t, "fast_info", {}) or {}
		price = fi.get("last_price")
		prev = fi.get("previous_close")
		if price is None or prev is None:
			h = t.history(period="5d", auto_adjust=False)
			if h is None or h.empty:
				raise ValueError("No history data")
			price = float(h["Close"].iloc[-1])
			prev = float(h["Close"].iloc[-2]) if len(h) > 1 else price
		change = float(price) - float(prev)
		change_pct = (change / float(prev)) * 100 if prev else 0.0
		info = {
			"symbol": symbol.upper(),
			"price": float(price),
			"previousClose": float(prev),
			"change": round(change, 6),
			"changePercent": round(change_pct, 6),
			"currency": fi.get("currency") or getattr(t, "info", {}).get("currency"),
			"market": fi.get("market") or getattr(t, "info", {}).get("market"),
			"shortName": getattr(t, "info", {}).get("shortName") or symbol.upper(),
		}
		_cache_set(key, info)
		return info
	except Exception as exc:
		raise HTTPException(status_code=502, detail=f"Yahoo quote error: {exc}")


@app.get("/yahoo/history")
def yahoo_history(
	symbol: str = Query(..., min_length=1),
	period: str = Query("6mo", description="1d/5d/1mo/3mo/6mo/1y/2y/5y/10y/ytd/max"),
	interval: str = Query("1d", description="1m/2m/5m/15m/30m/60m/90m/1h/1d/5d/1wk/1mo/3mo"),
	auto_adjust: bool = Query(True),
):
	key = f"history:{symbol.upper()}:{period}:{interval}:{auto_adjust}"
	cached = _cache_get(key)
	if cached is not None:
		return cached
	try:
		t = yf.Ticker(symbol)
		df = t.history(period=period, interval=interval, auto_adjust=auto_adjust)
		data = _serialize_frame(df)
		res = {"symbol": symbol.upper(), "period": period, "interval": interval, "data": data}
		_cache_set(key, res)
		return res
	except Exception as exc:
		raise HTTPException(status_code=502, detail=f"Yahoo history error: {exc}")


@app.get("/yahoo/spark")
def yahoo_spark(
	symbols: str = Query(..., description="Lista separada por comas, ej: AAPL,MSFT,GOOG"),
	period: str = Query("1mo"),
	interval: str = Query("1d"),
):
	symbols_list = [s.strip() for s in symbols.split(",") if s.strip()]
	if not symbols_list:
		raise HTTPException(status_code=400, detail="Debe proveer al menos un símbolo")
	key = f"spark:{','.join(symbols_list)}:{period}:{interval}"
	cached = _cache_get(key)
	if cached is not None:
		return cached
	try:
		payload = {}
		for sym in symbols_list:
			t = yf.Ticker(sym)
			df = t.history(period=period, interval=interval, auto_adjust=True)
			payload[sym.upper()] = [
				{"date": r["Date"].isoformat() if "Date" in r else r.get("date"), "close": r.get("Close")}
				for r in _serialize_frame(df)
			]
		_cache_set(key, payload)
		return payload
	except Exception as exc:
		raise HTTPException(status_code=502, detail=f"Yahoo spark error: {exc}")


@app.get("/yahoo/options/expirations")
def yahoo_options_expirations(symbol: str = Query(..., min_length=1)):
	symbol_u = symbol.upper()
	key = f"opt:expirations:{symbol_u}"
	cached = _cache_get(key)
	if cached is not None:
		return cached
	try:
		t = yf.Ticker(symbol)
		exps: List[str] = list(getattr(t, "options", []) or [])
		res = {"symbol": symbol_u, "expirations": exps}
		_cache_set(key, res, ttl=300)
		return res
	except Exception as exc:
		raise HTTPException(status_code=502, detail=f"Yahoo options expirations error: {exc}")


@app.get("/yahoo/options/chain")
def yahoo_options_chain(
	symbol: str = Query(..., min_length=1),
	expiration: str = Query(..., description="YYYY-MM-DD"),
):
	symbol_u = symbol.upper()
	key = f"opt:chain:{symbol_u}:{expiration}"
	cached = _cache_get(key)
	if cached is not None:
		return cached
	try:
		t = yf.Ticker(symbol)
		ch = t.option_chain(expiration)
		calls = _serialize_options_df(ch.calls)
		puts = _serialize_options_df(ch.puts)
		res = {"symbol": symbol_u, "expiration": expiration, "calls": calls, "puts": puts}
		_cache_set(key, res, ttl=120)
		return res
	except Exception as exc:
		raise HTTPException(status_code=502, detail=f"Yahoo options chain error: {exc}")


@app.get("/yahoo/options/chains")
def yahoo_options_chains(
	symbol: str = Query(..., min_length=1),
	date_from: Optional[str] = Query(None, alias="from", description="YYYY-MM-DD"),
	date_to: Optional[str] = Query(None, alias="to", description="YYYY-MM-DD"),
):
	symbol_u = symbol.upper()
	try:
		t = yf.Ticker(symbol)
		exps: List[str] = list(getattr(t, "options", []) or [])
		if not exps:
			return {"symbol": symbol_u, "expirations": [], "chains": {}}
		selected: List[str] = exps
		if date_from or date_to:
			def parse_date(v: Optional[str]) -> Optional[date]:
				if not v:
					return None
				return dt.fromisoformat(v).date()
			start = parse_date(date_from)
			end = parse_date(date_to) or parse_date(date_from)
			if start and end and end < start:
				start, end = end, start
			selected = []
			for s in exps:
				try:
					d = dt.fromisoformat(s).date()
					if (not start or d >= start) and (not end or d <= end):
						selected.append(s)
				except Exception:
					continue
		# Cache key depends on selection
		key = f"opt:chains:{symbol_u}:{','.join(selected)}"
		cached = _cache_get(key)
		if cached is not None:
			return cached
		chains: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
		for exp in selected:
			try:
				ch = t.option_chain(exp)
				chains[exp] = {
					"calls": _serialize_options_df(ch.calls),
					"puts": _serialize_options_df(ch.puts),
				}
			except Exception:
				# skip bad expiration
				continue
		res = {"symbol": symbol_u, "expirations": selected, "chains": chains}
		_cache_set(key, res, ttl=120)
		return res
	except Exception as exc:
		raise HTTPException(status_code=502, detail=f"Yahoo options chains error: {exc}")


if __name__ == "__main__":
	import uvicorn
	uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000))) 