# Portfolio Summary Feature - Quick Reference

## Overview
The trading bot now generates **two AI-powered summaries** after each trading cycle:
1. **Professional Summary** - Portfolio manager style (3-5 sentences)
2. **Quick Take** - Gen-Z style FOMO summary (1-2 sentences max)

---

## Key Features

### ✅ Dual Summary Generation
- **Professional**: Clear, detailed, educational tone for clients
- **Gen-Z Style**: Short, punchy, FOMO-inducing for social media

### ✅ Position Justifications Included
- Each summary explains WHY positions were entered
- References trading rationale and technical indicators
- Shows confidence levels and risk parameters

### ✅ Cost-Effective
- Uses **Google Gemini Flash 1.5-8b** via OpenRouter
- ~$2-3/month for 24/7 operation
- 50-70% cheaper than GPT-4o-mini

### ✅ Centralized Prompts
- All prompts stored in `bot/prompts_v1.py`
- Easy to customize and maintain
- `PROFESSIONAL_SUMMARY_PROMPT` and `SHORT_SUMMARY_PROMPT`

---

## CSV Storage

Both summaries are logged to `data/portfolio_state.csv`:

```csv
timestamp,total_balance,total_equity,total_return_pct,num_positions,position_details,total_margin,net_unrealized_pnl,sharpe_ratio,portfolio_summary,short_summary
```

**New columns:**
- `sharpe_ratio` - Risk-adjusted performance metric
- `portfolio_summary` - Professional detailed summary  
- `short_summary` - Gen-Z style quick take

---

## Example Output

### Professional Summary:
> Your portfolio currently stands at $10,456, up +4.57% with a Sharpe ratio of 1.92 indicating excellent risk-adjusted performance. We hold two positions: our ETH long (+$140, +3.6%) was entered based on strong bullish momentum with price breaking above key moving averages and positive MACD signals, while our BTC position continues to track toward its profit target. Both positions have stop-losses in place to protect capital, and we're monitoring for any invalidation signals while letting our winners run.

### Quick Take (Gen-Z Style):
> 💰 Locked in on ETH and BTC longs—both printing nicely with technicals still bullish and way above stop losses, riding the momentum while the market keeps pumping.

---

## Configuration

### Change Model
Edit `bot/trading_workflow.py`, line ~201:
```python
SUMMARY_MODEL = "google/gemini-flash-1.5-8b"  # Current default
```

**Alternatives:**
- `"qwen/qwen-2.5-7b-instruct"` - Ultra cheap (~$1-2/month)
- `"meta-llama/llama-3.1-8b-instruct:free"` - Free tier
- `"anthropic/claude-3-haiku"` - Premium quality (~$8-12/month)

### Customize Prompts
Edit `bot/prompts_v1.py`:
- Line 431: `PROFESSIONAL_SUMMARY_PROMPT`
- Line 451: `SHORT_SUMMARY_PROMPT`

---

## Use Cases

**Professional Summary:**
- Client reports and emails
- Portfolio review meetings
- Documentation and audit trail
- Long-form analysis

**Quick Take (Gen-Z):**
- Twitter/X posts
- Discord/Telegram updates
- Social media engagement
- Mobile push notifications

---

## Cost Breakdown

**Current Setup (Gemini Flash 1.5-8b):**
- Input: $0.075 per 1M tokens
- Output: $0.30 per 1M tokens
- **Monthly cost: ~$2-3** for 24/7 operation

**Comparison:**
- GPT-4o-mini: ~$5-7/month (50% more expensive)
- GPT-4o: ~$80-120/month (40x more expensive!)

---

## Troubleshooting

### "Model not found" error
Change to a different model:
```python
SUMMARY_MODEL = "qwen/qwen-2.5-7b-instruct"
```

### Poor quality summaries
- Try `"anthropic/claude-3-haiku"` for better quality
- Adjust temperature (lower = more conservative)
- Customize prompts in `prompts_v1.py`

### Summaries not appearing
- Check `OPENROUTER_API_KEY` in environment
- Look for errors in logs
- Verify model availability on OpenRouter

---

##Files Modified

1. **bot/prompts_v1.py** - Added summary prompts
2. **bot/utils.py** - Added summary columns to CSV
3. **bot/trading_workflow.py** - Added generation function and integration

---

## Quick Start

1. Ensure `OPENROUTER_API_KEY` is set in your environment
2. Run the bot normally: `python main.py`
3. Summaries will appear after each trading cycle
4. Check `data/portfolio_state.csv` for historical summaries

That's it! The feature is fully integrated and ready to use. 🚀

