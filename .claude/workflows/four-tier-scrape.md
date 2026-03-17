---
description: Progressive four-tier URL content scraping with automatic fallback strategy
---

# Four-Tier URL Content Scraping

Progressive escalation strategy to retrieve URL content using four fallback tiers.

## Decision Logic

```
START → Tier 1 (WebFetch) → Success? → Done
                              ↓ No
         Tier 2 (Curl + Chrome Headers) → Success? → Done
                              ↓ No
         Tier 3 (Browser Automation) → Success? → Done
                              ↓ No
         Tier 4 (Bright Data MCP) → Success? → Done
                              ↓ No
         Report failure
```

## Tier 1: WebFetch (Fast & Simple, ~2-5 sec)
```
WebFetch(url="[URL]", prompt="Extract all content and convert to markdown")
```

## Tier 2: Curl with Chrome Headers (~3-7 sec)
```bash
curl -L -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" \
  -H "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8" \
  -H "Accept-Language: en-US,en;q=0.9" \
  -H "Accept-Encoding: gzip, deflate, br" \
  -H "DNT: 1" \
  -H "Connection: keep-alive" \
  -H "Upgrade-Insecure-Requests: 1" \
  -H "Sec-Fetch-Dest: document" \
  -H "Sec-Fetch-Mode: navigate" \
  -H "Sec-Fetch-Site: none" \
  -H "Sec-Fetch-User: ?1" \
  -H "Cache-Control: max-age=0" \
  --compressed "[URL]"
```

## Tier 3: Browser Automation (Playwright, ~10-20 sec)
Use Playwright MCP if available for JavaScript-heavy sites.

## Tier 4: Bright Data MCP (~5-15 sec)
```
mcp__Brightdata__scrape_as_markdown(url="[URL]")
```

## Skip Logic
- User says "use Bright Data" → Skip to Tier 4
- Known SPA/JS-heavy site → Start at Tier 3
- Known CAPTCHA site → Start at Tier 4
- Previous failure on same domain → Start at Tier 2+

## Cost
- Tiers 1-3: Free
- Tier 4: ~$0.01/day average with heavy use
