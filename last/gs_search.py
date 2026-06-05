"""Search Google Scholar via Chrome CDP WebSocket."""
import json, time, urllib.request, urllib.parse, sys
import websocket

KEYWORDS = sys.argv[1] if len(sys.argv) > 1 else "children speech emotion recognition"

def cdp(ws, method, params=None):
    ws.send(json.dumps({"id": 1, "method": method, "params": params or {}}))
    return json.loads(ws.recv())

# Connect
tabs = json.loads(urllib.request.urlopen("http://127.0.0.1:9222/json").read())
page = next(t for t in tabs if t["type"] == "page")
ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=30)
cdp(ws, "Runtime.enable")

# Navigate
query = urllib.parse.quote(KEYWORDS)
url = f"https://scholar.google.com/scholar?q={query}&hl=en&num=10"
print(f"Navigating to Google Scholar...")
cdp(ws, "Page.navigate", {"url": url})

# Wait for load
ws.settimeout(25)
try:
    while True:
        msg = json.loads(ws.recv())
        if msg.get("method") == "Page.loadEventFired":
            print("Page loaded!")
            break
except:
    print("Load wait ended")

time.sleep(4)

# Scrape
scrape_js = """(async () => {
    for (let i = 0; i < 30; i++) {
        if (document.querySelector('#gs_res_ccl') || document.querySelector('#gs_captcha_ccl')) break;
        await new Promise(r => setTimeout(r, 500));
    }
    if (document.querySelector('#gs_captcha_ccl') || document.body.innerText.includes('unusual traffic')) {
        return JSON.stringify({ error: 'captcha', message: 'Google Scholar requires CAPTCHA verification.' });
    }
    const items = document.querySelectorAll('#gs_res_ccl .gs_r.gs_or.gs_scl');
    if (items.length === 0) {
        return JSON.stringify({ fallback: true, title: document.title, url: window.location.href,
            body: (document.body?.innerText || '').substring(0, 600) });
    }
    const results = Array.from(items).slice(0, 10).map((item, i) => {
        const titleEl = item.querySelector('.gs_rt a');
        const meta = item.querySelector('.gs_a')?.textContent || '';
        const parts = meta.split(' - ');
        const citedByEl = item.querySelector('.gs_fl a[href*="cites"]');
        return {
            n: i + 1,
            title: (titleEl?.textContent || item.querySelector('.gs_rt')?.textContent || '').trim(),
            href: titleEl?.href || '',
            authors: (parts[0] || '').trim(),
            journalYear: (parts[1] || '').trim(),
            citedBy: citedByEl?.textContent?.match(/\\\\d+/)?.[0] || '0',
            dataCid: item.getAttribute('data-cid') || '',
            snippet: (item.querySelector('.gs_rs')?.textContent || '').trim().substring(0, 300),
        };
    });
    return JSON.stringify({
        total: document.querySelector('#gs_ab_md')?.textContent?.trim() || '',
        resultCount: results.length,
        currentUrl: window.location.href,
        results
    });
})()"""

print("Scraping results...")
result = cdp(ws, "Runtime.evaluate", {"expression": scrape_js, "awaitPromise": True, "returnByValue": True})
ws.close()

value = result.get("result", {}).get("result", {}).get("value", "")
if not value:
    print("ERROR: No results returned")
    sys.exit(1)

data = json.loads(value)

if data.get("error") == "captcha":
    print("\n⚠️  GOOGLE SCHOLAR CAPTCHA!")
    print("Please open Chrome at http://127.0.0.1:9222 and solve the CAPTCHA, then re-run this script.")
    sys.exit(0)

print(f"\n{'='*70}")
print(f"GOOGLE SCHOLAR SEARCH: {KEYWORDS}")
print(f"Total: {data.get('total', '?')}")
print(f"Results: {data.get('resultCount', 0)}")
print(f"{'='*70}")

if data.get("fallback"):
    print(f"\nFALLBACK MODE")
    print(f"URL: {data.get('url', '')}")
    print(f"Title: {data.get('title', '')}")
    print(f"Body: {data.get('body', '')[:500]}")
else:
    # Use ASCII-safe output to avoid encoding issues
    for r in data.get("results", []):
        title = r['title'].encode('ascii', errors='replace').decode()
        authors = r['authors'].encode('ascii', errors='replace').decode()
        journal = r['journalYear'].encode('ascii', errors='replace').decode()
        snippet = r.get('snippet', '').encode('ascii', errors='replace').decode()
        print(f"\n{r['n']}. {title}")
        print(f"   Authors: {authors}")
        print(f"   {journal}")
        print(f"   Cited by: {r['citedBy']}  |  CID: {r['dataCid']}")
        if snippet:
            print(f"   {snippet[:200]}")
        if r.get("href"):
            print(f"   Link: {r['href'][:120]}")
