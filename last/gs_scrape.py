"""Navigate Google Scholar via Chrome CDP WebSocket and scrape results."""
import json
import time
import urllib.request
from websocket import create_connection

CDP_BASE = "http://127.0.0.1:9222"

def get_page_ws():
    tabs = json.loads(urllib.request.urlopen(f"{CDP_BASE}/json").read())
    for t in tabs:
        if t.get('type') == 'page':
            return t['webSocketDebuggerUrl'], t['id']
    raise Exception("No page tab")

def cdp_call(ws, method, params=None):
    msg = {"id": 1, "method": method, "params": params or {}}
    ws.send(json.dumps(msg))
    response = json.loads(ws.recv())
    return response

def navigate_and_scrape(keywords):
    import urllib.parse
    ws_url, tab_id = get_page_ws()
    print(f"Tab: {tab_id}")
    ws = create_connection(ws_url, timeout=30)

    # Enable Runtime
    cdp_call(ws, "Runtime.enable")

    # Navigate
    query = urllib.parse.quote(keywords)
    url = f"https://scholar.google.com/scholar?q={query}&hl=en&num=10"
    print(f"Go: {url[:100]}")
    cdp_call(ws, "Page.navigate", {"url": url})

    # Wait for page load
    print("Loading...")
    ws.settimeout(20)
    try:
        while True:
            msg = json.loads(ws.recv())
            if msg.get('method') == 'Page.loadEventFired':
                break
    except:
        pass

    time.sleep(3)

    # Scrape
    scrape_js = """(async () => {
        for (let i = 0; i < 30; i++) {
            if (document.querySelector('#gs_res_ccl') || document.querySelector('#gs_captcha_ccl')) break;
            await new Promise(r => setTimeout(r, 500));
        }
        if (document.querySelector('#gs_captcha_ccl') || document.body.innerText.includes('unusual traffic')) {
            return JSON.stringify({ error: 'captcha', message: 'CAPTCHA required. Please verify in Chrome.' });
        }
        const items = document.querySelectorAll('#gs_res_ccl .gs_r.gs_or.gs_scl');
        if (items.length === 0) {
            return JSON.stringify({
                fallback: true,
                url: window.location.href,
                title: document.title,
                body: document.body.innerText.substring(0, 800)
            });
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
                citedBy: citedByEl?.textContent?.match(/\\d+/)?.[0] || '0',
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

    print("Scraping...")
    result = cdp_call(ws, "Runtime.evaluate", {
        "expression": scrape_js,
        "awaitPromise": True,
        "returnByValue": True
    })
    ws.close()

    value = result.get('result', {}).get('result', {}).get('value', '')
    if not value:
        print("ERROR: No value returned")
        print(json.dumps(result, indent=2)[:500])
        return

    data = json.loads(value)

    if data.get('error') == 'captcha':
        print("\n⚠️  GOOGLE SCHOLAR CAPTCHA!")
        print(data['message'])
        return

    print(f"\n===== GOOGLE SCHOLAR =====")
    print(f"Query: {keywords}")
    print(f"Total: {data.get('total','?')}")
    print(f"Results: {data.get('resultCount', 0)}")

    if data.get('fallback'):
        print(f"FALLBACK — Title: {data.get('title','')}")
        print(f"Body: {data.get('body','')[:500]}")
        return

    for r in data.get('results', []):
        print(f"\n{r['n']}. {r['title']}")
        print(f"   Authors: {r['authors']}")
        print(f"   {r['journalYear']}")
        print(f"   Cited by: {r['citedBy']}  |  CID: {r['dataCid']}")
        if r.get('snippet'):
            print(f"   {r['snippet'][:200]}")

if __name__ == "__main__":
    navigate_and_scrape("children speech emotion recognition WavLM self-supervised 2024 2025")
