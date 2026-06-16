"""Google Scholar scraper via Chrome CDP HTTP endpoint."""
import urllib.request, json, time, urllib.parse

def cdp_http(method, params=None):
    body = json.dumps(params or {}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:9222/json/protocol/{method}",
        data=body,
        headers={"Content-Type": "application/json"}
    )
    resp = urllib.request.urlopen(req, timeout=60)
    return json.loads(resp.read())

# Get page tab
tabs = json.loads(urllib.request.urlopen("http://127.0.0.1:9222/json").read())
page = next(t for t in tabs if t['type'] == 'page')
print(f"Tab: {page['id']}")

# Navigate to Google Scholar
keywords = "children speech emotion recognition self-supervised learning"
query = urllib.parse.quote(keywords)
search_url = f"https://scholar.google.com/scholar?q={query}&hl=en&num=20"
print(f"Navigating: {search_url[:100]}...")

result = cdp_http("Page.navigate", {"url": search_url})
print(f"Navigate result frameId: {result.get('result',{}).get('frameId','?')}")

# Wait
print("Waiting 10s...")
time.sleep(10)

# Check what's on the page
js_check = """
(async () => {
    await new Promise(r => setTimeout(r, 3000));
    return JSON.stringify({
        title: document.title,
        url: window.location.href,
        bodyStart: (document.body?.innerText || '').substring(0, 800),
        hasResults: !!document.querySelector('#gs_res_ccl'),
        hasCaptcha: !!document.querySelector('#gs_captcha_ccl'),
        resultCount: document.querySelectorAll('#gs_res_ccl .gs_r').length
    });
})()
"""
result = cdp_http("Runtime.evaluate", {
    "expression": js_check,
    "awaitPromise": True,
    "returnByValue": True
})

value = result.get('result', {}).get('value', '')
if value:
    data = json.loads(value)
    print(f"\nTitle: {data.get('title','?')}")
    print(f"URL: {data.get('url','?')[:120]}")
    print(f"Has results: {data.get('hasResults')}")
    print(f"Has captcha: {data.get('hasCaptcha')}")
    print(f"Result count: {data.get('resultCount')}")
    print(f"\n--- Body start ---")
    print(data.get('bodyStart','')[:800])
else:
    print(f"No value returned")
    print(json.dumps(result, indent=2)[:600])
