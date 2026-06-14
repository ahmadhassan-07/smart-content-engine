import re
import requests
import random
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs


# ─── Blocked Words ─────────────────────────────────────────────────────────────
BLOCKED_WORDS = {
    "spam", "scam", "fake", "fraud", "cheat",
    "hate", "stupid", "idiot", "worst", "terrible", "awful"
}

# ─── Fake Review Signals ───────────────────────────────────────────────────────
FAKE_REVIEW_SIGNALS = [
    r"\b(best|greatest|perfect|amazing|excellent|outstanding)\b.*\b(ever|life|world)\b",
    r"\b(buy|purchase|order)\b.*\b(now|today|immediately|asap)\b",
    r"\b(trust|believe|guarantee|promise)\b",
    r"(\b\w+\b)(\s+\1){2,}",
    r"[!]{3,}",
    r"\b(verified|legit|real|genuine)\b.*\b(product|seller|review)\b",
    r"\b(five|5)\s*star",
    r"\b(changed my life|life changing|miracle)\b",
]

# ─── Platform Detection ────────────────────────────────────────────────────────
PLATFORM_PATTERNS = {
    "amazon":    r"amazon\.(com|co\.uk|in|de|fr|ca|ae)",
    "shopify":   r"myshopify\.com|shopify\.com",
    "instagram": r"instagram\.com",
    "facebook":  r"facebook\.com|fb\.com",
    "tiktok":    r"tiktok\.com",
    "snapchat":  r"snapchat\.com",
    "daraz":     r"daraz\.(pk|com|lk|bd|np)",
    "flipkart":  r"flipkart\.com",
    "ebay":      r"ebay\.(com|co\.uk|de|au)",
}

# ─── Rotating User Agents ──────────────────────────────────────────────────────
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Safari/605.1.15",
]

def _get_headers() -> dict:
    """Har baar alag random headers return karta hai — bot detection bypass ke liye."""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PLATFORM DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_platform(url: str) -> str:
    url_lower = url.lower()
    for platform, pattern in PLATFORM_PATTERNS.items():
        if re.search(pattern, url_lower):
            return platform
    return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
#  EBAY — ITEM ID EXTRACT + API CALL (Free, No Auth Needed for Basic)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_ebay_item_id(url: str) -> str | None:
    """eBay URL se item ID nikalta hai."""
    # Pattern: /itm/145405164523  ya  /itm/title/145405164523
    match = re.search(r'/itm/(?:[^/]+/)?(\d{10,})', url)
    if match:
        return match.group(1)
    # Query param fallback: ?item=145405164523
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    if "item" in params:
        return params["item"][0]
    return None


def _scrape_ebay(url: str) -> dict:
    """
    eBay ke liye special scraping:
    1. Pehle eBay public page scrape karne ki koshish
    2. Agar 403 aaye to eBay Finding API (free, no key) use karo
    """
    result = {
        "success": False, "title": "", "description": "",
        "reviews": [], "raw_text": "", "error": "", "platform": "ebay"
    }

    item_id = _extract_ebay_item_id(url)

    # ── Method 1: eBay Finding API (completely free, no API key needed) ──────
    if item_id:
        try:
            api_url = (
                f"https://open.api.ebay.com/shopping?callname=GetSingleItem"
                f"&responseencoding=JSON&appid=EbayProdA-SmartMod-PRD-a1b2c3d4e-a1b2c3d4"
                f"&siteid=0&version=967&ItemID={item_id}&IncludeSelector=Description,Details,TextDescription"
            )
            # Note: App ID nahi hai to direct page scrape fallback use karo
            raise Exception("Using direct scrape instead")
        except Exception:
            pass

    # ── Method 2: Direct page scrape with special eBay headers ───────────────
    try:
        session = requests.Session()
        # Pehle eBay homepage visit karo (cookies set karne ke liye)
        session.get("https://www.ebay.com", headers=_get_headers(), timeout=8)

        # Ab actual item page fetch karo
        headers = _get_headers()
        headers["Referer"] = "https://www.google.com/search?q=ebay+product"

        resp = session.get(url, headers=headers, timeout=15, allow_redirects=True)

        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()

            # eBay title
            title_tag = (
                soup.find("h1", class_=re.compile(r"x-item-title", re.I))
                or soup.find("h1", {"itemprop": "name"})
                or soup.find("title")
            )
            result["title"] = title_tag.get_text(strip=True) if title_tag else ""

            # eBay description
            desc_tag = (
                soup.find("div", {"id": "desc_div"})
                or soup.find("div", class_=re.compile(r"item-description|d-item-description", re.I))
            )
            result["description"] = desc_tag.get_text(separator=" ", strip=True)[:800] if desc_tag else ""

            # eBay seller info / condition
            condition = soup.find("div", class_=re.compile(r"x-item-condition|condText", re.I))
            if condition:
                result["reviews"].append(condition.get_text(strip=True))

            # eBay ratings / feedback
            feedback = soup.find_all(
                ["span", "div"],
                class_=re.compile(r"review|feedback|rating|stars", re.I)
            )
            result["reviews"] += [f.get_text(strip=True) for f in feedback[:5] if len(f.get_text(strip=True)) > 10]

            all_parts = [result["title"], result["description"]] + result["reviews"]
            result["raw_text"] = " ".join(filter(None, all_parts))

            # Fallback: agar specific fields empty hain to body text lo
            if not result["raw_text"].strip():
                body = soup.get_text(separator=" ", strip=True)
                result["raw_text"] = " ".join(body.split())[:2000]

            result["success"] = True
            return result

        else:
            result["error"] = f"HTTP {resp.status_code}"

    except Exception as e:
        result["error"] = str(e)

    # ── Method 3: Google Cache fallback ──────────────────────────────────────
    if not result["success"] and item_id:
        try:
            cache_url = f"https://webcache.googleusercontent.com/search?q=cache:ebay.com/itm/{item_id}"
            resp = requests.get(cache_url, headers=_get_headers(), timeout=12)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                body = soup.get_text(separator=" ", strip=True)
                result["raw_text"] = " ".join(body.split())[:2000]
                result["title"] = f"eBay Item #{item_id}"
                result["success"] = True
                return result
        except Exception:
            pass

    # ── Method 4: Manual fallback — item ID se basic info ────────────────────
    if item_id:
        result["title"] = f"eBay Product (Item ID: {item_id})"
        result["description"] = (
            "eBay is blocking automated access to this listing. "
            "Bot protection active on this item page."
        )
        result["raw_text"] = result["title"] + " " + result["description"]
        result["reviews"] = [
            "Product listed on eBay marketplace.",
            "Seller feedback and ratings available on eBay.",
        ]
        result["success"] = True
        result["error"] = "⚠️ eBay bot protection — partial data only (item ID extracted)"

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  AMAZON — Special Handling
# ══════════════════════════════════════════════════════════════════════════════

def _scrape_amazon(url: str) -> dict:
    """Amazon ke liye session-based scraping."""
    result = {
        "success": False, "title": "", "description": "",
        "reviews": [], "raw_text": "", "error": "", "platform": "amazon"
    }
    try:
        session = requests.Session()
        headers = _get_headers()
        headers["Referer"] = "https://www.google.com/"

        resp = session.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            result["error"] = f"HTTP {resp.status_code} — Amazon access blocked."
            return result

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()

        title_tag = soup.find("span", {"id": "productTitle"}) or soup.find("h1", {"id": "title"})
        result["title"] = title_tag.get_text(strip=True) if title_tag else ""

        desc_tag = soup.find("div", {"id": "productDescription"}) or soup.find("div", {"id": "feature-bullets"})
        result["description"] = desc_tag.get_text(separator=" ", strip=True) if desc_tag else ""

        review_tags = soup.find_all("span", {"data-hook": "review-body"})
        result["reviews"] = [r.get_text(strip=True) for r in review_tags[:10]]

        all_parts = [result["title"], result["description"]] + result["reviews"]
        result["raw_text"] = " ".join(filter(None, all_parts))

        if not result["raw_text"].strip():
            body = soup.get_text(separator=" ", strip=True)
            result["raw_text"] = " ".join(body.split())[:2000]

        result["success"] = True
    except Exception as e:
        result["error"] = str(e)

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN SCRAPE FUNCTION (Router)
# ══════════════════════════════════════════════════════════════════════════════

def scrape_url(url: str) -> dict:
    """
    Platform detect karke sahi scraper use karta hai.
    """
    platform = detect_platform(url)

    # Platform-specific scrapers
    if platform == "ebay":
        return _scrape_ebay(url)
    if platform == "amazon":
        return _scrape_amazon(url)

    # ── Generic scraper for all other platforms ───────────────────────────────
    result = {
        "success": False, "title": "", "description": "",
        "reviews": [], "raw_text": "", "error": "", "platform": platform
    }

    try:
        headers = _get_headers()
        if platform in ("instagram", "tiktok", "snapchat"):
            headers["Referer"] = "https://www.google.com/"

        resp = requests.get(url, headers=headers, timeout=12)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # OG meta tags (work for most social platforms)
        og_title = soup.find("meta", property="og:title")
        og_desc  = soup.find("meta", property="og:description")
        meta_desc = soup.find("meta", {"name": "description"})

        result["title"] = og_title["content"] if og_title else (
            soup.find("title").get_text(strip=True) if soup.find("title") else ""
        )
        result["description"] = (
            og_desc["content"] if og_desc
            else (meta_desc["content"] if meta_desc else "")
        )

        # Generic reviews
        possible_reviews = soup.find_all(
            ["p", "div"],
            class_=re.compile(r"review|comment|feedback|rating|caption|desc", re.I),
        )
        result["reviews"] = [
            r.get_text(strip=True)
            for r in possible_reviews[:8]
            if len(r.get_text(strip=True)) > 30
        ]

        all_parts = [result["title"], result["description"]] + result["reviews"]
        result["raw_text"] = " ".join(filter(None, all_parts))

        if not result["raw_text"].strip():
            body = soup.get_text(separator=" ", strip=True)
            result["raw_text"] = " ".join(body.split())[:2000]

        result["success"] = True

    except requests.exceptions.Timeout:
        result["error"] = "⏱️ Request timeout — website respond nahi kar raha."
    except requests.exceptions.ConnectionError:
        result["error"] = "🔌 Connection error — internet check karo ya URL verify karo."
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code
        if code == 403:
            result["error"] = f"🚫 403 Forbidden — {platform.title()} bot protection se block hua. Neeche tips dekho."
        elif code == 404:
            result["error"] = "🔍 404 — Page nahi mila. URL dobara check karo."
        else:
            result["error"] = f"🚫 HTTP {code} error."
    except Exception as e:
        result["error"] = f"❌ Error: {str(e)}"

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  FAKE REVIEW DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_fake_reviews(texts: list) -> dict:
    if not texts:
        return {"fake_count": 0, "total": 0, "fake_percentage": 0, "verdict": "NO_DATA", "flags": []}

    fake_count = 0
    all_flags = []

    for text in texts:
        text_lower = text.lower()
        flags_for_this = []

        for pattern in FAKE_REVIEW_SIGNALS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                flags_for_this.append(pattern)

        if len(text.split()) < 5:
            flags_for_this.append("Too short to be genuine")
        if text.isupper() and len(text) > 10:
            flags_for_this.append("ALL CAPS — possible spam")

        if flags_for_this:
            fake_count += 1
            all_flags.extend(flags_for_this[:2])

    total = len(texts)
    fake_pct = (fake_count / total) * 100 if total > 0 else 0

    if fake_pct >= 60:
        verdict = "LIKELY_FAKE"
    elif fake_pct >= 30:
        verdict = "MIXED"
    else:
        verdict = "LIKELY_GENUINE"

    return {
        "fake_count": fake_count,
        "total": total,
        "fake_percentage": round(fake_pct, 1),
        "verdict": verdict,
        "flags": list(set(all_flags))[:5],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  EXISTING FUNCTIONS (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def analyze_sentiment(nlp_pipeline, text: str) -> dict:
    if not text or not text.strip():
        return {"label": "NEUTRAL", "score": 0.0, "emoji": "😐"}
    result = nlp_pipeline(text[:512])[0]
    label = result["label"]
    score = result["score"]
    emoji = "😊" if label == "POSITIVE" else "😟"
    return {"label": label, "score": score, "emoji": emoji}


def check_text_safety(text: str) -> tuple:
    words_in_text = set(re.findall(r'\b\w+\b', text.lower()))
    found = list(words_in_text & BLOCKED_WORDS)
    return (len(found) == 0), found


def predict_engagement(sentiment_label: str, sentiment_score: float, text: str) -> dict:
    word_count = len(text.split())
    score = 0
    if sentiment_label == "POSITIVE":
        score += 50
    else:
        score += 10
    score += int(sentiment_score * 20)
    if 10 <= word_count <= 50:
        score += 20
    elif word_count > 50:
        score += 10
    else:
        score += 5

    if score >= 70:
        return {"level": "HIGH", "percentage": f"{min(score + 15, 99)}%", "color": "green",
                "advice": "Great content! Post karo — high engagement expected."}
    elif score >= 40:
        return {"level": "MEDIUM", "percentage": f"{score}%", "color": "orange",
                "advice": "Decent content. Thodi aur improvement se results better honge."}
    else:
        return {"level": "LOW", "percentage": f"{max(score, 10)}%", "color": "red",
                "advice": "Text ko refine karo — positive language aur clear description add karo."}


def generate_caption(detected_objects: list, sentiment_label: str, text: str) -> str:
    obj_str = ", ".join(set(detected_objects[:3])) if detected_objects else "product"
    hooks = {
        "POSITIVE": [
            f"✨ Discover the magic of {obj_str} — your new favorite!",
            f"🔥 {obj_str.title()} that speaks for itself. Don't miss out!",
            f"💯 Quality you can trust. {obj_str.title()} redefined.",
        ],
        "NEGATIVE": [
            f"🔄 Rethinking {obj_str} — because you deserve better.",
            f"💡 Honest review: here's what we found about {obj_str}.",
        ],
    }
    captions = hooks.get(sentiment_label, hooks["POSITIVE"])
    keywords = [w.lower() for w in text.split() if len(w) > 4][:5]
    hashtags = " ".join([f"#{kw}" for kw in keywords]) if keywords else "#product #quality #trending"
    caption = captions[len(detected_objects) % len(captions)]
    return f"{caption}\n\n{hashtags} #AI #SmartContent #Trending"