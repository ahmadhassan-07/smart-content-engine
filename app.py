import streamlit as st
from PIL import Image
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from utils.vision_helper import run_object_detection, check_content_safety
from utils.nlp_helper import (
    analyze_sentiment,
    check_text_safety,
    predict_engagement,
    generate_caption,
    scrape_url,
    detect_fake_reviews,
    detect_platform,
)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Smart Moderator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4F46E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #6B7280;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #374151;
        border-left: 4px solid #4F46E5;
        padding-left: 0.6rem;
        margin: 1rem 0;
    }
    .platform-badge {
        display: inline-block;
        background: #EEF2FF;
        color: #4F46E5;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    /* Platform buttons in sidebar */
    div[data-testid="stButton"] button {
        width: 100%;
        text-align: left;
        background: transparent;
        border: 1px solid #374151;
        color: inherit;
        border-radius: 8px;
        padding: 0.3rem 0.6rem;
        margin-bottom: 2px;
        font-size: 0.85rem;
        transition: all 0.2s;
    }
    div[data-testid="stButton"] button:hover {
        background: #EEF2FF;
        border-color: #4F46E5;
        color: #4F46E5;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🤖 AI Smart Content Moderator</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Multi-Modal AI Engine — Computer Vision · NLP · Fake Review Detection · Link Analyzer</p>',
    unsafe_allow_html=True,
)

# ─── Initialize session state ─────────────────────────────────────────────────
for key in ["cv_done", "nlp_done", "link_analyzed", "caption",
            "annotated_img", "detected_objects", "obj_counts",
            "is_safe", "safety_msg", "sentiment", "text_safe",
            "blocked_words", "engagement", "user_text_saved",
            "link_scraped", "link_sentiment", "link_text_safe",
            "link_blocked", "link_engagement", "fake_result",
            "selected_platform_url"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    confidence_threshold = st.slider(
        "CV Detection Confidence",
        min_value=0.1, max_value=1.0, value=0.4, step=0.05,
    )
    st.markdown("---")
    st.info(
        "**Models Used:**\n"
        "- 🎯 YOLOv8n (Object Detection)\n"
        "- 🧠 DistilBERT (Sentiment)\n"
        "- 🔍 BeautifulSoup (Scraper)\n"
        "- 📊 Custom ML Scoring\n"
        "- ✍️ Rule-based Caption Gen"
    )
    st.markdown("---")

    # ── Clickable Platform Buttons ────────────────────────────────────────────
    st.markdown("**🌐 Supported Platforms** *(click to load example)*")

    platform_examples = {
        "🛒 Amazon":    "https://www.amazon.com/dp/B08N5WRWNW",
        "🛍️ Shopify":  "https://allbirds.com/products/mens-wool-runners",
        "🛍️ Daraz":    "https://www.daraz.pk/products/",
        "📘 Facebook":  "https://www.facebook.com/",
        "📸 Instagram": "https://www.instagram.com/p/",
        "🎵 TikTok":    "https://www.tiktok.com/@tiktok",
        "👻 Snapchat":  "https://www.snapchat.com/",
        "🛒 eBay":      "https://www.ebay.com/itm/145405164523",
        "🌐 Any URL":   "https://",
    }

    for label, example_url in platform_examples.items():
        if st.button(label, key=f"plat_{label}"):
            st.session_state["selected_platform_url"] = example_url
            # Clear previous link analysis
            st.session_state["link_analyzed"] = None
            st.session_state["link_scraped"] = None

    st.caption("⬆️ Click karo → Tab 2 mein URL auto-fill ho jayega")
    st.markdown("---")
    st.caption("First run par models auto-download honge (~150MB) ⏳")

# ─── Model Loading ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models — please wait...")
def load_models():
    from ultralytics import YOLO
    from transformers import pipeline
    cv_model = YOLO("yolov8n.pt")
    nlp_model = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )
    return cv_model, nlp_model

cv_model, nlp_model = load_models()

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🖼️  Image & Text Analyzer", "🔗  Link / URL Analyzer"])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — Image & Text
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    # ── LEFT: Computer Vision ────────────────────────────────────────────────
    with col1:
        st.markdown('<p class="section-title">📸 Computer Vision & Media Analysis</p>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Image upload karo (JPG / PNG / JPEG)",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded_file is not None:
            pil_image = Image.open(uploaded_file).convert("RGB")
            st.image(pil_image, caption="📤 Uploaded Image", use_column_width=True)

            if st.button("🔍 Analyze Image", use_container_width=True):
                with st.spinner("YOLOv8 objects detect kar raha hai..."):
                    annotated_img, detected_objects, obj_counts = run_object_detection(
                        cv_model, pil_image, confidence_threshold
                    )
                    is_safe, safety_msg = check_content_safety(detected_objects)

                st.session_state["cv_done"]         = True
                st.session_state["annotated_img"]   = annotated_img
                st.session_state["detected_objects"] = detected_objects
                st.session_state["obj_counts"]      = obj_counts
                st.session_state["is_safe"]         = is_safe
                st.session_state["safety_msg"]      = safety_msg

        if st.session_state.get("cv_done"):
            st.image(st.session_state["annotated_img"], caption="🎯 Detected Objects", use_column_width=True)
            obj_counts = st.session_state.get("obj_counts", {})
            if obj_counts:
                st.markdown("**Detected Objects:**")
                cols_obj = st.columns(min(len(obj_counts), 3))
                for i, (obj, count) in enumerate(obj_counts.items()):
                    cols_obj[i % 3].metric(label=obj.title(), value=count)
            else:
                st.info("Koi object detect nahi hua — confidence slider kam karo.")

            if st.session_state.get("is_safe"):
                st.success(st.session_state["safety_msg"])
            else:
                st.error(st.session_state["safety_msg"])

    # ── RIGHT: NLP + ML ──────────────────────────────────────────────────────
    with col2:
        st.markdown('<p class="section-title">📝 NLP Text Analytics & Engagement Prediction</p>', unsafe_allow_html=True)

        user_text = st.text_area(
            "Product description ya user review likho:",
            value="This product is absolutely amazing! The quality is top-notch and delivery was super fast. Highly recommended to everyone!",
            height=120,
        )
        st.caption(f"Word count: {len(user_text.split()) if user_text else 0}")

        if st.button("🧠 Analyze Text & Predict Engagement", use_container_width=True):
            if not user_text.strip():
                st.warning("Pehle kuch text likho!")
            else:
                with st.spinner("DistilBERT analyze kar raha hai..."):
                    sentiment_result = analyze_sentiment(nlp_model, user_text)
                    is_text_safe, blocked_found = check_text_safety(user_text)
                    engagement = predict_engagement(
                        sentiment_result["label"], sentiment_result["score"], user_text
                    )

                st.session_state["nlp_done"]        = True
                st.session_state["sentiment"]       = sentiment_result
                st.session_state["text_safe"]       = is_text_safe
                st.session_state["blocked_words"]   = blocked_found
                st.session_state["engagement"]      = engagement
                st.session_state["user_text_saved"] = user_text

        if st.session_state.get("nlp_done"):
            sentiment  = st.session_state["sentiment"]
            engagement = st.session_state["engagement"]

            st.markdown("**🎭 Sentiment Analysis:**")
            sc1, sc2 = st.columns(2)
            sc1.metric("Sentiment", f"{sentiment['emoji']} {sentiment['label']}")
            sc2.metric("Confidence", f"{sentiment['score']:.0%}")

            if sentiment["label"] == "POSITIVE":
                st.success("✅ Positive sentiment — safe for posting!")
            else:
                st.error("⚠️ Negative sentiment — review karo before posting.")

            st.markdown("**🛡️ Text Safety:**")
            if st.session_state["text_safe"]:
                st.success("✅ No abusive or spam words found.")
            else:
                st.error(f"🚫 Flagged words: `{', '.join(st.session_state['blocked_words'])}`")

            st.markdown("---")
            st.markdown('<p class="section-title">📊 ML Engagement Prediction</p>', unsafe_allow_html=True)
            eng = engagement["level"]
            if eng == "HIGH":
                st.success(f"🚀 **{eng} — {engagement['percentage']} estimated reach**")
            elif eng == "MEDIUM":
                st.warning(f"📈 **{eng} — {engagement['percentage']} estimated reach**")
            else:
                st.error(f"📉 **{eng} — {engagement['percentage']} estimated reach**")
            st.info(f"💡 {engagement['advice']}")

    # ── Caption Generator ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">✍️ AI Caption & Hashtag Generator</p>', unsafe_allow_html=True)
    g1, g2 = st.columns([1, 2])
    with g1:
        gen_btn = st.button("🪄 Generate Caption", use_container_width=True)
    with g2:
        if gen_btn:
            if not st.session_state.get("cv_done") and not st.session_state.get("nlp_done"):
                st.warning("Pehle image ya text analyze karo!")
            else:
                objects  = st.session_state.get("detected_objects") or []
                sent_lbl = (st.session_state.get("sentiment") or {}).get("label", "POSITIVE")
                saved_txt = st.session_state.get("user_text_saved") or user_text
                caption  = generate_caption(objects, sent_lbl, saved_txt)
                st.session_state["caption"] = caption

    if st.session_state.get("caption"):
        st.text_area("📋 Generated Caption:", value=st.session_state["caption"], height=120)
        st.success("Caption ready! Copy karo aur post karo. 🎉")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — LINK / URL ANALYZER
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">🔗 Post / Product Link Analyzer</p>', unsafe_allow_html=True)
    st.write(
        "Kisi bhi platform ka link paste karo — Amazon, Daraz, Facebook, Instagram, "
        "TikTok, Snapchat, Shopify, eBay — AI poori post/product ko analyze karega."
    )

    # ── Auto-fill from sidebar click ──────────────────────────────────────────
    prefill_url = st.session_state.get("selected_platform_url") or ""

    # ── URL Input ─────────────────────────────────────────────────────────────
    url_input = st.text_input(
        "🔗 URL / Link paste karo:",
        value=prefill_url,
        placeholder="https://www.amazon.com/dp/... ya https://www.daraz.pk/products/...",
        key="url_input_field",
    )

    # ── Platform Badge ────────────────────────────────────────────────────────
    platform_icons = {
        "amazon":    "🛒 Amazon",
        "shopify":   "🛍️ Shopify",
        "instagram": "📸 Instagram",
        "facebook":  "📘 Facebook",
        "tiktok":    "🎵 TikTok",
        "snapchat":  "👻 Snapchat",
        "daraz":     "🛍️ Daraz",
        "flipkart":  "🛒 Flipkart",
        "ebay":      "🛒 eBay",
        "unknown":   "🌐 Website",
    }

    if url_input and url_input.startswith("http"):
        detected_plat = detect_platform(url_input)
        badge_label   = platform_icons.get(detected_plat, "🌐 Website")
        st.markdown(f'<span class="platform-badge">{badge_label} detected</span>', unsafe_allow_html=True)

    # ── Analyze Button ────────────────────────────────────────────────────────
    analyze_btn = st.button("🚀 Analyze Link", key="link_analyze_btn")

    if analyze_btn:
        # Validation
        if not url_input or not url_input.strip():
            st.error("❌ Koi URL nahi diya — URL field mein link paste karo.")
            st.stop()

        clean_url = url_input.strip()
        if not clean_url.startswith("http"):
            st.error("❌ Valid URL chahiye — https:// ya http:// se shuru hona chahiye.")
            st.stop()

        # ── Scraping ──────────────────────────────────────────────────────────
        with st.spinner("🔍 URL se content scrape ho raha hai..."):
            scraped = scrape_url(clean_url)

        # ── Scraping Failed completely ─────────────────────────────────────
        if not scraped.get("success"):
            st.error(f"**Scraping Failed:** {scraped.get('error', 'Unknown error')}")
            st.info(
                "💡 **Tips:**\n"
                "- Amazon/Daraz product pages best kaam karte hain\n"
                "- Instagram/TikTok login ke baghair restrict karte hain\n"
                "- eBay kabhi kabhi bot protection se block karta hai\n"
                "- Koi aur URL try karo"
            )
            st.stop()

        # ── Partial data warning (eBay case) ──────────────────────────────
        if scraped.get("error") and scraped.get("success"):
            st.warning(f"⚠️ {scraped['error']}")

        # ── Check if we have analyzable text ─────────────────────────────
        raw_text = scraped.get("raw_text", "").strip()

        if not raw_text:
            st.warning(
                "⚠️ Page se readable text nahi mila.\n\n"
                "Ye platform JavaScript se content load karta hai "
                "(Instagram, TikTok) — static scraper access nahi kar sakta.\n\n"
                "Amazon ya Daraz ki product page try karo."
            )
            # Show what was scraped at least
            with st.expander("🔎 Scraped Data (Raw)", expanded=True):
                st.json(scraped)
            st.stop()

        # ── Run NLP Analysis ──────────────────────────────────────────────
        with st.spinner("🧠 AI content analyze kar raha hai..."):
            link_sentiment = analyze_sentiment(nlp_model, raw_text)
            link_text_safe, link_blocked = check_text_safety(raw_text)
            link_engagement = predict_engagement(
                link_sentiment["label"], link_sentiment["score"], raw_text
            )
            reviews_to_check = scraped.get("reviews") or [raw_text]
            fake_result = detect_fake_reviews(reviews_to_check)

        # Store results
        st.session_state["link_scraped"]    = scraped
        st.session_state["link_sentiment"]  = link_sentiment
        st.session_state["link_text_safe"]  = link_text_safe
        st.session_state["link_blocked"]    = link_blocked
        st.session_state["link_engagement"] = link_engagement
        st.session_state["fake_result"]     = fake_result
        st.session_state["link_analyzed"]   = True

    # ══════════════════════════════════════════════════════════════════════════
    #  RESULTS — Only show if analysis was done
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state.get("link_analyzed") is True:

        scraped         = st.session_state["link_scraped"]
        link_sentiment  = st.session_state["link_sentiment"]
        link_engagement = st.session_state["link_engagement"]
        fake_result     = st.session_state["fake_result"]
        raw_text        = scraped.get("raw_text", "")

        st.markdown("---")
        st.success("✅ Analysis complete!")

        # ── Scraped Content Preview ───────────────────────────────────────
        with st.expander("📄 Scraped Content Preview", expanded=False):
            if scraped.get("title"):
                st.markdown(f"**Title:** {scraped['title']}")
            if scraped.get("description"):
                st.markdown(f"**Description:** {scraped['description'][:600]}")
            if scraped.get("reviews"):
                st.markdown(f"**Reviews Found:** {len(scraped['reviews'])}")
                for i, rev in enumerate(scraped["reviews"][:4], 1):
                    st.caption(f"Review {i}: {rev[:200]}")
            if not scraped.get("title") and not scraped.get("description"):
                st.text(raw_text[:400] + "...")

        # ── 3 Result Cards ────────────────────────────────────────────────
        r1, r2, r3 = st.columns(3)

        with r1:
            st.markdown("### 🎭 Sentiment")
            emoji = link_sentiment["emoji"]
            label = link_sentiment["label"]
            conf  = link_sentiment["score"]
            if label == "POSITIVE":
                st.success(f"**{emoji} {label}**\n\nConfidence: {conf:.0%}")
            else:
                st.error(f"**{emoji} {label}**\n\nConfidence: {conf:.0%}")

        with r2:
            st.markdown("### 🕵️ Review Authenticity")
            verdict  = fake_result["verdict"]
            fake_pct = fake_result["fake_percentage"]
            total    = fake_result["total"]
            flagged  = fake_result["fake_count"]

            if verdict == "LIKELY_FAKE":
                st.error(
                    f"**🚨 LIKELY FAKE**\n\n"
                    f"{fake_pct}% reviews suspicious\n\n"
                    f"({flagged}/{total} flagged)"
                )
            elif verdict == "MIXED":
                st.warning(
                    f"**⚠️ MIXED**\n\n"
                    f"{fake_pct}% reviews suspicious\n\n"
                    f"({flagged}/{total} flagged)"
                )
            elif verdict == "NO_DATA":
                st.info("**ℹ️ NO REVIEWS**\n\nKoi dedicated review section nahi mila.")
            else:
                st.success(
                    f"**✅ LIKELY GENUINE**\n\n"
                    f"Only {fake_pct}% suspicious\n\n"
                    f"({flagged}/{total} flagged)"
                )

        with r3:
            st.markdown("### 📊 Predicted Engagement")
            eng = link_engagement["level"]
            pct = link_engagement["percentage"]
            if eng == "HIGH":
                st.success(f"**🚀 {eng}**\n\n{pct} estimated reach")
            elif eng == "MEDIUM":
                st.warning(f"**📈 {eng}**\n\n{pct} estimated reach")
            else:
                st.error(f"**📉 {eng}**\n\n{pct} estimated reach")

        # ── Detailed Breakdown ────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<p class="section-title">📋 Detailed Analysis Report</p>', unsafe_allow_html=True)

        det1, det2 = st.columns(2)

        with det1:
            st.markdown("**🛡️ Content Safety:**")
            if st.session_state["link_text_safe"]:
                st.success("✅ No abusive/spam language found.")
            else:
                blocked = st.session_state["link_blocked"]
                st.error(f"🚫 Flagged words: `{', '.join(blocked)}`")

            if fake_result.get("flags"):
                st.markdown("**🔍 Suspicious Signals Found:**")
                for flag in fake_result["flags"][:4]:
                    display = flag if len(flag) < 70 else flag[:67] + "..."
                    st.caption(f"• {display}")

        with det2:
            st.markdown("**⚖️ Overall Verdict:**")

            is_positive    = link_sentiment["label"] == "POSITIVE"
            is_genuine     = fake_result["verdict"] in ("LIKELY_GENUINE", "MIXED", "NO_DATA")
            is_safe_content = st.session_state["link_text_safe"]

            if is_positive and is_genuine and is_safe_content:
                st.success(
                    "✅ **TRUSTED CONTENT**\n\n"
                    "Sentiment positive hai, reviews genuine lagte hain, "
                    "aur koi safety issue nahi."
                )
            elif not is_positive and fake_result["verdict"] == "LIKELY_FAKE":
                st.error(
                    "🚨 **HIGH RISK CONTENT**\n\n"
                    "Negative sentiment + fake reviews — "
                    "is post/product se sawdhaan rahein!"
                )
            elif fake_result["verdict"] == "LIKELY_FAKE":
                st.error(
                    "🚨 **FAKE REVIEWS DETECTED**\n\n"
                    "Reviews mein manipulation ke signs hain. "
                    "Purchase karne se pehle soch lein."
                )
            elif not is_positive:
                st.warning(
                    "⚠️ **NEGATIVE CONTENT**\n\n"
                    "Sentiment negative hai — is post/product ke "
                    "baare mein negative feedback ho sakta hai."
                )
            else:
                st.warning(
                    "⚠️ **MIXED SIGNALS**\n\n"
                    "Kuch positive, kuch concerning. "
                    "Extra research karna recommended hai."
                )

            st.info(f"💡 {link_engagement['advice']}")

        # ── Text Sample ────────────────────────────────────────────────────
        with st.expander("📝 Analyzed Text Sample", expanded=False):
            st.text(raw_text[:800] + ("..." if len(raw_text) > 800 else ""))

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("🤖 Built with Streamlit · YOLOv8 · DistilBERT · BeautifulSoup | Smart Content Engine v2.1")