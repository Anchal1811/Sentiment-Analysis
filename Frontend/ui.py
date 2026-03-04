import streamlit as st
import requests

# 1. Page Configuration
st.set_page_config(
    page_title="Swiggy Sentiment Dashboard", 
    page_icon="🍜",
    layout="wide"
)

# Professional Header
st.title("🍜 Swiggy Restaurant Manager Insights")
st.markdown("""
    *Analyze customer feedback in real-time using our Bi-LSTM Deep Learning engine to improve service quality and customer retention.*
""")
st.markdown("---")

# 2. Layout Columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Analyze New Customer Review")
    review_text = st.text_area(
        "Review Content", 
        placeholder="e.g., The butter chicken was amazing, but the delivery took 1 hour...",
        height=150
    )

with col2:
    st.subheader("Metadata")
    food_item = st.text_input("Food Item Name (Optional)", placeholder="e.g., Butter Chicken")
    st.info("Tip: Mentioning the food item helps in tracking specific kitchen issues.")

# 3. Analysis Logic
if st.button("Run Sentiment Analysis", use_container_width=True):
    if not review_text.strip():
        st.warning("⚠️ Please enter a review before running the analysis.")
    else:
        with st.spinner("🧠 Bi-LSTM Model is analyzing context..."):
            try:
                # Docker Service Discovery URL
                # 'backend' is the service name defined in docker-compose.yml
                BACKEND_URL = "http://backend:8001/predict"
                payload = {"text": review_text}
                
                # API Call with Timeout
                response = requests.post(BACKEND_URL, json=payload, timeout=15)
                response.raise_for_status() 
                
                data = response.json()

                # 4. FIXED LOGIC: Explicit Case-Insensitive Matching
                # This ensures "Negative" or "negative" both trigger the correct UI
                sentiment_raw = data.get('sentiment', 'Unknown')
                sentiment = sentiment_raw.strip().lower()
                confidence = data.get('confidence', 0.0)

                st.markdown("### Analysis Result")
                
                if sentiment == "positive":
                    st.success(f"### ✅ Positive Sentiment ({confidence:.2%})")
                    st.balloons()
                    st.markdown("**Insight:** The customer is happy! Consider 'liking' this review on the Swiggy app to build loyalty.")
                
                elif sentiment == "negative":
                    st.error(f"### ❌ Negative Sentiment ({confidence:.2%})")
                    # Actionable business logic
                    st.warning(f"🚨 **Action Required**: High priority intervention for **{food_item if food_item else 'this order'}**.")
                    st.markdown("""
                        **Suggested Actions:**
                        * Flag this to the kitchen/delivery team.
                        * Offer a discount coupon via Swiggy chat to prevent a 1-star rating.
                    """)
                
                else:
                    st.info(f"### ℹ️ {sentiment_raw.title()} Sentiment ({confidence:.2%})")
                    st.write("The model found this review to be neutral or mixed.")

            except requests.exceptions.ConnectionError:
                st.error("🚨 **Backend Offline**: The UI cannot find the API. Ensure your Docker containers are running.")
            except Exception as e:
                st.error(f"Failed to process sentiment: {e}")

# 5. Footer
st.markdown("---")
st.caption("v1.2.0 | Engine: Bi-LSTM | Infrastructure: FastAPI + Docker")