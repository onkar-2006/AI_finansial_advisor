import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
from langchain_groq import ChatGroq
import plotly.graph_objects as go

# Setup Streamlit UI
st.set_page_config(page_title="📈 Stock Predictor & Chatbot", layout="wide")
st.title("📈 AI-Powered finansial advisor")

# Sidebar for Stock Prediction Inputs
st.sidebar.header("📊 Stock Prediction")
stock_symbol = st.sidebar.text_input("📌 Enter Stock Symbol:", "AAPL", key="stock_input")
forecast_days = st.sidebar.slider("🔮 Select Forecast Period:", 5, 30, 10)
api_key = st.sidebar.text_input("🔑 Enter Groq API Key", type="password", key="api_key")

# Initialize session states for chat history and prediction
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "prediction_summary" not in st.session_state:
    st.session_state.prediction_summary = ""

# **Stock Prediction Logic**
if st.sidebar.button("🔍 Predict Stock Prices"):
    with st.spinner("Fetching stock data..."):
        stock = yf.Ticker(stock_symbol)
        df = stock.history(period="5y")[["Close"]].reset_index()

        if df.empty:
            st.error("No stock data found. Please check the stock symbol.")
        else:
            df.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)
            df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)

            # Train Prophet Model
            model = Prophet(yearly_seasonality=True, daily_seasonality=False)
            model.fit(df)

            # Create future dataframe
            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)

            # Display Predictions
            st.subheader(f"📉 {stock_symbol.upper()} - {forecast_days}-Day Forecast")

            # ✅ **Enhanced Interactive Stock Chart**
            fig = go.Figure()

            # Add historical data as a line chart
            fig.add_trace(go.Scatter(x=df["ds"], y=df["y"], mode="lines", name="Actual Price", line=dict(color="blue")))

            # Add predicted data as a line chart
            fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Predicted Price",
                                     line=dict(color="red")))

            # Add confidence intervals
            fig.add_trace(go.Scatter(
                x=forecast["ds"], y=forecast["yhat_upper"],
                mode="lines", line=dict(width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=forecast["ds"], y=forecast["yhat_lower"],
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(255,0,0,0.2)",
                name="Confidence Interval"
            ))

            fig.update_layout(title=f"{stock_symbol.upper()} Stock Price Prediction", xaxis_title="Date",
                              yaxis_title="Price (USD)")
            st.plotly_chart(fig, use_container_width=True)

            # ✅ **Candlestick Chart for Better Visualization**
            st.subheader("📊 Historical Candlestick Chart")

            df_candle = stock.history(period="6mo")  # Get 6 months of historical data

            fig_candle = go.Figure(data=[
                go.Candlestick(
                    x=df_candle.index,
                    open=df_candle["Open"],
                    high=df_candle["High"],
                    low=df_candle["Low"],
                    close=df_candle["Close"],
                    name="Candlestick"
                )
            ])

            fig_candle.update_layout(title=f"{stock_symbol.upper()} - 6-Month Candlestick Chart", xaxis_title="Date",
                                     yaxis_title="Stock Price (USD)")
            st.plotly_chart(fig_candle, use_container_width=True)

            # ✅ **Moving Averages Chart**
            st.subheader(f"📊 {stock_symbol.upper()} - Moving Averages Chart")
            df["MA_50"] = df["y"].rolling(window=50).mean()
            df["MA_200"] = df["y"].rolling(window=200).mean()

            ma_fig = go.Figure()
            ma_fig.add_trace(
                go.Scatter(x=df["ds"], y=df["y"], mode="lines", name="Actual Price", line=dict(color="blue")))
            ma_fig.add_trace(
                go.Scatter(x=df["ds"], y=df["MA_50"], mode="lines", name="50-Day MA", line=dict(color="orange")))
            ma_fig.add_trace(
                go.Scatter(x=df["ds"], y=df["MA_200"], mode="lines", name="200-Day MA", line=dict(color="purple")))

            ma_fig.update_layout(title="Stock Moving Averages", xaxis_title="Date", yaxis_title="Price (USD)")
            st.plotly_chart(ma_fig, use_container_width=True)

            # ✅ **Volume Traded Chart**
            st.subheader(f"📊 {stock_symbol.upper()} - Volume Traded")
            df_vol = stock.history(period="5y").reset_index()
            volume_fig = go.Figure()
            volume_fig.add_trace(
                go.Bar(x=df_vol["Date"], y=df_vol["Volume"], name="Trading Volume", marker=dict(color="blue")))

            volume_fig.update_layout(title="Stock Trading Volume", xaxis_title="Date", yaxis_title="Volume")
            st.plotly_chart(volume_fig, use_container_width=True)

            # ✅ **Bollinger Bands Chart**
            st.subheader(f"📊 {stock_symbol.upper()} - Bollinger Bands")
            df["MA_20"] = df["y"].rolling(window=20).mean()
            df["Upper_Band"] = df["MA_20"] + (df["y"].rolling(window=20).std() * 2)
            df["Lower_Band"] = df["MA_20"] - (df["y"].rolling(window=20).std() * 2)

            bb_fig = go.Figure()
            bb_fig.add_trace(
                go.Scatter(x=df["ds"], y=df["y"], mode="lines", name="Actual Price", line=dict(color="blue")))
            bb_fig.add_trace(
                go.Scatter(x=df["ds"], y=df["Upper_Band"], mode="lines", name="Upper Band", line=dict(color="green")))
            bb_fig.add_trace(
                go.Scatter(x=df["ds"], y=df["Lower_Band"], mode="lines", name="Lower Band", line=dict(color="red")))

            bb_fig.update_layout(title="Bollinger Bands", xaxis_title="Date", yaxis_title="Price (USD)")
            st.plotly_chart(bb_fig, use_container_width=True)

            # ✅ **RSI (Relative Strength Index)**
            st.subheader(f"📊 {stock_symbol.upper()} - RSI Indicator")
            delta = df["y"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df["RSI"] = 100 - (100 / (1 + rs))

            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=df["ds"], y=df["RSI"], mode="lines", name="RSI", line=dict(color="purple")))

            rsi_fig.update_layout(title="Relative Strength Index (RSI)", xaxis_title="Date", yaxis_title="RSI Value",
                                  yaxis=dict(range=[0, 100]))
            st.plotly_chart(rsi_fig, use_container_width=True)

            # Generate AI Explanation
            if api_key:
                llm = ChatGroq(model="llama3-8b-8192", api_key=api_key)

                prediction_summary = f"""
                The AI-driven forecast for {stock_symbol.upper()} over the next {forecast_days} days is:

                - Expected price range: ${round(forecast['yhat'].min(), 2)} - ${round(forecast['yhat'].max(), 2)}
                - The trend suggests {'an upward' if forecast['yhat'].iloc[-1] > forecast['yhat'].iloc[0] else 'a downward'} movement.
                """

                # Store AI explanation in session state
                st.session_state.prediction_summary = prediction_summary

                # Store AI response in chat history
                st.session_state.chat_history.append(("AI", prediction_summary))

                st.subheader("🧠 AI Explanation")
                st.write(prediction_summary)

# **Two-Sided Chat Interface (Like a Chatbot)**
st.subheader("💬 Stock Chatbot")

# **Create a chat container with right-aligned messages**
chat_container = st.container()

# **Display chat history**
with chat_container:
    for role, message in st.session_state.chat_history:
        if role == "User":
            st.markdown(f"<div style='text-align: right; background: #DCF8C6; padding: 8px; border-radius: 10px; margin: 5px 0;'>👤 <b>You:</b> {message}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align: left; background: #ECECEC; padding: 8px; border-radius: 10px; margin: 5px 0;'>🤖 <b>AI:</b> {message}</div>", unsafe_allow_html=True)

# **User Input Box for Chat**
user_question = st.text_input("Ask AI a follow-up question about the stock prediction:", key="user_input")

# **Process User Query**
if user_question and api_key:
    if "last_question" not in st.session_state or st.session_state.last_question != user_question:
        llm = ChatGroq(model="llama3-8b-8192", api_key=api_key)

        # System prompt to restrict AI responses to stock-related answers only
        system_prompt = f"""
        You are a financial assistant specializing in stock price forecasting.
        Your knowledge is based on Prophet-based stock predictions.
        Always refer to the given forecast and do NOT answer unrelated questions.

        Here is the stock prediction context:
        {st.session_state.prediction_summary}

        Only answer user questions related to this prediction.
        """

        # AI Response to user question
        response = llm.invoke(f"{system_prompt}\nUser: {user_question}\nAI:")
        ai_reply = response.content if hasattr(response, "content") else str(response)

        # Store conversation in chat history
        st.session_state.chat_history.append(("User", user_question))
        st.session_state.chat_history.append(("AI", ai_reply))

        # Save last question to prevent infinite loops
        st.session_state.last_question = user_question

        # Refresh chat UI
        st.rerun()
