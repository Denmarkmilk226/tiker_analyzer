"""
Stock Analyzer v3 - Multi-Stock Comparison with Gemini AI (Streamlit Version)
Natural language ticker selection + side-by-side comparison
"""
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
import json
import os
import streamlit as st

# Configure matplotlib backend for web compatibility
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Gemini API import (optional)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# TensorFlow import (optional)
try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    LSTM_AVAILABLE = True
except:
    LSTM_AVAILABLE = False

# === CONFIGURATION ===
MIN_REVERSE_PERCENT = 0.02
SEQUENCE_LENGTH = 10
RSI_OVERSOLD = 30
VOLUME_MULTIPLIER = 1.5
DEFAULT_PREDICTION_DAYS = 5

# === GEMINI SETTINGS FILE ===
SETTINGS_FILE = "gemini_settings.json"

def load_gemini_settings():
    """Load Gemini API settings from file"""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"api_key": "", "enabled": True}

def save_gemini_settings(settings):
    """Save Gemini API settings to file"""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        return True
    except:
        return False


def get_tickers_from_gemini(user_query, api_key):
    """
    Use Gemini to convert natural language to stock tickers
    Returns: (success, tickers_list, error_message)
    """
    if not GEMINI_AVAILABLE:
        return False, [], "Gemini library not installed. Run: pip install google-generativeai"

    if not api_key or api_key.strip() == "":
        return False, [], "API key not provided"

    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')

        # Create prompt
        prompt = f"""You are a stock ticker expert. Convert the following natural language stock request into a JSON array of valid US stock ticker symbols.

Rules:
1. Return ONLY a valid JSON array of ticker symbols
2. Use uppercase ticker symbols
3. Include only major US stocks (no OTC, no foreign exchanges)
4. Maximum 20 tickers
5. If request is vague, use common interpretation
6. No explanations, just the JSON array

Examples:
User: "Compare Apple and Microsoft"
Response: ["AAPL", "MSFT"]

User: "Top 5 tech companies"
Response: ["AAPL", "MSFT", "GOOGL", "META", "AMZN"]

User: "FAANG stocks"
Response: ["META", "AAPL", "AMZN", "NFLX", "GOOGL"]

Now convert this request:
User: "{user_query}"
Response:"""

        # Call Gemini API
        response = model.generate_content(prompt)
        response_text = response.text.strip()

        # Parse JSON
        # Remove markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        tickers = json.loads(response_text)

        # Validate
        if not isinstance(tickers, list) or len(tickers) == 0:
            return False, [], "Invalid response format from Gemini"

        # Clean and validate tickers
        valid_tickers = []
        for ticker in tickers[:20]:  # Max 20
            if isinstance(ticker, str) and ticker.strip():
                valid_tickers.append(ticker.strip().upper())

        if len(valid_tickers) == 0:
            return False, [], "No valid tickers found in response"

        return True, valid_tickers, None

    except Exception as e:
        error_msg = str(e)

        # Check for rate limit
        if "quota" in error_msg.lower() or "limit" in error_msg.lower() or "429" in error_msg:
            return False, [], "⚠️ Gemini API usage limit exceeded. Please try again later or use manual input."

        # Check for API key error
        if "api" in error_msg.lower() and ("key" in error_msg.lower() or "invalid" in error_msg.lower()):
            return False, [], "Invalid API key. Please check your Gemini API key."

        return False, [], f"Gemini API error: {error_msg}"


def fetch_financial_data(ticker, progress_callback=None):
    """Fetch quarterly financial statement data"""
    try:
        stock = yf.Ticker(ticker)
        income_stmt = stock.quarterly_income_stmt.T
        balance_sheet = stock.quarterly_balance_sheet.T
        info = stock.info

        if income_stmt.empty or balance_sheet.empty:
            return pd.DataFrame()

        financial_df = pd.DataFrame(index=income_stmt.index)

        # Calculate key metrics
        if 'Net Income' in income_stmt.columns:
            market_cap = info.get('marketCap', 0)
            if market_cap > 0:
                financial_df['PER'] = market_cap / income_stmt['Net Income']
            else:
                financial_df['PER'] = np.nan
            shares = info.get('sharesOutstanding', 1)
            financial_df['EPS'] = income_stmt['Net Income'] / shares

        if 'Total Assets' in balance_sheet.columns and 'Total Liabilities Net Minority Interest' in balance_sheet.columns:
            book_value = balance_sheet['Total Assets'] - balance_sheet.get('Total Liabilities Net Minority Interest', 0)
            market_cap = info.get('marketCap', 0)
            if market_cap > 0:
                financial_df['PBR'] = market_cap / book_value
            else:
                financial_df['PBR'] = np.nan
            financial_df['BPS'] = book_value / info.get('sharesOutstanding', 1)

        if 'Net Income' in income_stmt.columns and 'Stockholders Equity' in balance_sheet.columns:
            financial_df['ROE'] = income_stmt['Net Income'] / balance_sheet['Stockholders Equity']

        if 'Net Income' in income_stmt.columns and 'Total Revenue' in income_stmt.columns:
            financial_df['Net_Margin'] = income_stmt['Net Income'] / income_stmt['Total Revenue']

        if 'Total Debt' in balance_sheet.columns and 'Stockholders Equity' in balance_sheet.columns:
            financial_df['DE_Ratio'] = balance_sheet['Total Debt'] / balance_sheet['Stockholders Equity']

        if 'Total Revenue' in income_stmt.columns:
            financial_df['Revenue_Growth'] = income_stmt['Total Revenue'].pct_change(periods=4)

        if 'EPS' in financial_df.columns:
            financial_df['EPS_Growth'] = financial_df['EPS'].pct_change(periods=4)

        financial_df = financial_df.replace([np.inf, -np.inf], np.nan)
        financial_df = financial_df.sort_index()

        return financial_df

    except:
        return pd.DataFrame()


def analyze_single_stock(ticker, start_date, end_date, selected_features, prediction_days, progress_callback=None):
    """
    Analyze a single stock and return results
    Returns: dict with predictions and metrics, or None if failed
    """
    try:
        if progress_callback:
            progress_callback(f"Analyzing {ticker}...")

        # Download data
        buffer_days = 150 + prediction_days
        adjusted_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=buffer_days)).strftime('%Y-%m-%d')

        data = yf.download(ticker, start=adjusted_start, end=end_date, progress=False, auto_adjust=False)

        if data.empty or len(data) < 60:
            return None

        # Fix MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0):
                data.columns = data.columns.droplevel(1)
            elif 'Close' in data.columns.get_level_values(1):
                data.columns = data.columns.droplevel(0)
            else:
                data.columns = data.columns.get_level_values(1)

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            return None

        if 'Adj Close' in data.columns:
            data = data.drop(columns=['Adj Close'])

        # Get financial data
        financial_df = fetch_financial_data(ticker, progress_callback)
        if not financial_df.empty:
            data = data.join(financial_df, how='left')
            data = data.ffill()

        # Calculate technical indicators
        data['MA_5'] = data['Close'].rolling(window=5).mean()
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()

        try:
            data.ta.rsi(append=True, length=14, close=data['Close'])
        except:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI_14'] = 100 - (100 / (1 + rs))

        try:
            data.ta.macd(append=True, close=data['Close'])
        except:
            pass

        try:
            data.ta.stoch(append=True, high=data['High'], low=data['Low'], close=data['Close'])
        except:
            pass

        data['VMA_20'] = data['Volume'].rolling(window=20).mean()
        data['Price_Change'] = data['Close'].pct_change()
        data['Volume_Change'] = data['Volume'].pct_change()
        data['High_Low_Ratio'] = (data['High'] - data['Low']) / data['Close']
        data['Close_Open_Ratio'] = (data['Close'] - data['Open']) / data['Open']

        # Selective dropna
        critical_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_5', 'MA_20', 'MA_50', 'Price_Change', 'Volume_Change']
        existing_critical = [col for col in critical_cols if col in data.columns]
        data = data.dropna(subset=existing_critical)
        data = data.fillna(0)

        # Filter to date range
        start_dt = pd.to_datetime(start_date)
        data = data[data.index >= start_dt]

        if len(data) < 30 + prediction_days:
            return None

        # Create targets
        data['Target_Direction'] = (data['Close'].shift(-prediction_days) / data['Close'] > 1 + MIN_REVERSE_PERCENT).astype(int)
        data['Target_Return'] = (data['Close'].shift(-prediction_days) / data['Close'] - 1) * 100

        # Create signals
        feature_columns = {}

        if 'RSI' in selected_features and 'RSI_14' in data.columns:
            RSI_UPPER_BOUND = RSI_OVERSOLD + 5
            data['Signal_RSI'] = np.where(
                (data['RSI_14'] > data['RSI_14'].shift(1)) &
                (data['RSI_14'] < RSI_UPPER_BOUND) &
                (data['RSI_14'] >= RSI_OVERSOLD), 1, 0)
            feature_columns['Signal_RSI'] = 'RSI 반등'

        if 'MACD' in selected_features and 'MACDh_12_26_9' in data.columns:
            data['Signal_MACD'] = np.where(
                (data['MACDh_12_26_9'].shift(1) < 0) &
                (data['MACDh_12_26_9'] >= 0), 1, 0)
            feature_columns['Signal_MACD'] = 'MACD 골든크로스'

        if 'MA' in selected_features and 'MA_5' in data.columns:
            data['Signal_MA'] = np.where(data['Close'] > data['MA_5'], 1, 0)
            feature_columns['Signal_MA'] = 'MA 회복'

        if 'Volume' in selected_features and 'VMA_20' in data.columns:
            data['Signal_Volume'] = np.where(
                (data['Volume'] >= data['VMA_20'] * VOLUME_MULTIPLIER) &
                (data['Close'] > data['Close'].shift(1)), 1, 0)
            feature_columns['Signal_Volume'] = '대량 거래량'

        # Additional features
        additional_features = ['Price_Change', 'Volume_Change', 'High_Low_Ratio', 'Close_Open_Ratio', 'RSI_14', 'MACDh_12_26_9']
        financial_features = ['PER', 'PBR', 'EPS', 'BPS', 'ROE', 'Net_Margin', 'DE_Ratio', 'Revenue_Growth', 'EPS_Growth']

        signal_features = list(feature_columns.keys())
        all_features = signal_features + additional_features + financial_features
        all_features = [f for f in all_features if f in data.columns]

        if len(all_features) == 0:
            return None

        # Selective dropna for targets
        critical_signal_cols = ['Target_Direction', 'Target_Return'] + all_features
        existing_signal_cols = [col for col in critical_signal_cols if col in data.columns]
        data = data.dropna(subset=existing_signal_cols)

        if len(data) < 20:
            return None

        # Validate sufficient data for train/test split
        if len(data) < 20 + prediction_days:
            return None

        # Train models
        train_data = data.iloc[:-prediction_days].copy()
        X_all = train_data[all_features].values
        y_direction = train_data['Target_Direction'].values
        y_return = train_data['Target_Return'].values
        latest_features = data[all_features].iloc[-prediction_days:].values

        # Classification model
        clf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        clf_model.fit(X_all, y_direction)

        # Regression model
        reg_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        reg_model.fit(X_all, y_return)

        # Predictions for the latest available date
        latest_prediction_features = data[all_features].iloc[-1]
        direction_proba = clf_model.predict_proba(latest_prediction_features.values.reshape(1, -1))[0][1] * 100
        expected_return = reg_model.predict(latest_prediction_features.values.reshape(1, -1))[0]

        # Model performance on TRAIN data
        y_pred_direction = clf_model.predict(X_all)
        y_pred_return = reg_model.predict(X_all)
        clf_accuracy = accuracy_score(y_direction, y_pred_direction) * 100
        reg_mae = mean_absolute_error(y_return, y_pred_return)
        reg_r2 = r2_score(y_return, y_pred_return)

        # Store prediction results for visualization
        train_data['Predicted_Direction'] = y_pred_direction
        train_data['Prediction_Match'] = np.where(train_data['Predicted_Direction'] == train_data['Target_Direction'], 1, 0)

        # Get current price and financial metrics
        current_price = data['Close'].iloc[-1]
        latest_financials = {}

        for feat in financial_features:
            if feat in data.columns and not pd.isna(data[feat].iloc[-1]):
                latest_financials[feat] = data[feat].iloc[-1]

        # Get active signals
        active_signals = {}
        for sig in signal_features:
            if sig in data.columns:
                active_signals[sig] = int(data[sig].iloc[-1])

        return {
            'ticker': ticker,
            'success': True,
            'prediction_days': prediction_days,
            'direction_proba': direction_proba,
            'expected_return': expected_return,
            'clf_accuracy': clf_accuracy,
            'reg_mae': reg_mae,
            'reg_r2': reg_r2,
            'current_price': current_price,
            'financial_metrics': latest_financials,
            'active_signals': active_signals,
            'data_points': len(train_data),
            'features_used': len(all_features),
            'has_financials': len(latest_financials) > 0,
            'train_data_for_plot': train_data.tail(200)
        }

    except Exception as e:
        return None


def compare_multiple_stocks(tickers, start_date, end_date, selected_features, prediction_days, progress_callback=None):
    """
    Compare multiple stocks
    Returns list of results
    """
    results = []

    for i, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(f"Analyzing {ticker} ({i+1}/{len(tickers)})...")

        result = analyze_single_stock(ticker, start_date, end_date, selected_features, prediction_days, progress_callback)

        if result is not None:
            results.append(result)
        else:
            results.append({
                'ticker': ticker,
                'success': False,
                'error': 'Insufficient data or analysis failed'
            })

    return results


def plot_analysis_chart(ticker, train_data, prediction_days):
    """Generate chart showing price action and model accuracy"""
    try:
        data = train_data.copy()

        data['Success'] = np.where(
            (data['Prediction_Match'] == 1) & (data['Target_Direction'] == 1),
            data['Close'],
            np.nan
        )
        data['Failure'] = np.where(
            (data['Prediction_Match'] == 0) & (data['Target_Direction'] == 1),
            data['Close'],
            np.nan
        )
        data['False_Positive'] = np.where(
            (data['Prediction_Match'] == 0) & (data['Target_Direction'] == 0),
            data['Close'],
            np.nan
        )

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(data.index, data['Close'], label='Close Price', color='#2c3e50', linewidth=1.5)

        ax.scatter(data.index, data['Success'],
                    label='Correctly Predicted Up',
                    marker='^', color='#2ecc71', s=60, zorder=5)

        ax.scatter(data.index, data['Failure'],
                    label='Missed Up Move (False Negative)',
                    marker='v', color='#e67e22', s=60, zorder=5)

        ax.scatter(data.index, data['False_Positive'],
                    label='False Signal (False Positive)',
                    marker='x', color='#e74c3c', s=60, zorder=5)

        latest_timepoint = data.index[-1]

        ax.axvline(x=latest_timepoint, color='#3498db', linestyle='--', linewidth=2,
                    label=f'Latest Model Training Point ({latest_timepoint.strftime("%Y-%m-%d")})')

        ax.set_title(f'[{ticker}] Model Performance Visualization (Target: {prediction_days}-Day UP)', fontsize=14)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.6)

        return fig

    except Exception as e:
        st.error(f"Failed to generate chart for {ticker}: {e}")
        return None


# ===============================
# STREAMLIT APP
# ===============================

def main():
    st.set_page_config(
        page_title="Stock Analyzer v3",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #2c3e50;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #7f8c8d;
            text-align: center;
            margin-bottom: 2rem;
        }
        .stButton>button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'gemini_settings' not in st.session_state:
        st.session_state.gemini_settings = load_gemini_settings()
    if 'tickers' not in st.session_state:
        st.session_state.tickers = 'AAPL, MSFT, GOOGL'

    # Sidebar
    with st.sidebar:
        st.header("🤖 Gemini AI Settings")

        gemini_enabled = st.checkbox(
            "Enable Gemini",
            value=st.session_state.gemini_settings.get('enabled', True)
        )

        api_key = st.text_input(
            "API Key",
            value=st.session_state.gemini_settings.get('api_key', ''),
            type="password",
            help="Get your free API key from Google AI Studio"
        )

        if st.button("💾 Save API Key"):
            st.session_state.gemini_settings['api_key'] = api_key
            st.session_state.gemini_settings['enabled'] = gemini_enabled
            if save_gemini_settings(st.session_state.gemini_settings):
                st.success("✅ API key saved!")
            else:
                st.error("❌ Failed to save API key")

        if st.button("🔑 Get Free API Key"):
            st.markdown("[Click here to get API key](https://aistudio.google.com/app/apikey)")

        st.markdown("---")

        st.header("📊 Analysis Settings")

        prediction_days = st.number_input(
            "Prediction Days (N)",
            min_value=1,
            max_value=30,
            value=DEFAULT_PREDICTION_DAYS,
            help="Number of days into the future to predict"
        )

        st.subheader("Technical Indicators")
        use_rsi = st.checkbox("RSI (Relative Strength Index)", value=True)
        use_macd = st.checkbox("MACD (Moving Average Convergence)", value=True)
        use_ma = st.checkbox("Moving Average", value=True)
        use_volume = st.checkbox("Volume Analysis", value=True)

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365*2)
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now()
            )

    # Main content
    st.markdown('<div class="main-header">📊 Stock Analyzer v3</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-Stock Comparison with Gemini AI</div>', unsafe_allow_html=True)

    # Gemini Natural Language Section
    if gemini_enabled:
        with st.expander("🤖 Natural Language Stock Selection", expanded=True):
            st.info("💡 Examples: 'Top 5 tech stocks' | 'FAANG companies' | 'Apple and its competitors'")

            nl_query = st.text_input(
                "Describe the stocks you want to analyze:",
                value="Compare Apple and Microsoft",
                key="nl_query"
            )

            if st.button("🔍 Get Tickers from Gemini"):
                if not GEMINI_AVAILABLE:
                    st.error("❌ Gemini library not installed. Run: pip install google-generativeai")
                elif not api_key or api_key.strip() == "":
                    st.error("❌ Please enter your Gemini API key in the sidebar")
                elif not nl_query or nl_query.strip() == "":
                    st.error("❌ Please enter a stock query")
                else:
                    with st.spinner("🔄 Calling Gemini API..."):
                        success, tickers_list, error = get_tickers_from_gemini(nl_query, api_key)

                        if success:
                            tickers_str = ', '.join(tickers_list)
                            st.session_state.tickers = tickers_str
                            st.success(f"✅ Found {len(tickers_list)} tickers: {tickers_str}")
                        else:
                            st.error(f"❌ {error}")

    # Manual Ticker Input
    st.subheader("📝 Manual Ticker Input")
    tickers_input = st.text_input(
        "Enter stock tickers (comma-separated):",
        value=st.session_state.tickers,
        help="Example: AAPL, MSFT, GOOGL, TSLA"
    )

    # Analyze Button
    if st.button("🚀 Compare Stocks", type="primary", use_container_width=True):
        # Parse tickers
        tickers_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

        if len(tickers_list) == 0:
            st.error("❌ Please enter at least one ticker")
            return

        if len(tickers_list) > 20:
            st.warning("⚠️ Maximum 20 tickers allowed. Using first 20.")
            tickers_list = tickers_list[:20]

        # Get selected features
        selected_features = []
        if use_rsi:
            selected_features.append('RSI')
        if use_macd:
            selected_features.append('MACD')
        if use_ma:
            selected_features.append('MA')
        if use_volume:
            selected_features.append('Volume')

        if not selected_features:
            st.error("❌ Please select at least one indicator in the sidebar")
            return

        # Run analysis
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(msg):
            status_text.text(msg)

        with st.spinner(f"Analyzing {len(tickers_list)} stocks..."):
            results = compare_multiple_stocks(
                tickers_list,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                selected_features,
                prediction_days,
                progress_callback
            )

        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")

        # Display results
        display_results(results, prediction_days)


def display_results(results, prediction_days):
    """Display analysis results"""
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]

    if len(successful) == 0:
        st.error("❌ No stocks could be analyzed. Please check tickers and date range.")
        return

    # Sort by expected return
    successful.sort(key=lambda x: x.get('expected_return', -999), reverse=True)

    st.success(f"✅ Successfully analyzed {len(successful)} out of {len(results)} stocks")

    # Summary Table
    st.subheader(f"📊 Summary Table (N={prediction_days} Day Forecast)")

    summary_data = []
    for i, r in enumerate(successful, 1):
        summary_data.append({
            'Rank': f"#{i}",
            'Ticker': r['ticker'],
            'Price': f"${r['current_price']:.2f}",
            'Direction %': f"{r['direction_proba']:.1f}%",
            'Expected Return': f"{r['expected_return']:+.2f}%",
            'Accuracy': f"{r['clf_accuracy']:.1f}%",
            'R²': f"{r['reg_r2']:.3f}",
            'Financials': "✅" if r['has_financials'] else "❌"
        })

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Detailed Analysis
    st.subheader("📈 Detailed Analysis")

    for i, r in enumerate(successful, 1):
        with st.expander(f"#{i} - {r['ticker']} - Expected Return: {r['expected_return']:+.2f}%"):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Current Price", f"${r['current_price']:.2f}")
            with col2:
                st.metric("Direction Probability", f"{r['direction_proba']:.1f}%")
            with col3:
                st.metric("Expected Return", f"{r['expected_return']:+.2f}%")
            with col4:
                st.metric("Model Accuracy", f"{r['clf_accuracy']:.1f}%")

            col5, col6, col7, col8 = st.columns(4)
            with col5:
                st.metric("Return MAE", f"{r['reg_mae']:.2f}%")
            with col6:
                st.metric("R² Score", f"{r['reg_r2']:.3f}")
            with col7:
                st.metric("Data Points", r['data_points'])
            with col8:
                st.metric("Features Used", r['features_used'])

            # Financial Metrics
            if r['has_financials']:
                st.markdown("**📊 Financial Metrics:**")
                fin_cols = st.columns(4)
                fin_metrics = r['financial_metrics']

                for idx, (key, val) in enumerate(fin_metrics.items()):
                    with fin_cols[idx % 4]:
                        if key in ['ROE', 'Net_Margin', 'Revenue_Growth', 'EPS_Growth']:
                            st.write(f"**{key}:** {val*100:.2f}%")
                        else:
                            st.write(f"**{key}:** {val:.2f}")

            # Active Signals
            if r['active_signals']:
                st.markdown("**🔔 Active Signals:**")
                sig_cols = st.columns(len(r['active_signals']))
                for idx, (sig, active) in enumerate(r['active_signals'].items()):
                    with sig_cols[idx]:
                        status = "🟢 Active" if active == 1 else "⚫ Inactive"
                        st.write(f"**{sig.replace('Signal_', '')}:** {status}")

            # Chart
            if 'train_data_for_plot' in r:
                st.markdown("**📉 Price Chart & Model Performance:**")
                fig = plot_analysis_chart(r['ticker'], r['train_data_for_plot'], r['prediction_days'])
                if fig:
                    st.pyplot(fig)

    # Rankings
    st.subheader("🏆 Rankings")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Top 5 by Expected Return:**")
        for i, r in enumerate(successful[:5], 1):
            st.write(f"{i}. **{r['ticker']}**: {r['expected_return']:+.2f}%")

    with col2:
        st.markdown("**Top 5 by Direction Probability:**")
        sorted_by_prob = sorted(successful, key=lambda x: x['direction_proba'], reverse=True)
        for i, r in enumerate(sorted_by_prob[:5], 1):
            st.write(f"{i}. **{r['ticker']}**: {r['direction_proba']:.1f}%")

    with col3:
        st.markdown("**Best Fundamentals (ROE):**")
        with_roe = [r for r in successful if 'ROE' in r['financial_metrics']]
        if with_roe:
            sorted_by_roe = sorted(with_roe, key=lambda x: x['financial_metrics'].get('ROE', -999), reverse=True)
            for i, r in enumerate(sorted_by_roe[:5], 1):
                roe = r['financial_metrics']['ROE']
                st.write(f"{i}. **{r['ticker']}**: {roe*100:.2f}%")
        else:
            st.write("No ROE data available")

    # Failed stocks
    if failed:
        st.warning(f"⚠️ Failed to analyze {len(failed)} stocks:")
        for r in failed:
            st.write(f"• **{r['ticker']}**: {r.get('error', 'Unknown error')}")

    # Interpretation
    with st.expander("💡 How to Interpret Results"):
        st.markdown(f"""
        - **Expected Return**: Predicted return over the next {prediction_days} days
        - **Direction Probability**: Chance of a {prediction_days}-day increase of 2%+
        - **Accuracy**: Model accuracy on training data (for visualization)
        - **R²**: Regression model fit quality (higher is better, max 1.0)
        - **Active Signals**: Technical indicators that are currently triggered
        """)


if __name__ == "__main__":
    main()
