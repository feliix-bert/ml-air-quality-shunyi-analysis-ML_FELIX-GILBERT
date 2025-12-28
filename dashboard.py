import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Air Quality Dashboard",
    page_icon="🌫️",
    layout="wide"
)

# =====================
# CUSTOM CSS (POPPINS)
# =====================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>
/* FORCE FONT FOR ALL STREAMLIT ELEMENTS */
html, body, [class*="css"], [class*="st-"], 
h1, h2, h3, h4, h5, h6, p, span, div, label {
    font-family: 'Poppins', sans-serif !important;
}
.metric-card {
    background: #1c1f26;
    padding: 20px;
    border-radius: 18px;
    color: white;
}
.metric-title {
    font-size: 14px;
    color: #9ca3af;
}
.metric-value {
    font-size: 32px;
    font-weight: 700;
}
.insight {
    background: #111827;
    padding: 16px;
    border-radius: 12px;
    border-left: 4px solid #22c55e;
    margin-top: 12px;
    margin-bottom: 32px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# =====================
# LOAD & CLEAN DATA
# =====================
@st.cache_data
def load_data():
    df = pd.read_csv("PRSA_Data_Shunyi_20130301-20170228.csv")
    df['datetime'] = pd.to_datetime(df[['year','month','day','hour']])
    df = df.sort_values('datetime').set_index('datetime')

    df['PM2.5'] = df['PM2.5'].interpolate(method='time')
    df[['TEMP','DEWP','PRES','WSPM']] = df[['TEMP','DEWP','PRES','WSPM']].fillna(method='ffill')

    df['PM2.5_clean'] = df['PM2.5'].clip(
        df['PM2.5'].quantile(0.01),
        df['PM2.5'].quantile(0.99)
    )

    return df.reset_index()

df = load_data()

# =====================
# SIDEBAR
# =====================
st.sidebar.title("Air Quality Control Panel")

year_range = st.sidebar.slider(
    "Rentang Tahun Analisis",
    int(df['datetime'].dt.year.min()),
    int(df['datetime'].dt.year.max()),
    (2014, 2016)
)

selected_features = st.sidebar.multiselect(
    "Variabel Cuaca",
    ['TEMP','DEWP','PRES','WSPM'],
    default=['TEMP','DEWP','PRES','WSPM']
)

show_model = st.sidebar.checkbox("Tampilkan Analisis Regresi", True)

st.sidebar.markdown("""
**Kategori PM2.5**
- < 35 : Baik  
- 35–75 : Sedang  
- > 75 : Tidak Sehat  
""")

# =====================
# FILTER DATA
# =====================
data = df[
    (df['datetime'].dt.year >= year_range[0]) &
    (df['datetime'].dt.year <= year_range[1])
]

# =====================
# HEADER
# =====================
st.title("Air Quality Intelligence Dashboard")
st.caption("Public Health & Environmental Decision Support")

# =====================
# INSIGHT AWAL
# =====================
st.markdown("""
<div class="insight">
<b>Insight Awal:</b><br>
Konsentrasi PM2.5 memiliki keterkaitan dengan kondisi meteorologi seperti suhu,
kelembapan, tekanan udara, dan kecepatan angin. Perubahan kondisi cuaca dapat
mempengaruhi akumulasi dan dispersi partikel udara, sehingga berdampak langsung
terhadap kualitas udara dan risiko kesehatan masyarakat.
</div>
""", unsafe_allow_html=True)

# =====================
# METRIC CARDS
# =====================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Rata-rata PM2.5</div>
        <div class="metric-value">{data['PM2.5_clean'].mean():.1f}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">PM2.5 Maksimum</div>
        <div class="metric-value">{data['PM2.5_clean'].max():.1f}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    risk = (data['PM2.5_clean'] > 75).mean()*100
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Hari Berisiko Tinggi</div>
        <div class="metric-value">{risk:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

# =====================
# TREND GRAPH
# =====================
st.subheader("Tren PM2.5")

monthly = (
    data.set_index('datetime')['PM2.5_clean']
    .resample('M').mean().reset_index()
)

fig_trend = px.line(
    monthly,
    x='datetime',
    y='PM2.5_clean',
    template="plotly_dark",
    markers=True
)

st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("""
<div class="insight">
Pola temporal PM2.5 menunjukkan fluktuasi yang konsisten,
mengindikasikan adanya pengaruh musiman dan aktivitas manusia
yang perlu diperhatikan dalam kebijakan pengendalian polusi.
</div>
""", unsafe_allow_html=True)

# =====================
# DISTRIBUTION
# =====================
st.subheader("Distribusi PM2.5")

fig_dist = px.histogram(
    data,
    x='PM2.5_clean',
    nbins=40,
    marginal="box",
    template="plotly_dark"
)

st.plotly_chart(fig_dist, use_container_width=True)

st.markdown("""
<div class="insight">
Distribusi PM2.5 bersifat right-skewed, menunjukkan adanya
kejadian polusi ekstrem yang meskipun jarang,
memiliki dampak signifikan terhadap kualitas udara.
</div>
""", unsafe_allow_html=True)

# =====================
# REGRESSION
# =====================
if show_model:
    st.subheader("Analisis Regresi PM2.5")

    X = data[selected_features]
    y = data['PM2.5_clean']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    fig_reg = px.scatter(
        x=y_test,
        y=y_pred,
        labels={'x':'Actual PM2.5','y':'Predicted PM2.5'},
        template="plotly_dark"
    )

    st.plotly_chart(fig_reg, use_container_width=True)
    st.write(f"**R² Score:** {r2:.2f}")
    st.write(f"**RMSE:** {rmse:.2f}")

    st.markdown("""
    <div class="insight">
    Hasil regresi menunjukkan bahwa variabel cuaca berkontribusi
    dalam menjelaskan variasi PM2.5. Model ini dapat dimanfaatkan
    sebagai alat pendukung pengambilan keputusan berbasis data
    dalam konteks kesehatan publik dan lingkungan.
    </div>
    """, unsafe_allow_html=True)

# =====================
# FOOTER
# =====================
st.caption("Machine Learning Assignment • Air Quality Shunyi Dataset • Felix Gilbert")
