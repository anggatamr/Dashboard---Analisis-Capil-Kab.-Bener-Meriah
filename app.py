import streamlit as st
import pandas as pd
import joblib
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt # Added for interactive charts
import base64
from sklearn.cluster import KMeans

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Dashboard Analisis Capil",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Theme Configuration ---
THEMES = {
    "🏠 Ringkasan Data": {
        "primary": "#3B82F6", # Blue
        "secondary_bg": "#EFF6FF",
        "header_gradient": "-webkit-linear-gradient(45deg, #2563EB, #60A5FA)",
        "app_bg": "linear-gradient(135deg, #F9FAFB 0%, #EFF6FF 100%)",
        "button_gradient": "linear-gradient(45deg, #3B82F6, #2563EB)",
        "icon": "📊"
    },
    "🎯 Peta Sebaran & Prediksi": {
        "primary": "#10B981", # Emerald Green
        "secondary_bg": "#ECFDF5",
        "header_gradient": "-webkit-linear-gradient(45deg, #059669, #34D399)",
        "app_bg": "linear-gradient(135deg, #F0FDF4 0%, #D1FAE5 100%)",
        "button_gradient": "linear-gradient(45deg, #10B981, #059669)",
        "icon": "🗺️"
    },
    "🔮 Proyeksi Masa Depan": {
        "primary": "#8B5CF6", # Violet
        "secondary_bg": "#F5F3FF",
        "header_gradient": "-webkit-linear-gradient(45deg, #7C3AED, #A78BFA)",
        "app_bg": "linear-gradient(135deg, #FAF5FF 0%, #E9D5FF 100%)",
        "button_gradient": "linear-gradient(45deg, #8B5CF6, #7C3AED)",
        "icon": "🔮"
    },
    "📚 Panduan Pengguna": {
        "primary": "#F59E0B", # Amber
        "secondary_bg": "#FFFBEB",
        "header_gradient": "-webkit-linear-gradient(45deg, #D97706, #FBBF24)",
        "app_bg": "linear-gradient(135deg, #FFF7ED 0%, #FEF3C7 100%)",
        "button_gradient": "linear-gradient(45deg, #F59E0B, #D97706)",
        "icon": "📚"
    },
    "💬 Kritik & Saran": {
        "primary": "#EC4899", # Pink/Rose
        "secondary_bg": "#FDF2F8",
        "header_gradient": "-webkit-linear-gradient(45deg, #DB2777, #F472B6)",
        "app_bg": "linear-gradient(135deg, #FDF2F8 0%, #FCE7F3 100%)",
        "button_gradient": "linear-gradient(45deg, #EC4899, #DB2777)",
        "icon": "💬"
    }
}

# --- Custom CSS untuk White Theme & Dynamic Gradients ---
def local_css(page_name):
    theme = THEMES.get(page_name, THEMES["🏠 Ringkasan Data"])
    
    # Base Global Styles (White Theme)
    base_style = f"""
    <style>
    /* Global Font */
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: #333333;
    }}
    
    /* Remove top padding */
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: #FFFFFF;
        border-right: 1px solid #E5E7EB;
    }}
    
    /* Metrics Cards */
    div[data-testid="metric-container"] {{
        background-color: #FFFFFF;
        border-left: 5px solid {theme['primary']};
        border-top: 1px solid #E5E7EB;
        border-right: 1px solid #E5E7EB;
        border-bottom: 1px solid #E5E7EB;
        padding: 15px;
        border-radius: 12px;
        color: #1F2937;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s;
    }}
    div[data-testid="metric-container"]:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }}
    
    /* Headers - Dynamic Gradient */
    h1, h2, h3 {{
        color: #111827;
        font-weight: 700;
    }}
    
    /* Gradient Text Class */
    .gradient-text {{
        background: {theme['header_gradient']};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        border-bottom: 1px solid #E5E7EB;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        border-radius: 6px;
        color: #6B7280;
        padding: 8px 16px;
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {theme['secondary_bg']} !important;
        color: {theme['primary']} !important;
        border-bottom: 2px solid {theme['primary']};
    }}
    
    /* Buttons */
    .stButton > button {{
        background: {theme['button_gradient']};
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }}
    .stButton > button:hover {{
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        transform: translateY(-1px);
        opacity: 0.9;
    }}
    
    /* Dynamic Background */
    .stApp {{
        background: {theme['app_bg']};
        background-attachment: fixed;
    }}
    </style>
    """
    
    st.markdown(base_style, unsafe_allow_html=True)



# --- Load Data & Models ---
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('datacapil2025-2026.csv', sep=';', engine='python')
        
        # Data Cleaning: Fix duplicates in Kecamatan (e.g. "Bukit " vs "Bukit")
        if 'kecamatan' in data.columns:
            data['kecamatan'] = data['kecamatan'].astype(str).str.strip().str.title()

        data['tanggal'] = pd.to_datetime(data['tanggal'], dayfirst=True)
        data['day'] = data['tanggal'].dt.day
        data['month'] = data['tanggal'].dt.month
        data['year'] = data['tanggal'].dt.year
        data['week'] = data['tanggal'].dt.isocalendar().week
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def load_models():
    try:
        rf = joblib.load('rf_model_keterangan.pkl')
        scaler = joblib.load('scaler.pkl')
        kmeans = joblib.load('kmeans_model_k4.pkl')
        hw = joblib.load('hw_forecast_model.pkl')
        return rf, scaler, kmeans, hw
    except FileNotFoundError as e:
        st.warning(f"⚠️ Model file not found: {e}. Some features will be disabled.")
        return None, None, None, None
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None, None

df_dash = load_data()
if not df_dash.empty:
    rf_model, scaler, kmeans_model, hw_model = load_models()
else:
    rf_model, scaler, kmeans_model, hw_model = None, None, None, None

# --- Disclaimer Popup (First Visit Only) ---
@st.dialog("ℹ️ Informasi Pengguna")
def show_disclaimer():
    st.markdown("""
    ### Haloo! Selamat Datang! 👋
    
    Dashboard ini dikembangkan sebagai **proyek pembelajaran/edukasi**.
    
    Harap diingat bahwa:
    *   Masih banyak aspek yang dapat dikembangkan lebih lanjut.
    *   Data dan prediksi mungkin belum sepenuhnya akurat.
    *   Kritik dan saran sangat diapresiasi!
    
    *Terima kasih telah mencoba.* 🙏
    ### by Angga Tamara
    """)
    if st.button("Saya Mengerti & Lanjutkan", type="primary"):
        st.rerun()

if "has_seen_disclaimer" not in st.session_state:
    show_disclaimer()
    st.session_state["has_seen_disclaimer"] = True


# --- Sidebar Navigation ---
st.sidebar.title("📌 Menu Utama")
page = st.sidebar.radio("Pilih Halaman", 
    ["🏠 Ringkasan Data", "🎯 Peta Sebaran & Prediksi", "🔮 Proyeksi Masa Depan", "📚 Panduan Pengguna", "💬 Kritik & Saran"], 
    index=0
)


# --- Apply CSS based on Page ---
local_css(page)
current_theme = THEMES.get(page, THEMES["💬 Kritik & Saran"] if page == "💬 Kritik & Saran" else THEMES["🏠 Ringkasan Data"])

# --- Sidebar Filters (Global) ---
# ... rest of sidebar code ...
st.sidebar.markdown("### 🔍 Filter Data")
kecamatan_filter = "Semua Kecamatan"
if not df_dash.empty:
    # Ensure all values are strings and handle NaNs to avoid sorting errors
    unique_kec = df_dash['kecamatan'].dropna().astype(str).unique().tolist()
    kecamatan_list = ["Semua Kecamatan"] + sorted(unique_kec)
    kecamatan_filter = st.sidebar.selectbox("Pilih Kecamatan:", kecamatan_list)

st.sidebar.info("Gunakan filter di atas untuk menyesuaikan tampilan data statistik.")


# --- Content Pages ---
# --- Content Pages ---
if page == "Ringkasan Data" or page == "🏠 Ringkasan Data": # Handle with/without emoji
    st.markdown(f"# {current_theme['icon']} <span class='gradient-text'>Ringkasan Data Kependudukan</span>", unsafe_allow_html=True)
    st.markdown("Gambaran umum aktivitas pengurusan dokumen kependudukan periode **2025-2026**.")
    
    if df_dash.empty:
        st.warning("Data belum dimuat.")
    else:
        # Apply Filter
        df_view = df_dash.copy()
        if kecamatan_filter != "Semua Kecamatan":
            df_view = df_view[df_view['kecamatan'] == kecamatan_filter]
            st.success(f"Menampilkan data untuk kecamatan: **{kecamatan_filter}**")
            
        total_ktp = len(df_view)
        sudah_diambil = len(df_view[df_view['keterangan'] == 'Sudah Diambil'])
        belum_diambil = total_ktp - sudah_diambil
        
        # Metrics Row
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Dokumen", f"{total_ktp:,}")
        c2.metric("✅ Sudah Diambil", f"{sudah_diambil:,}")
        c3.metric("⏳ Belum Diambil", f"{belum_diambil:,}")
        
        st.divider()
        
        # Charts Row
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📅 Tren Aktivitas Harian")
            if not df_view.empty:
                daily_trend = df_view.groupby('tanggal').size().reset_index(name='Jumlah')
                st.area_chart(daily_trend.set_index('tanggal'), color=current_theme['primary'])
            else:
                st.info("Tidak ada data untuk filter ini.")
            
        with col2:
            st.subheader("📊 Status Dokumen")
            if not df_view.empty:
                fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
                fig_pie.patch.set_alpha(0) # Transparent background
                status_counts = df_view['keterangan'].value_counts()
                if not status_counts.empty:
                    colors = [current_theme['primary'], '#EF4444']
                    wedges, texts, autotexts = ax_pie.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', 
                                                        startangle=140, colors=colors, textprops=dict(color="white"))
                    st.pyplot(fig_pie)
                else:
                    st.write("Data tidak cukup.")
            else:
                st.write("-")

elif page == "Peta Sebaran & Prediksi" or page == "🎯 Peta Sebaran & Prediksi":
    st.markdown(f"# {current_theme['icon']} <span class='gradient-text'>Peta Sebaran & Prediksi</span>", unsafe_allow_html=True)
    st.markdown("""
    Fitur ini menggunakan **Kecerdasan Buatan (AI)** untuk membantu analisis mendalam:
    
    *   **Prediksi Status**: Memperkirakan apakah dokumen akan diambil tepat waktu berdasarkan pola sejarah.
    *   **Peta Sebaran**: Mengelompokkan pola aktivitas masyarakat untuk perencanaan distribusi yang lebih baik.
    """)
    
    tab1, tab2 = st.tabs(["🔮 Cek Status Dokumen", "🗺️ Peta Pola Wilayah"])
    
    with tab1:
        st.subheader("🔮 Prediksi Pengambilan Dokumen")
        st.info("""
        **ℹ️ Penjelasan Kegunaan:**
        Fitur ini membantu Anda memprediksi apakah seorang pemohon camderung akan **langsung mengambil** dokumennya atau **menunda**. 
        
        *   **Sudah Diambil**: Kemungkinan besar pemohon akan datang tepat waktu.
        *   **Belum Diambil**: Pemohon berisiko tidak mengambil dokumen. *Tips: Hubungi pemohon ini lebih dulu.*
        """)
        
        with st.container(border=True):
            st.write("**Masukkan Data Pemohon:**")
            col_in1, col_in2 = st.columns([1, 2])
            
            with col_in1:
                tgl_input = st.date_input(
                    "Pilih Tanggal",
                    value=pd.Timestamp("2025-08-21"),
                    min_value=pd.Timestamp("2025-08-21"),
                    max_value=pd.Timestamp("2026-02-09"),
                    help="Rentang Data: 21 Agustus 2025 - 09 Februari 2026"
                )

            
            col_in4, col_in5 = st.columns(2)
            
            # IMPROVEMENT: Use Names instead of raw IDs for Kecamatan
            # Assuming LabelEncoder was used alphabetically on the dataset
            if not df_dash.empty:
                kecamatan_list = sorted(df_dash['kecamatan'].dropna().unique().tolist())
                selected_kec = col_in4.selectbox("Pilih Kecamatan", kecamatan_list, help="Pilih nama kecamatan sesuai KTP")
                try:
                    kec_enc = kecamatan_list.index(selected_kec)
                    st.caption(f"ℹ️ ID Sistem: {kec_enc}") 
                except ValueError:
                    kec_enc = 0
            else:
                kec_enc = col_in4.number_input("ID Kecamatan (Manual)", 0, 12, 0)

            gender = col_in5.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
            gender_enc = 0 if gender == "Laki-laki" else 1

            nama_input = st.text_input("Nama Lengkap Pemohon", placeholder="Contoh: Budi Santoso")
            
            if st.button("🚀 Analisa Status Sekarang", use_container_width=True):
                if not nama_input:
                    st.warning("⚠️ Mohon isi Nama Lengkap terlebih dahulu.")
                elif rf_model is not None:
                    # Feature Extraction Logic
                    day = tgl_input.day
                    month = tgl_input.month
                    weekday = tgl_input.weekday() # 0=Monday, 6=Sunday
                    is_weekend = 1 if weekday >= 5 else 0
                    
                    s = str(nama_input)
                    words = s.split()
                    name_len = len(s)
                    name_words = len(words)
                    has_digit = int(any(ch.isdigit() for ch in s))
                    
                    titles = ['Dr','Dr.','Drs','Bpk','Bpk.','Ibu','Ibu.','Bu','Bu.','Pak','Pak.','H','H.','Hj','Hj.','Mr','Mrs']
                    has_title = int(any(w.rstrip('.').title() in [t.rstrip('.') for t in titles] for w in words))
                    
                    input_data = [[day, month, weekday, is_weekend, kec_enc, gender_enc, name_len, name_words, has_digit, has_title]]
                    
                    try:
                        pred = rf_model.predict(input_data)
                        is_taken = pred[0] == 1 # 1=Sudah, 0=Belum
                        
                        st.divider()
                        if is_taken:
                            st.success("### ✅ Hasil Prediksi: SUDAH DIAMBIL")
                            st.markdown("Sistem memperkirakan dokumen ini **akan diambil tepat waktu** oleh pemohon.")
                        else:
                            st.error("### ⏳ Hasil Prediksi: BELUM DIAMBIL")
                            st.markdown("Sistem memperkirakan dokumen ini **berisiko tertunda** pengambilannya.")
                            
                    except Exception as e:
                        st.error(f"Terjadi kesalahan teknis: {e}")
                else:
                    st.error("Model AI belum dimuat dengan sempurna.")
            
    with tab2:
        st.subheader("🗺️ Segmentasi Pola Wilayah (Resource Allocation)")
        st.info("Analisis ini mengelompokkan **Kecamatan** berdasarkan beban kerja dan performa pengambilan, untuk membantu **alokasi petugas/sumber daya**.")
        
        if not df_dash.empty:
            # 1. Prepare Data per Kecamatan
            kec_stats = df_dash.groupby('kecamatan').agg(
                Total_Permohonan=('nama', 'count'),
                Sudah_Diambil=('keterangan', lambda x: (x == 'Sudah Diambil').sum())
            ).reset_index()
            
            kec_stats['Pickup_Rate'] = (kec_stats['Sudah_Diambil'] / kec_stats['Total_Permohonan']) * 100
            
            # 2. Perform Dynamic Clustering (K-Means)
            # Use Volume & Rate as features
            X_kec = kec_stats[['Total_Permohonan', 'Pickup_Rate']].values
            
            # Determine optimal k (small dataset, stick to 3 or 4 mostly)
            n_clusters = 3
            if len(kec_stats) < 3: n_clusters = 1
            
            kmeans_kec = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kec_stats['Cluster'] = kmeans_kec.fit_predict(X_kec)
            
            # 3. Labeling Clusters (Automated logic based on centroids)
            # Calculate mean of each cluster to define labels
            cluster_profile = kec_stats.groupby('Cluster')[['Total_Permohonan', 'Pickup_Rate']].mean()
            
            def get_cluster_label(row):
                # Simple heuristic based on volume and rate relative to dataset mean
                avg_vol = kec_stats['Total_Permohonan'].mean()
                avg_rate = kec_stats['Pickup_Rate'].mean()
                
                vol_status = "Tinggi" if row['Total_Permohonan'] > avg_vol else "Rendah"
                rate_status = "Tinggi" if row['Pickup_Rate'] > avg_rate else "Rendah"
                
                if vol_status == "Tinggi" and rate_status == "Rendah":
                    return "⚠️ Prioritas Penanganan (Sibuk & Tertunda)"
                elif vol_status == "Tinggi" and rate_status == "Tinggi":
                    return "✅ Sibuk & Efisien (Pertahankan)"
                elif vol_status == "Rendah" and rate_status == "Tinggi":
                    return "👌 Stabil (Beban Ringan)"
                else: 
                    return "🔍 Perlu Perhatian (Sepi tapi Tertunda)"

            # Apply labels map
            cluster_labels = {i: get_cluster_label(row) for i, row in cluster_profile.iterrows()}
            kec_stats['Kategori'] = kec_stats['Cluster'].map(cluster_labels)
            
            # 4. Visualization (Interactive with Altair)
            st.markdown("---")
            
            # Create interactive chart
            chart = alt.Chart(kec_stats).mark_circle(size=200).encode(
                x=alt.X('Total_Permohonan', title='Total Permohonan (Beban Kerja)'),
                y=alt.Y('Pickup_Rate', title='Persentase Pengambilan (%)'),
                color=alt.Color('Kategori', legend=alt.Legend(title="Kategori Cluster", orient="bottom")),
                tooltip=[
                    alt.Tooltip('kecamatan', title='Kecamatan'),
                    alt.Tooltip('Total_Permohonan', title='Jml Permohonan'),
                    alt.Tooltip('Pickup_Rate', title='Rate Pengambilan', format='.2f'),
                    alt.Tooltip('Kategori', title='Status')
                ]
            ).properties(
                title='Peta Persebaran Kecamatan (Interaktif - Arahkan Mouse)',
                height=400
            ).interactive()

            st.altair_chart(chart, use_container_width=True)

            # Recommendations in Cards
            st.markdown("### 💡 Rekomendasi Tindakan Strategis")
            
            rec_cols = st.columns(2)
            
            for i, label in enumerate(kec_stats['Kategori'].unique()):
                kec_list = kec_stats[kec_stats['Kategori'] == label]['kecamatan'].tolist()
                kec_str = ", ".join(kec_list)
                
                # Determine box color/type
                if "Prioritas" in label:
                    box_type = "error"
                    icon = "🚨"
                elif "Sibuk & Efisien" in label:
                    box_type = "success"
                    icon = "✅"
                elif "Stabil" in label:
                    box_type = "info"
                    icon = "👌"
                else:
                    box_type = "warning"
                    icon = "⚠️"

                with rec_cols[i % 2]:
                    with st.container(border=True):
                        st.markdown(f"#### {icon} {label}")
                        if "Prioritas" in label:
                            st.write("**Saran:** Tambah loket pengambilan atau kirim notifikasi massal.")
                        elif "Sibuk & Efisien" in label:
                            st.write("**Saran:** Berikan apresiasi atau jadikan contoh SOP.")
                        elif "Stabil" in label:
                            st.write("**Saran:** Monitoring berkala saja.")
                        else:
                            st.write("**Saran:** Cek kendala distribusi spesifik di sini.")
                        
                        st.caption(f"**Wilayah:** {kec_str}")

            # Show Data Table
            with st.expander("Lihat Data Detail per Kecamatan"):
                st.dataframe(kec_stats[['kecamatan', 'Total_Permohonan', 'Pickup_Rate', 'Kategori']].sort_values('Pickup_Rate'), use_container_width=True)
                
        else:
            st.info("Data tidak cukup untuk menampilkan peta pola.")

elif page == "Proyeksi Masa Depan" or page == "🔮 Proyeksi Masa Depan":
    st.markdown(f"# {current_theme['icon']} <span class='gradient-text'>Proyeksi Masa Depan</span>", unsafe_allow_html=True)
    st.markdown("Halaman ini meramalkan jumlah layanan di masa depan berdasarkan tren masa lalu.")
    
    if hw_model is not None and not df_dash.empty:
        try:
            forecast_steps = 14
            forecast = hw_model.forecast(forecast_steps)
            
            last_date = df_dash['tanggal'].max()
            f_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps)
            
            f_df = pd.DataFrame({"Tanggal": f_dates, "Prediksi Jumlah": forecast.values})
            
            st.subheader("📈 Grafik Peramalan 14 Hari Kedepan")
            st.write("Garis oranye menunjukkan perkiraan jumlah aktivitas.")
            
            st.line_chart(f_df.set_index("Tanggal"), color=current_theme['primary'])
            
            with st.expander("Lihat Angka Detail Peramalan"):
                st.dataframe(f_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat forecasting: {e}")
    else:
        st.warning("Model Forecasting belum tersedia.")

elif page == "Panduan Pengguna" or page == "📚 Panduan Pengguna":
    st.markdown(f"# {current_theme['icon']} <span class='gradient-text'>Panduan Penggunaan</span>", unsafe_allow_html=True)
    st.markdown("""
    Selamat datang di **Dashboard Analisis Data Capil**. Berikut cara menggunakan aplikasi ini:
    
    ### 1. 🏠 Ringkasan Data
    *   Halaman ini menampilkan statistik umum.
    *   Warna Tema: **Biru** (Professional & Trust).
    
    ### 2. 🎯 Peta Sebaran & Prediksi
    *   **Tab Cek Status**: Prediksi AI.
    *   **Tab Peta Pola**: Analisis Clustering.
    *   Warna Tema: **Hijau Emerald** (Growth & Mapping).
    
    ### 3. 🔮 Proyeksi Masa Depan
    *   Melihat perkiraan jumlah pemohon.
    *   Warna Tema: **Ungu Violet** (Future & Insight).
    
    ### 4. 💬 Kritik & Saran
    *   Sampaikan masukan Anda untuk pengembangan aplikasi ini.
    *   Warna Tema: **Pink Rose** (Feedback & Care).
    """)

elif page == "Kritik & Saran" or page == "💬 Kritik & Saran":
    st.markdown(f"# {current_theme['icon']} <span class='gradient-text'>Kritik & Saran</span>", unsafe_allow_html=True)
    st.markdown("Kami sangat menghargai masukan Anda untuk pengembangan aplikasi ini.")
    
    with st.container(border=True):
        st.subheader("📝 Form Masukan Pengguna")
        with st.form("feedback_form", clear_on_submit=True):
            name = st.text_input("Nama (Opsional)", placeholder="Nama Anda")
            feedback = st.text_area("Kritik / Saran", placeholder="Tulis masukan Anda di sini...", height=150)
            submitted = st.form_submit_button("Kirim Masukan 🚀")
            
            if submitted:
                if feedback:
                    try:
                        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                        new_data = pd.DataFrame({"Tanggal": [timestamp], "Nama": [name if name else "Anonim"], "Pesan": [feedback]})
                        
                        file_path = "feedback.csv"
                        header = not pd.io.common.file_exists(file_path)
                        new_data.to_csv(file_path, mode='a', header=header, index=False)
                        
                        st.success("✅ Terima kasih! Masukan Anda telah tersimpan.")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Gagal menyimpan masukan: {e}")
                else:
                    st.warning("⚠️ Mohon isi kolom kritik/saran.")
