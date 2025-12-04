import streamlit as st
import pandas as pd
from datetime import datetime
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import easyocr

# ==========================================
# 1. KONFIGURASI HALAMAN & DESIGN (APP-LIKE)
# ==========================================
st.set_page_config(
    page_title="FreshGuard",
    page_icon="🍃",
    layout="centered", # Pastikan ini centered (bukan mobile) agar tidak error
    initial_sidebar_state="collapsed"
)

# Custom CSS untuk Tampilan Aplikasi Profesional
st.markdown("""
    <style>
    /* Hapus elemen bawaan Streamlit yang mengganggu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* --- FIX: FORCE DARK TEXT (MENGATASI CHROME DARK MODE) --- */
    .stApp, .stMarkdown, p, h1, h2, h3, h4, h5, h6, span, div, label, 
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: #333333 !important;
    }

    /* Background App */
    .stApp {
        background-color: #f8f9fa;
    }

    /* --- FIX: FILE UPLOADER BOX --- */
    /* Mengubah warna background kotak upload menjadi putih terang & border hijau */
    [data-testid="stFileUploader"] section {
        background-color: #ffffff !important;
        border: 2px dashed #4CAF50 !important;
        border-radius: 15px !important;
        padding: 20px !important;
    }
    
    /* Mengubah warna teks instruksi di dalam kotak upload */
    [data-testid="stFileUploader"] section > div, 
    [data-testid="stFileUploader"] section span,
    [data-testid="stFileUploader"] section small {
        color: #555555 !important;
    }

    /* Mengubah tombol 'Browse files' menjadi hijau */
    [data-testid="stFileUploader"] button {
        background-color: #2E7D32 !important;
        color: white !important;
        border: none !important;
    }

    /* Header Custom (Pengecualian Warna Putih) */
    .app-header {
        background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%);
        padding: 30px 20px;
        border-radius: 0 0 25px 25px;
        color: white !important; /* Force White */
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(46, 125, 50, 0.2);
    }
    .app-header h1 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        font-size: 28px;
        margin: 0;
        padding: 0;
        color: white !important; /* Force White */
    }
    .app-header p {
        font-size: 14px;
        opacity: 0.9;
        margin-top: 5px;
        color: white !important; /* Force White */
    }

    /* Card Styling */
    .css-1r6slb0, .stMarkdown {
        font-family: 'Segoe UI', sans-serif;
    }
    
    .info-card {
        background: white;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 15px;
        border: 1px solid #f0f0f0;
    }

    /* Button Styling - Pill Shape */
    .stButton>button {
        background: #2E7D32;
        color: white !important; /* Force White */
        border-radius: 50px;
        border: none;
        height: 48px;
        font-weight: 600;
        width: 100%;
        box-shadow: 0 4px 6px rgba(46, 125, 50, 0.2);
        transition: transform 0.1s;
    }
    .stButton>button:hover {
        background: #1B5E20;
        transform: scale(1.02);
    }
    
    /* Status Badges (Pengecualian Warna Khusus) */
    .badge {
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        margin-top: 10px;
        animation: fadeIn 0.5s;
    }
    .badge-fresh { background-color: #e8f5e9; color: #1b5e20 !important; border: 1px solid #c8e6c9; }
    .badge-rotten { background-color: #ffebee; color: #b71c1c !important; border: 1px solid #ffcdd2; }
    .badge-warning { background-color: #fff3e0; color: #e65100 !important; border: 1px solid #ffe0b2; }

    /* Typography */
    h3 { font-size: 18px; font-weight: 600; color: #333 !important; }
    .small-text { font-size: 12px; color: #888 !important; }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD DATA & MODEL
# ==========================================

@st.cache_data
def load_csv_data():
    try:
        df = pd.read_csv("OCR dataset.csv")
        return df
    except FileNotFoundError:
        return None

df_sayuran = load_csv_data()

@st.cache_resource
def load_vision_model():
    try:
        # Ganti dengan nama model terbaik Anda
        model = tf.keras.models.load_model('best_model.h5', compile=False)
        return model
    except Exception:
        return None

model = load_vision_model()

@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['en', 'id'], gpu=False)

ocr_reader = load_ocr_reader()

# ==========================================
# 3. KONFIGURASI & HELPER
# ==========================================

VALID_VEGETABLES = [
    "tomato", "potato", "pare", "okra"
]

CLASS_NAMES = [
    "fresh okra", "fresh pare", "fresh potato", "fresh tomato", 
    "rotten okra", "rotten pare", "rotten potato", "rotten tomato"
]

translation_map = {
    "Tomat": "Tomato", "Kentang": "Potato", "Pare": "Bitter Fruit", 
    "Bitter melon": "Bitter Fruit", "Bayam": "Spinach", "Paprika": "Bell Pepper", 
    "Timun": "Cucumber", "Terong": "Eggplant", "Brokoli": "Broccoli", 
    "Wortel": "Carrot", "Okra": "Okra"
}

# --- PERBAIKAN 1: Mapping Otomatis (Ignore Case) ---
# Membuat dictionary baru agar "pare", "Pare", "PARE" otomatis terdeteksi
translation_map_lower = {k.lower(): v for k, v in translation_map.items()}

def get_vege_info(nama_input):
    """Mencari info detail sayuran dari CSV berdasarkan nama"""
    if df_sayuran is None: return None
    
    # Bersihkan input
    nama_clean = nama_input.lower().strip()
    
    # Cari nama Inggris-nya
    nama_search = translation_map_lower.get(nama_clean, nama_input)

    # Normalisasi nama kolom CSV agar aman (ubah ke lowercase & hilangkan spasi)
    df_temp = df_sayuran.copy()
    df_temp.columns = df_temp.columns.str.strip().str.lower()

    # Cari di kolom 'name' (hasil lower dari 'Name')
    item = df_temp[df_temp['name'].str.lower() == nama_search.lower()]
    
    if item.empty:
        item = df_temp[df_temp['name'].str.lower().str.contains(nama_search.lower())]
        if item.empty: return None

    data = item.iloc[0]
    
    try:
        # AKSES KOLOM SESUAI HEADER CSV YANG SUDAH DI-LOWERCASE
        # Header Asli: "Shelf Life (days)" -> Jadi: "shelf life (days)"
        life_str = str(data['shelf life (days)'])
        max_life = int(''.join(filter(str.isdigit, life_str))) if any(c.isdigit() for c in life_str) else 7

        return {
            "name": data['name'],                               # Asli: Name
            "storage": data['storage requirements'],            # Asli: Storage Requirements
            "benefit": data['health benefits'],                 # Asli: Health Benefits (Pakai 's')
            "max_life": max_life
        }
    except KeyError as e:
        st.error(f"Error Database: Kolom {e} tidak ditemukan. Pastikan header CSV:// filepath: d:\kuliah sem 3\AI LETSGOOOOo\FreshGuard\streamlit_app.py")

def hitung_shelf_life_ocr(nama_input, tanggal_beli):
    data = get_vege_info(nama_input)
    if not data: return None
    
    hari_ini = datetime.now().date()
    lama_simpan = (hari_ini - tanggal_beli).days
    sisa_waktu = data['max_life'] - lama_simpan
    
    data['days_stored'] = lama_simpan
    data['remaining'] = sisa_waktu
    return data

def proses_ocr_struk(reader, image):
    image_np = np.array(image)
    result = reader.readtext(image_np, detail=0)
    full_text = " ".join(result).lower()
    
    detected_items = []
    for key in translation_map.keys():
        if key.lower() in full_text:
            detected_items.append(key)
            
    if not detected_items and df_sayuran is not None:
        for name in df_sayuran['Name'].unique():
            if name.lower() in full_text:
                detected_items.append(name)

    return detected_items, full_text

def proses_gambar_cv(model, image):
    if model is None: return None, 0, []
    
    target_size = (224, 224)
    try:
        if hasattr(model, 'input_shape'):
            shape = model.input_shape
            if shape and len(shape) == 4:
                h, w = shape[1], shape[2]
                if h and w: target_size = (w, h)
    except: pass

    image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    normalized_image_array = (img_array.astype(np.float32) / 255.0)
    data = np.expand_dims(normalized_image_array, axis=0)

    prediction = model.predict(data)
    idx = np.argmax(prediction)
    confidence = prediction[0][idx]
    label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else "Unknown"
    
    top_indices = np.argsort(prediction[0])[-3:][::-1]
    top_predictions = []
    for i in top_indices:
        if i < len(CLASS_NAMES):
            top_predictions.append((CLASS_NAMES[i], float(prediction[0][i])))
            
    return label, confidence, top_predictions

# ==========================================
# 4. USER INTERFACE (APP LAYOUT)
# ==========================================

# Header Section
st.markdown("""
    <div class='app-header'>
        <h1>FreshGuard</h1>
        <p>Smart Food Assistant</p>
    </div>
""", unsafe_allow_html=True)

# Navigation
tab1, tab2 = st.tabs(["🧾 Scan Struk", "📷 Cek Fisik"])

# --- TAB 1: OCR ---
with tab1:
    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.write("**Monitor Kadaluarsa**")
    st.caption("Upload struk belanja untuk otomatis mencatat tanggal beli.")
    
    # Input Tanggal
    tgl_beli = st.date_input("Tanggal Beli", value=datetime.now())
    
    # GANTI DARI SELECTBOX KE RADIO BUTTON (Agar opsi terlihat jelas)
    input_method = st.radio("Metode Input:", ["Upload File", "Kamera"], horizontal=True, key="ocr_src")

    img_file = None
    if input_method == "Upload File":
        img_file = st.file_uploader("Pilih Gambar Struk", type=["jpg", "png", "jpeg"])
    else:
        img_file = st.camera_input("Ambil Foto Struk")

    if img_file:
        st.image(img_file, use_container_width=True)
        if st.button("Analisa Struk"):
            with st.spinner("Scanning..."):
                img = Image.open(img_file)
                items, raw = proses_ocr_struk(ocr_reader, img)
                
                if items:
                    hasil = hitung_shelf_life_ocr(items[0], tgl_beli)
                    if hasil:
                        # Result Card
                        st.markdown("---")
                        st.subheader(f"📦 {hasil['name']}")
                        
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Disimpan", f"{hasil['days_stored']} hr")
                        c2.metric("Max", f"{hasil['max_life']} hr")
                        c3.metric("Sisa", f"{hasil['remaining']} hr", delta=hasil['remaining'])
                        
                        if hasil['remaining'] < 0:
                            st.markdown(f"<div class='badge badge-rotten'>⛔ EXPIRED ({abs(hasil['remaining'])} hari lalu)</div>", unsafe_allow_html=True)
                        elif hasil['remaining'] <= 2:
                            st.markdown(f"<div class='badge badge-warning'>⚠ SEGERA KONSUMSI</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div class='badge badge-fresh'>✅ AMAN DIKONSUMSI</div>", unsafe_allow_html=True)
                        
                        with st.expander("💡 Info Penyimpanan & Manfaat"):
                            st.write(f"**Cara Simpan:** {hasil['storage']}")
                            st.write(f"**Manfaat:** {hasil['benefit']}")
                else:
                    st.error("Tidak ditemukan nama sayuran di struk.")
    st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 2: COMPUTER VISION ---
with tab2:
    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.write("**Cek Kesegaran (AI)**")
    st.caption("Deteksi kualitas fisik sayuran secara real-time.")
    
    # GANTI DARI SELECTBOX KE RADIO BUTTON
    input_method_cv = st.radio("Metode Input:", ["Upload File", "Kamera"], horizontal=True, key="cv_src")
    
    img_cv = None
    if input_method_cv == "Upload File":
        img_cv = st.file_uploader("Upload Foto Sayur", type=["jpg", "png", "jpeg"], key="cv_up")
    else:
        img_cv = st.camera_input("Ambil Foto Sayur", key="cv_cam")

    if img_cv:
        image = Image.open(img_cv).convert("RGB")
        st.image(image, use_container_width=True)
        
        if st.button("Cek Kondisi"):
            if model is None:
                st.error("Model AI Error.")
            else:
                with st.spinner('Menganalisa tekstur...'):
                    label, conf, top3 = proses_gambar_cv(model, image)
                    
                    label_lower = label.lower()
                    detected_name = None
                    for v in VALID_VEGETABLES:
                        if v in label_lower:
                            detected_name = v.capitalize()
                            break
                    
                    condition = "Unknown"
                    if "fresh" in label_lower: condition = "Fresh"
                    elif "rotten" in label_lower: condition = "Rotten"

                    st.markdown("---")
                    if detected_name:
                        # Tampilan Hasil Utama
                        if condition == "Fresh":
                            st.markdown(f"""
                            <div class='badge badge-fresh'>
                                <h2 style='margin:0; color:#1b5e20'>✨ SEGAR</h2>
                                <p style='margin:0'>{detected_name} ({conf*100:.0f}%)</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # --- FITUR BARU: REKOMENDASI PENYIMPANAN ---
                            # Ambil data dari CSV berdasarkan nama sayur yang terdeteksi
                            info_sayur = get_vege_info(detected_name)
                            
                            if info_sayur:
                                st.markdown("### 💡 Rekomendasi AI")
                                
                                # --- TAMBAHAN: Tampilkan Shelf Life ---
                                col_life, col_empty = st.columns([1, 1])
                                with col_life:
                                    st.metric("Estimasi Umur Simpan", f"{info_sayur['max_life']} Hari")
                                
                                st.info(f"**Cara Penyimpanan:**\n\n{info_sayur['storage']}")
                                st.success(f"**Manfaat Kesehatan:**\n\n{info_sayur['benefit']}")
                            else:
                                st.caption("Info detail penyimpanan tidak ditemukan di database.")
                                
                        elif condition == "Rotten":
                            st.markdown(f"""
                            <div class='badge badge-rotten'>
                                <h2 style='margin:0; color:#b71c1c'>🍄 BUSUK</h2>
                                <p style='margin:0'>{detected_name} ({conf*100:.0f}%)</p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.warning("Sayuran ini terdeteksi mengalami pembusukan. Sebaiknya tidak dikonsumsi atau pisahkan bagian yang busuk.")
                    else:
                        st.markdown(f"<div class='badge badge-warning'>❓ {label}</div>", unsafe_allow_html=True)
                        st.caption("Objek tidak dikenali sebagai sayuran yang terdaftar.")
                        
                    # Debug Toggle
                    with st.expander("🔍 Lihat Detail Probabilitas (Debug)"):
                        st.write("Apa yang dilihat AI?")
                        for lbl, conf in top3:
                            st.write(f"**{lbl}**: {conf*100:.1f}%")
                            st.progress(conf)
                        
                        st.info("""
                        **Tips jika salah deteksi:**
                        1. Pastikan urutan `CLASS_NAMES` di kode sama persis dengan urutan abjad folder training Anda.
                        2. Cek pencahayaan foto.
                        3. Pastikan background foto bersih (polos).
                        """)
    
    st.markdown("</div>", unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
    <div class='footer'>
        <p>&copy; 2025 <b>FreshGuard</b>. All Rights Reserved.<br>
        Developed by <b>Team Haniel</b></p>
    </div>
""", unsafe_allow_html=True)