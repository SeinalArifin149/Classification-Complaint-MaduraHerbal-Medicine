import re
import string
from pathlib import Path

import joblib
import nltk
from nltk.corpus import stopwords
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin

# =====================================================
# 1. KONFIGURASI APLIKASI
# =====================================================

st.set_page_config(
    page_title="Klasifikasi Keluhan Jamu Madura",
    page_icon="🌿",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model_pipeline.pkl"
KAMUS_PATH = BASE_DIR / "kamuskatabaku.xlsx"
DATA_PATH = BASE_DIR / "data_jamu_322.xlsx"


# =====================================================
# 2. LOAD KAMUS KATA BAKU
# =====================================================

@st.cache_data
def load_kamus():
    kamus_df = pd.read_excel(KAMUS_PATH)

    # Kolom pertama = kata tidak baku (slang), Kolom kedua = kata baku (formal)
    kamus_dict = dict(
        zip(
            kamus_df.iloc[:, 0].astype(str).str.lower(),
            kamus_df.iloc[:, 1].astype(str).str.lower()
        )
    )

    return kamus_dict


# =====================================================
# 3. TEXT PREPROCESSOR (SINKRON DENGAN MODEL TRAINED)
# =====================================================

class TextPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, kamus_dict=None):
        self.kamus_dict = kamus_dict if kamus_dict is not None else {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.preprocess(teks) for teks in X]

    def preprocess(self, teks):
        teks = str(teks).lower()

        # Menghapus URL
        teks = re.sub(r"http\S+|www\S+", " ", teks)

        # Menghapus angka
        teks = re.sub(r"\d+", " ", teks)

        # Menghapus tanda baca
        teks = teks.translate(
            str.maketrans("", "", string.punctuation)
        )

        # Menghapus karakter selain huruf dan spasi
        teks = re.sub(r"[^a-zA-Z\s]", " ", teks)

        # Menghapus spasi berlebih
        teks = re.sub(r"\s+", " ", teks).strip()

        # Normalisasi kata tidak baku (menggunakan getattr agar aman dari AttributeError)
        kamus = getattr(self, "kamus_dict", {})
        kata = teks.split()
        kata = [kamus.get(k, k) for k in kata]
        teks = " ".join(kata)

        # Menghapus stopwords Bahasa Indonesia
        try:
            stop_words = set(stopwords.words("indonesian"))
            kata = [
                k for k in teks.split()
                if k not in stop_words
            ]
            teks = " ".join(kata)

        except LookupError:
            pass

        return teks


# =====================================================
# 4. LOAD MODEL DAN DATASET
# =====================================================

@st.cache_resource
def load_model():
    model_obj = joblib.load(MODEL_PATH)
    kamus = load_kamus()

    # Menyuntikkan kamus_dict ke transformer TextPreprocessor di dalam pipeline pickle
    if hasattr(model_obj, "named_steps"):
        for step_obj in model_obj.named_steps.values():
            if isinstance(step_obj, TextPreprocessor):
                step_obj.kamus_dict = kamus
    elif hasattr(model_obj, "steps"):
        for _, step_obj in model_obj.steps:
            if isinstance(step_obj, TextPreprocessor):
                step_obj.kamus_dict = kamus

    return model_obj


@st.cache_data
def load_dataset():
    df = pd.read_excel(DATA_PATH)
    df.columns = df.columns.astype(str).str.strip()
    return df


try:
    kamus_dict = load_kamus()
    model = load_model()
    data_jamu = load_dataset()

except FileNotFoundError as e:
    st.error(f"File tidak ditemukan: {e}")
    st.stop()

except Exception as e:
    st.error(f"Gagal memuat data atau model: {e}")
    st.stop()


# =====================================================
# 5. FUNGSI NORMALISASI KATEGORI
# =====================================================

def normalisasi_kategori(teks):
    teks = str(teks).strip().lower()
    teks = re.sub(r"^jamu\s*", "", teks)
    teks = re.sub(r"[^a-z0-9]", "", teks)
    return teks


# =====================================================
# 6. HEADER APLIKASI
# =====================================================

st.title("🌿 Klasifikasi Keluhan Jamu Madura")

st.write(
    """
    Aplikasi ini mengklasifikasikan keluhan pengguna ke dalam kategori **Jamu Madura**
    menggunakan algoritma **Multinomial Naive Bayes** serta memberikan rekomendasi produk
    jamu yang relevan dengan keluhan Anda.
    """
)

st.divider()


# =====================================================
# 7. FORM INPUT KELUHAN
# =====================================================

st.subheader("Masukkan Keluhan Pengguna")

keluhan = st.text_area(
    "Tuliskan keluhan:",
    placeholder="Contoh: Saya mengalami keputihan berlebih dan merasa gatal di organ intim...",
    height=150
)

tombol_prediksi = st.button(
    "Klasifikasikan Keluhan",
    type="primary",
    use_container_width=True
)


# =====================================================
# 8. PROSES KLASIFIKASI
# =====================================================

if tombol_prediksi:

    if not keluhan.strip():

        st.warning("Silakan masukkan keluhan terlebih dahulu.")

    else:

        try:
            # Prediksi kategori menggunakan pipeline
            prediksi = str(model.predict([keluhan])[0])

            # Menghitung probabilitas tiap kelas
            probabilitas = model.predict_proba([keluhan])[0]

            # Mengambil nama kelas dari model
            kelas = model.classes_

            # Menghitung tingkat keyakinan (confidence)
            confidence = max(probabilitas) * 100

            st.divider()

            # -----------------------------------------
            # HASIL PREDIKSI
            # -----------------------------------------

            st.subheader("Hasil Klasifikasi")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label="Kategori Keluhan",
                    value=prediksi
                )

            with col2:
                st.metric(
                    label="Tingkat Keyakinan Model",
                    value=f"{confidence:.2f}%"
                )

            # -----------------------------------------
            # PROBABILITAS KATEGORI
            # -----------------------------------------

            st.subheader("Probabilitas Setiap Kategori")

            df_probabilitas = pd.DataFrame({
                "Kategori": kelas,
                "Probabilitas (%)": probabilitas * 100
            })

            df_probabilitas = df_probabilitas.sort_values(
                by="Probabilitas (%)",
                ascending=False
            )

            st.bar_chart(
                df_probabilitas.set_index("Kategori")
            )

            st.dataframe(
                df_probabilitas.style.format({
                    "Probabilitas (%)": "{:.2f}%"
                }),
                use_container_width=True,
                hide_index=True
            )

            # =================================================
            # 9. REKOMENDASI PRODUK JAMU BERDASARKAN KELUHAN
            # =================================================

            st.divider()

            st.subheader("🌿 Rekomendasi Produk Jamu Berdasarkan Keluhan")

            if "KHASIAT" not in data_jamu.columns:

                st.error("Kolom KHASIAT tidak ditemukan dalam dataset.")

            else:

                # 1. Cleaning input keluhan pengguna
                preprocessor_temp = TextPreprocessor(kamus_dict=kamus_dict)
                keluhan_clean = preprocessor_temp.preprocess(keluhan)
                kata_keluhan = set(keluhan_clean.split())

                # 2. Fungsi hitung relevansi kata kunci keluhan dengan khasiat jamu
                def hitung_relevansi(khasiat_teks):
                    if pd.isna(khasiat_teks):
                        return 0
                    khasiat_clean = preprocessor_temp.preprocess(str(khasiat_teks))
                    kata_khasiat = set(khasiat_clean.split())
                    return len(kata_keluhan.intersection(kata_khasiat))

                data_rekomendasi = data_jamu.copy()

                # Hitung skor kecocokan kata kunci
                data_rekomendasi["skor_relevansi"] = data_rekomendasi["KHASIAT"].apply(hitung_relevansi)

                # Normalisasi nama kategori untuk matching
                kategori_prediksi = normalisasi_kategori(prediksi)
                if "KATEGORI" in data_rekomendasi.columns:
                    data_rekomendasi["kategori_normal"] = (
                        data_rekomendasi["KATEGORI"]
                        .fillna("")
                        .apply(normalisasi_kategori)
                    )

                # Filter Produk: Pilih kategori hasil prediksi & yang memiliki khasiat paling cocok dengan keluhan
                data_terkait = data_rekomendasi[
                    (data_rekomendasi["kategori_normal"] == kategori_prediksi) &
                    (data_rekomendasi["skor_relevansi"] > 0)
                ].sort_values(by="skor_relevansi", ascending=False)

                # Fallback: Jika tidak ada kata spesifik yang match, tampilkan produk di kategori prediksi
                if data_terkait.empty and "KATEGORI" in data_rekomendasi.columns:
                    data_terkait = data_rekomendasi[
                        data_rekomendasi["kategori_normal"] == kategori_prediksi
                    ]

                if not data_terkait.empty:

                    st.success(
                        f"Ditemukan {len(data_terkait)} produk jamu "
                        f"yang sesuai dengan keluhan Anda dalam kategori **{prediksi}**."
                    )

                    # Urutan kolom yang ingin ditampilkan (Nama Jamu ditaruh di posisi paling awal)
                    kolom_tampilan_ideal = [
                        "NAMA JAMU",
                        "JAMU",
                        "NAMA PRODUK",
                        "NAMA",
                        "KATEGORI",
                        "KHASIAT",
                        "KANDUNGAN",
                        "ATURAN MINUM",
                        "EFEK SAMPING",
                        "JENIS",
                        "PRODUSEN"
                    ]

                    # Mengambil kolom yang benar-benar tersedia di file excel
                    kolom_tersedia = [
                        kolom for kolom in kolom_tampilan_ideal
                        if kolom in data_terkait.columns
                    ]

                    # Tampilkan tabel rekomendasi produk
                    st.dataframe(
                        data_terkait[kolom_tersedia].fillna("-"),
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.warning(
                        f"Belum ditemukan produk jamu yang sesuai untuk keluhan tersebut dalam kategori **{prediksi}**."
                    )

                    with st.expander("Informasi pencocokan kategori"):

                        st.write("Hasil prediksi:", prediksi)

                        st.write(
                            "Kategori setelah normalisasi:",
                            kategori_prediksi
                        )

                        st.write(
                            "Daftar Kategori dalam dataset:",
                            data_jamu["KATEGORI"].unique().tolist()
                        )

        except Exception as e:

            st.error(f"Terjadi kesalahan saat klasifikasi: {e}")


# =====================================================
# 10. INFORMASI DATASET (SIDEBAR)
# =====================================================

with st.sidebar:

    st.header("Informasi Dataset")

    st.metric(
        "Jumlah Data Produk",
        len(data_jamu)
    )

    st.metric(
        "Jumlah Kategori",
        data_jamu["KATEGORI"].nunique()
        if "KATEGORI" in data_jamu.columns else 0
    )

    st.divider()

    st.caption(
        "Sistem Klasifikasi Keluhan & Rekomendasi Jamu Madura"
    )