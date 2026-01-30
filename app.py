import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# === Load model dan fitur ===
with open("finalmodel_terbaru_klasifikasiPCOS.sav", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
selected_features = bundle["features"]

# Hilangkan kemungkinan fitur duplikat (aman, tidak mengubah logika)
selected_features = list(dict.fromkeys(selected_features))

# === Judul dan instruksi ===
st.title("🧬 Prediksi PCOS dengan Random Forest")
st.write("Masukkan data berikut untuk melakukan prediksi:")
st.caption("⚠️ Jangan gunakan tanda koma (,) — gunakan tanda titik (.) untuk angka desimal. Bisa diisi lebih/kurang dari contoh range.")

# === Inisialisasi session state untuk riwayat ===
if "history" not in st.session_state:
    st.session_state.history = []

# === Mapping deskripsi & contoh input ===
feature_info = {
    "Age (yrs)": {"desc": "Masukkan usia pasien", "range": "Contoh: 15 - 45"},
    "Follicle No. (R)": {"desc": "Masukkan jumlah folikel di ovarium kanan", "range": "Contoh: 0 - 25"},
    "Follicle No. (L)": {"desc": "Masukkan jumlah folikel di ovarium kiri", "range": "Contoh: 0 - 25"},
    "Skin darkening (Y/N)": {"desc": "Apakah terdapat penggelapan kulit", "range": "Pilih: Tidak (0) / Ya (1)"},
    "Weight gain(Y/N)": {"desc": "Apakah terjadi peningkatan berat badan", "range": "Pilih: Tidak (0) / Ya (1)"},
    "hair growth(Y/N)": {"desc": "Apakah terjadi pertumbuhan rambut berlebih", "range": "Pilih: Tidak (0) / Ya (1)"},
    "AMH(ng/mL)": {"desc": "Masukkan nilai Anti-Müllerian Hormone", "range": "Contoh: 1 - 10"},
    "Cycle(R/I)": {"desc": "Tipe siklus menstruasi", "range": "Pilih: Regular = Teratur (2) atau Irregular = Tidak Teratur (4)"},
    "LH(mIU/mL)": {"desc": "Masukkan nilai Luteinizing Hormone", "range": "Contoh: 2 - 20"},
    "FSH(mIU/mL)": {"desc": "Masukkan nilai Follicle-Stimulating Hormone", "range": "Contoh: 3 - 15"},
}

# === Form input ===
user_input = {}
st.markdown("### 🧾 Form Input Data")

for feature in selected_features:
    if feature in feature_info:
        st.markdown(
            f"**{feature}**  \nℹ️ {feature_info[feature]['desc']} — {feature_info[feature]['range']}"
        )

    if feature in ["Skin darkening (Y/N)", "Weight gain(Y/N)", "hair growth(Y/N)"]:
        pilihan = st.selectbox(
            feature,
            ["Pilih...", "Tidak (0)", "Ya (1)"],
            label_visibility="collapsed",
            key=feature
        )
        user_input[feature] = None if pilihan == "Pilih..." else (1.0 if "Ya" in pilihan else 0.0)

    elif feature == "Cycle(R/I)":
        pilihan = st.selectbox(
            feature,
            ["Pilih...", "Regular = Teratur (2)", "Irregular = Tidak Teratur (4)"],
            label_visibility="collapsed",
            key=feature
        )
        user_input[feature] = None if pilihan == "Pilih..." else (4.0 if "Irregular" in pilihan else 2.0)

    else:
        val = st.text_input(feature, "", label_visibility="collapsed", key=feature)
        if val.strip() == "":
            user_input[feature] = None
        else:
            try:
                user_input[feature] = float(val.replace(",", "."))
            except ValueError:
                st.error(f"Input {feature} harus berupa angka!")
                user_input[feature] = None

    # Tombol Aksi
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    pred_btn = st.button("🔍 Prediksi")
with col2:
    reset_btn = st.button("🔁 Reset Hasil Prediksi")
with col3:
    history_btn = st.button("📊 Lihat Riwayat Prediksi (jika ada)")

# === Reset Form ===
if reset_btn:
    for feature in selected_features:
        if feature in st.session_state:
            del st.session_state[feature]
        st.rerun()

# === Jika tombol prediksi ditekan ===
if pred_btn:
    if any(v is None for v in user_input.values()):
        st.warning("⚠️ Harap isi semua data sebelum melakukan prediksi.")
    else:
        input_df = pd.DataFrame([user_input], columns=selected_features)

        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        # === Data yang diuji ===
        st.subheader("📋 Data yang Diuji")
        satuan_map = {
            "Age (yrs)": "tahun",
            "Follicle No. (R)": "folikel",
            "Follicle No. (L)": "folikel",
            "AMH(ng/mL)": "ng/mL",
            "FSH(mIU/mL)": "mIU/mL",
            "LH(mIU/mL)": "mIU/mL"
        }

        for feature, value in user_input.items():
            if feature in ["Skin darkening (Y/N)", "Weight gain(Y/N)", "hair growth(Y/N)"]:
                st.write(f"**{feature}:** {value} (0=Tidak, 1=Ya)")
            elif feature == "Cycle(R/I)":
                st.write(f"**{feature}:** {value} (0=Regular, 1=Irregular)")
            else:
                st.write(f"**{feature}:** {value} {satuan_map.get(feature, '')}")

        # === Hasil prediksi ===
        st.markdown("---")
        if prediction == 1:
            st.markdown(
                f"<h2 style='text-align:center;color:#FF4500;'>⚠️ Hasil Prediksi: PCOS</h2>"
                f"<h3 style='text-align:center;'>Probabilitas: {probabilities[1]:.2%}</h3>",
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div style="
                    background-color: rgba(255, 69, 0, 0.15);
                    padding: 16px;
                    border-radius: 10px;
                    border-left: 6px solid #FF4500;
                    margin-top: 16px;">
                    🧾 <strong>Rekomendasi Sistem:</strong><br>
                    Sistem menyarankan untuk melakukan konsultasi ke dokter spesialis kandungan untuk pemeriksaan lebih lanjut.
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<h2 style='text-align:center;color:#1E90FF;'>💡 Hasil Prediksi: Tidak PCOS</h2>"
                f"<h3 style='text-align:center;'>Probabilitas: {probabilities[0]:.2%}</h3>",
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div style="
                    background-color: rgba(30, 144, 255, 0.15);
                    padding: 16px;
                    border-radius: 10px;
                    border-left: 6px solid #1E90FF;
                    margin-top: 16px;">
                    🧾 <strong>Rekomendasi Sistem:</strong><br>
                    Tetap jaga pola hidup sehat dan lakukan pemeriksaan rutin. Segera konsultasi ke dokter apabila muncul keluhan lain.
                </div>
                """,
                unsafe_allow_html=True
            )
        st.caption("⚠️ *Catatan: Sistem ini hanya berfungsi sebagai alat bantu prediksi, bukan diagnosis medis.*")

        # === Simpan ke riwayat ===
        st.session_state.history.append({
            "Prediksi": "PCOS" if prediction == 1 else "Tidak PCOS",
            "Probabilitas_PCOS": f"{probabilities[1]:.2%}",
            "Probabilitas_Tidak_PCOS": f"{probabilities[0]:.2%}",
            **user_input
        })

        # === Visualisasi probabilitas ===
        st.subheader("📊 Visualisasi Probabilitas")
        fig, ax = plt.subplots()
        ax.bar(["Tidak PCOS", "PCOS"], probabilities)
        ax.set_ylabel("Probabilitas")
        ax.set_ylim(0, 1)
        for i, v in enumerate(probabilities):
            ax.text(i, v + 0.02, f"{v:.2%}", ha="center", fontsize=10)
        st.pyplot(fig)

# === Tampilkan riwayat prediksi ===
if history_btn:
    if len(st.session_state.history) > 0:
        st.subheader("📜 Riwayat Prediksi")
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)
    else:
        st.info("Belum ada riwayat prediksi yang tersimpan.")

















