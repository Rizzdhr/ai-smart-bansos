from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')

def generate_alasan(penghasilan, usia, pekerjaan, kondisi_rumah, status):
    alasan = []

    # Penghasilan
    if penghasilan < 1000000:
        alasan.append({"icon": "✓", "teks": "Penghasilan sangat rendah (< Rp 1 juta)", "positif": True})
    elif penghasilan < 2500000:
        alasan.append({"icon": "✓", "teks": "Penghasilan rendah (< Rp 2,5 juta)", "positif": True})
    elif penghasilan < 5000000:
        alasan.append({"icon": "~", "teks": "Penghasilan menengah", "positif": None})
    else:
        alasan.append({"icon": "✗", "teks": "Penghasilan cukup tinggi (> Rp 5 juta)", "positif": False})

    # Usia
    if usia >= 60:
        alasan.append({"icon": "✓", "teks": "Usia lanjut (≥ 60 tahun)", "positif": True})
    elif usia <= 17:
        alasan.append({"icon": "~", "teks": "Usia anak/remaja", "positif": None})
    else:
        alasan.append({"icon": "~", "teks": "Usia produktif", "positif": None})

    # Pekerjaan (0=tidak bekerja, 1=buruh, 2=karyawan, 3=profesional, 4=pejabat)
    pekerjaan_label = {
        0: ("✓", "Kategori pekerjaan: Tidak/belum bekerja", True),
        1: ("✓", "Kategori pekerjaan: Buruh/pekerja informal", True),
        2: ("~", "Kategori pekerjaan: Karyawan/pegawai", None),
        3: ("✗", "Kategori pekerjaan: Tenaga profesional", False),
        4: ("✗", "Kategori pekerjaan: Pengusaha/pejabat", False),
    }
    icon, teks, positif = pekerjaan_label.get(pekerjaan, ("~", "Kategori pekerjaan: Tidak diketahui", None))
    alasan.append({"icon": icon, "teks": teks, "positif": positif})

    # Kondisi rumah (0=buruk, 1=sedang, 2=baik)
    if kondisi_rumah == 0:
        alasan.append({"icon": "✓", "teks": "Kondisi rumah buruk", "positif": True})
    elif kondisi_rumah == 1:
        alasan.append({"icon": "~", "teks": "Kondisi rumah sedang", "positif": None})
    elif kondisi_rumah == 2:
        alasan.append({"icon": "✗", "teks": "Kondisi rumah baik", "positif": False})
    else:
        alasan.append({"icon": "✗", "teks": "Kondisi rumah tidak dapat dianalisis", "positif": False})

    return alasan


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    penghasilan   = data['penghasilan']
    usia          = data['usia']
    pekerjaan     = data['pekerjaan']
    kondisi_rumah = data['kondisi_rumah']


    # Jika kondisi rumah tidak terdeteksi, tolak langsung
    if kondisi_rumah == 3:
        return jsonify({
            "status": "ditolak",
            "skor": 0.05,
            "alasan": [
                {"icon": "✗", "teks": "Foto rumah tidak dapat dianalisis, silakan upload ulang", "positif": False}
            ]
        })

    fitur = [[penghasilan, usia, pekerjaan, kondisi_rumah]]

    prob = model.predict_proba(fitur)[0][1]

    # Cap berdasarkan kondisi rumah
    if kondisi_rumah == 2:      # rumah baik
        prob = min(prob, 0.30)
    elif kondisi_rumah == 1:    # rumah sedang
        prob = min(prob, 0.70)
    elif kondisi_rumah == 0:    # rumah buruk
        prob = min(prob, 0.95)

    # Clamp selalu jalan
    prob = max(0.05, min(0.95, float(prob)))


    status = "diterima" if prob > 0.5 else "ditolak"

    alasan = generate_alasan(penghasilan, usia, pekerjaan, kondisi_rumah, status)

    return jsonify({
        "status": status,
        "skor": float(prob),
        "alasan": alasan
    })

if __name__ == '__main__':
    app.run(port=5000)
