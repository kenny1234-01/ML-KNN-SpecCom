from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# โหลดโมเดลและ label_encoders
model = joblib.load('knn_model_k3.pkl')
label_encoders = joblib.load('label_encoders.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])

    # ตรวจสอบและแปลงข้อมูลด้วย label_encoders
    for column in df.columns:
        if column in label_encoders:
            encoder = label_encoders[column]
            df[column] = encoder.transform(df[column])

    # ทำการคาดการณ์
    prediction = model.predict_proba(df)
    classes = model.classes_
    def rank_classes(prob):
        # เรียงลำดับ class จากความน่าจะเป็นที่มากที่สุดไปหาน้อยที่สุด
        ranked_indices = np.argsort(prob)[::-1]
        ranked_classes = [classes[i] for i in ranked_indices]
        ranked_probs = [prob[i] for i in ranked_indices]

        return pd.Series({
            'Rank1': ranked_classes[0],
            'Rank2': ranked_classes[1],
            'Rank3': ranked_classes[2],
            'ProbabilityRank1': ranked_probs[0],
            'ProbabilityRank2': ranked_probs[1],
            'ProbabilityRank3': ranked_probs[2],
        })
    
    # ใช้ฟังก์ชัน rank_classes เพื่อคำนวณและเพิ่มคอลัมน์อันดับใน DataFrame
    ranking_df = pd.DataFrame(prediction).apply(rank_classes, axis=1)

    # รวมข้อมูลของสเปคคอมพิวเตอร์และอันดับ class เข้าไว้ใน DataFrame เดียวกัน
    result_df = pd.concat([df, ranking_df], axis=1)
    print(result_df)

    return jsonify({'prediction': result_df.to_dict(orient='records')})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
