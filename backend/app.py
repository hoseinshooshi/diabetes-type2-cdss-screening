from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os
from datetime import datetime
from openai import OpenAI 
from dotenv import load_dotenv



app = Flask(__name__)
CORS(app)
load_dotenv()

# DeepSeek API Configuration
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
class DeepSeekMedicalAdvisor:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = None
        self.is_available = False
        self._initialize_deepseek()


    def _initialize_deepseek(self):
        """راه‌‌اندازی DeepSeek با کتابخانه رسمی"""
        try:
            print("🌐 در حال راه‌اندازی اتصال به DeepSeek …") 
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com"
            )

            # connect test
            test_response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": "سلام"},
                ],
                stream=False
            )

            print("✅ پاسخ تستی از DeepSeek دریافت شد:", test_response.choices[0].message.content)
            self.is_available = True
            print("✅ DeepSeek API با موفقیت راه‌اندازی شد")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ خطا در راه‌اندازی DeepSeek: {e}")
            self.is_available = False

    def generate_personalized_advice(self, prediction, features, sex, age):
        """تولید توصیه‌های شخصی‌شده با DeepSeek"""
        if not self.is_available:
            print("⚠️ DeepSeek در دسترس نیست — استفاده از توصیهٔ جایگزین")
            return self._get_fallback_recommendations(prediction)

        try:
            glucose = features[1]
            bp = features[2]
            bmi = features[5]
            skin = features[3]
            insulin = features[4]
            dpf = features[6]

            prompt = f"""
            شما یک پزشک متخصص دیابت هستید. لطفاً تحلیل پزشکی و توصیه‌های شخصی‌شده ارائه دهید:

            📊 اطلاعات بیمار:
            - سن: {age} سال
            - جنسیت: {'زن' if sex == 'female' else 'مرد'}
            - قند خون: {glucose} mg/dL
            - فشار خون: {bp} mmHg
            - BMI: {bmi}
            - ضخامت پوست: {skin} mm
            - انسولین: {insulin} mu U/ml
            - سابقه خانوادگی: {dpf}

            🔍 تشخیص: {'⚠️ دیابت نوع ۲' if prediction == 1 else '✅ وضعیت طبیعی'}

            لطفاً:
            ۱. تحلیل مختصر از وضعیت بیمار ارائه دهید
            ۲. ۳‑۴ توصیه عملی و شخصی‌شده بدهید
            ۳. سطح فوریت را مشخص کنید
            ۴. به زبان فارسی ساده و قابل فهم
            """

            print("📤 ارسال درخواست به DeepSeek …")
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "شما یک پزشک متخصص دیابت هستید. تحلیل پزشکی دقیق و توصیه‌های عملی ارائه دهید."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000,
                stream=False
            )

            response_text = response.choices[0].message.content
            print("📥 پاسخ از DeepSeek دریافت شد")
            return self._parse_deepseek_response(response_text)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ خطا در تولید محتوا با DeepSeek: {e}")
            return self._get_fallback_recommendations(prediction)

    def _parse_deepseek_response(self, response_text):
        """پارس کردن پاسخ DeepSeek"""
        return {
            "analysis": response_text,
            "is_ai_generated": True,
            "source": "DeepSeek AI"
        }

    def _get_fallback_recommendations(self, prediction):
        """توصیه‌های پیش‌فرض اگر DeepSeek خطا داد"""
        if prediction == 1:
            return {
                "analysis": "با توجه به داده‌های ورودی، احتمال دیابت نوع ۲ وجود دارد. توصیه می‌شود با پزشک متخصص مشورت نمایید.",
                "is_ai_generated": False,
                "source": "سیستم پیش‌فرض"
            }
        else:
            return {
                "analysis": "داده‌ها در محدوده طبیعی قرار دارند. ادامه سبک زندگی سالم و غربالگری منظم توصیه می‌شود.",
                "is_ai_generated": False,
                "source": "سیستم پیش‌فرض"
            }


deepseek_advisor = DeepSeekMedicalAdvisor(DEEPSEEK_API_KEY)

class DiabetesModelTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
    
    def load_and_preprocess_data(self):
        """بارگذاری و پیش‌پردازش داده‌ها"""
        print("📥 در حال بارگذاری داده‌ها از PIMA dataset...")
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        columns = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
        ]
        
        try:
            data = pd.read_csv(url, names=columns)
            print(f"✅ داده‌ها با موفقیت بارگذاری شدند. تعداد نمونه‌ها: {len(data)}")
            
            # irrational values
            medical_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
            for col in medical_columns:
                data[col] = data[col].replace(0, np.nan)
                data[col].fillna(data[col].median(), inplace=True)
            
            return data
            
        except Exception as e:
            print(f"❌ خطا در بارگذاری داده‌ها: {e}")
            return None
    
    def train_model(self):
        """آموزش مدل"""
        print("🎯 شروع آموزش مدل Random Forest...")
        data = self.load_and_preprocess_data()
        
        if data is None:
            print("❌ خطا: داده‌ها بارگذاری نشدند")
            return False
        
        X = data.drop("Outcome", axis=1)
        y = data["Outcome"]
        
        # data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # noramlixation
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # model training
        print("⏳ در حال آموزش مدل...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # model valuation
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ مدل با موفقیت آموزش داده شد")
        print(f"📊 دقت مدل روی داده تست: {accuracy:.4f}")
        
        return True
    
    def save_model(self):
        """ذخیره مدل و اسکیلر"""
        try:
            with open("model.pkl", "wb") as f:
                pickle.dump(self.model, f)
            with open("scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)
            print("💾 مدل و اسکیلر با موفقیت ذخیره شدند")
            return True
        except Exception as e:
            print(f"❌ خطا در ذخیره مدل: {e}")
            return False

def load_or_train_model():
    """بارگذاری مدل یا آموزش مدل جدید اگر وجود نداشت"""
    try:
        if not os.path.exists("model.pkl") or not os.path.exists("scaler.pkl"):
            print("🔍 فایل مدل یافت نشد. در حال آموزش مدل جدید...")
            trainer = DiabetesModelTrainer()
            if trainer.train_model():
                if trainer.save_model():
                    return trainer.model, trainer.scaler
            return None, None
        
        # existing file
        print("📂 در حال بارگذاری مدل از فایل...")
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        print("✅ مدل و اسکیلر با موفقیت بارگذاری شدند")
        return model, scaler
        
    except Exception as e:
        print(f"❌ خطا در بارگذاری/آموزش مدل: {e}")
        return None, None


print("🚀 شروع راه‌اندازی سامانه CDSS...")
model, scaler = load_or_train_model()

if model is None or scaler is None:
    print("❌ خطای بحرانی: مدل نمی‌تواند بارگذاری یا آموزش داده شود")
    exit(1)

class DiabetesCDSS:
    def __init__(self):
        self.diabetic_ranges = {
            'Glucose': {'min': 126, 'max': None, 'unit': 'mg/dL'},
            'BMI': {'min': 25, 'max': None, 'unit': 'kg/m²'},
            'BloodPressure': {'min': 140, 'max': 90, 'unit': 'mmHg'},
            'Age': {'min': 45, 'max': None, 'unit': 'years'}
        }
    
    def rule_based_adjustment(self, features, prediction):
        """قوانین ساده برای بهبود پیش‌بینی"""
        pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age = features
        
       
        if glucose <= 100 and bp <= 70 and bmi < 25:
            return 0
        if bmi < 23.4 and dpf <= 0.647 and glucose <= 110:
            return 0
        if age < 30 and bmi < 25 and glucose <= 100:
            return 0
            
  
        if glucose >= 200 or bmi >= 35 or (age > 50 and glucose >= 150):
            return 1
            
        return prediction
    
    def check_abnormal_values(self, features, feature_names):
        """بررسی مقادیر غیرطبیعی"""
        abnormal = []
        
        ranges = {
            'Glucose': (70, 200),
            'BloodPressure': (60, 140),
            'BMI': (18, 40),
            'Age': (18, 100),
            'Pregnancies': (0, 15),
            'SkinThickness': (10, 50),
            'Insulin': (0, 300)
        }
        
        for i, name in enumerate(feature_names):
            if name in ranges:
                min_val, max_val = ranges[name]
                if features[i] < min_val or features[i] > max_val:
                    abnormal.append({
                        'feature': name,
                        'value': features[i],
                        'normal_range': f"{min_val}-{max_val}"
                    })
        
        return abnormal
    
    def generate_recommendations(self, prediction, features, sex, age):
        """تولید توصیه‌های شخصی‌شده"""
        if prediction == 1:
            return self._diabetic_recommendations(features, sex, age)
        else:
            return self._non_diabetic_recommendations(features, sex, age)
    
    def _diabetic_recommendations(self, features, sex, age):
        """توصیه‌های برای افراد دیابتی"""
        recommendations = [
            {
                "category": "قند خون",
                "advice": "قبل غذا 80-130 mg/dL، دو ساعت بعد غذا زیر 180 mg/dL",
                "priority": "high"
            },
            {
                "category": "پایش منظم",
                "advice": "اندازه‌گیری روزانه قند خون و ثبت نتایج",
                "priority": "high"
            },
            {
                "category": "دارو",
                "advice": "مصرف به موقع داروها طبق دستور پزشک",
                "priority": "high"
            },
            {
                "category": "فعالیت بدنی",
                "advice": "حداقل 150 دقیقه ورزش هوازی در هفته",
                "priority": "medium"
            },
            {
                "category": "تغذیه",
                "advice": "مصرف سبزیجات، کاهش کربوهیدرات و نمک",
                "priority": "medium"
            },
            {
                "category": "سبک زندگی",
                "advice": "پرهیز از سیگار و الکل، مدیریت استرس",
                "priority": "medium"
            }
        ]
        

        if age > 50:
            recommendations.append({
                "category": "پایش سلامت",
                "advice": "معاینه چشم و کلیه سالیانه",
                "priority": "medium"
            })
        
        if sex == "female":
            recommendations.append({
                "category": "بارداری",
                "advice": "مشاوره قبل از بارداری در صورت برنامه‌ریزی",
                "priority": "low"
            })
        
        return recommendations
    
    def _non_diabetic_recommendations(self, features, sex, age):
        """توصیه‌های برای افراد غیردیابتی"""
        glucose, bmi, age_val = features[1], features[5], features[7]
        
        recommendations = [
            {
                "category": "پیشگیری",
                "advice": "تغذیه سالم و فعالیت بدنی منظم",
                "priority": "medium"
            }
        ]
        

        if glucose > 100 or bmi > 25 or age_val > 45:
            next_screening = "6 ماه آینده"
            recommendations.append({
                "category": "غربالگری",
                "advice": "تکرار تست در 6 ماه آینده به دلیل عوامل خطر",
                "priority": "medium"
            })
        else:
            next_screening = "1 سال آینده"
            recommendations.append({
                "category": "غربالگری",
                "advice": "تکرار تست سالانه برای پایش سلامت",
                "priority": "low"
            })
        
        return recommendations, next_screening

cdss = DiabetesCDSS()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print(f"📨 دریافت درخواست پیش‌بینی: {data}")

        required_fields = ["Glucose", "BloodPressure", "BMI", "Age", "Sex"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"فیلد {field} الزامی است"}), 400

        pregnancies = float(data["Pregnancies"]) if data["Sex"] == "female" else 0.0
        
        features = [
            pregnancies,
            float(data["Glucose"]),
            float(data["BloodPressure"]),
            float(data.get("SkinThickness", 29)),  
            float(data.get("Insulin", 80)),        
            float(data["BMI"]),
            float(data.get("DiabetesPedigreeFunction", 0.5)),
            float(data["Age"])
        ]
        
        feature_names = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]
        
        print(f"🔢 ویژگی‌های پردازش شده: {features}")
        
        # random forest
        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)[0]
        print(f"🤖 پیش‌بینی اولیه مدل: {prediction}")
        
        # simple rules
        final_prediction = cdss.rule_based_adjustment(features, prediction)
        print(f"🎯 پیش‌بینی نهایی پس از قوانین: {final_prediction}")
        
        # irrational values
        abnormal_values = cdss.check_abnormal_values(features, feature_names)
        print(f"⚠️  مقادیر غیرطبیعی: {abnormal_values}")
        
        
        recommendations_data = cdss.generate_recommendations(
            final_prediction, features, data["Sex"], float(data["Age"])
        )
        
        try:
            ai_analysis = deepseek_advisor.generate_personalized_advice(
                final_prediction, features, data["Sex"], float(data["Age"])
            )
        except Exception as e:
            print(f"⚠️  خطا در DeepSeek: {e}")
            ai_analysis = None
        

        response = {
            "prediction": int(final_prediction),
            "risk_level": "بالا" if final_prediction == 1 else "پایین",
            "confidence": "high",
            "abnormal_values": abnormal_values,
            "timestamp": datetime.now().isoformat()
        }
        

        if final_prediction == 1:
            response["recommendations"] = recommendations_data
            response["followup"] = "ارجاع فوری به پزشک و انجام آزمایش HbA1c ظرف 1 ماه"
            response["alert"] = "نیاز به مداخله پزشکی فوری"
        else:
            response["recommendations"], response["next_screening"] = recommendations_data
            response["alert"] = "مقادیر در محدوده طبیعی" if not abnormal_values else "برخی مقادیر نیاز به توجه دارند"
        
        if ai_analysis:
            response["ai_analysis"] = ai_analysis
            response["model_used"] = "Random Forest + DeepSeek AI"
        else:
            response["model_used"] = "Random Forest"
        
        print(f"📤 ارسال پاسخ: {response}")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ خطا در پردازش درخواست: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy", 
        "service": "Diabetes CDSS",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "خوش آمدید به سامانه CDSS غربالگری دیابت نوع ۲",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "پیش‌بینی دیابت و دریافت توصیه‌ها",
            "GET /health": "بررسی وضعیت سامانه"
        }
    })

if __name__ == "__main__":
    print("🌟 سامانه CDSS آماده ارائه خدمات است")
    print("📍 در حال اجرا روی http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)