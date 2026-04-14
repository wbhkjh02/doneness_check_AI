from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import os

# ==========================================
# 1. 플라스크 웹 서버 초기화
# ==========================================
app = Flask(__name__)
CORS(app)  # 어디서든 이 서버에 접근할 수 있게 허용

# ==========================================
# 2. AI 모델 불러오기
#    - 로컬에서 실행할 때: models/jiho_best.pt 파일이 있으면 됩니다
#    - 서버(Render 등)에 배포할 때: 환경변수 MODEL_PATH로 경로 지정 가능
# ==========================================
MODEL_PATH = os.environ.get("MODEL_PATH", "models/jiho_best.pt")

try:
    model = YOLO(MODEL_PATH)
    print("✨ AI 모델 장착 완료!")
    print(f"   모델 경로: {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ 모델을 불러오는 데 실패했습니다: {e}")
    print(f"   시도한 경로: {MODEL_PATH}")
    model = None

# ==========================================
# 3. 서버가 살아있는지 확인하는 주소 (선택사항)
#    브라우저에서 '서버주소/' 를 열면 확인 가능합니다
# ==========================================
@app.route('/', methods=['GET'])
def home():
    status = "✅ 모델 로드됨" if model else "❌ 모델 로드 실패"
    return jsonify({
        "app": "Meat Master AI",
        "status": status,
        "message": "서버가 정상 작동 중입니다!"
    })

# ==========================================
# 4. 고기 판별 API
#    프론트엔드(HTML)에서 사진을 보내면 결과를 돌려줍니다
# ==========================================
@app.route('/predict', methods=['POST'])
def predict():
    # 모델이 제대로 안 불러와졌을 때
    if model is None:
        return jsonify({"error": "AI 모델이 로드되지 않았습니다. 서버 로그를 확인하세요."}), 500

    # 사진 파일이 제대로 전송됐는지 확인
    if 'file' not in request.files:
        return jsonify({"error": "사진 파일이 전송되지 않았습니다."}), 400

    file = request.files['file']

    try:
        # 사진을 읽어서 AI가 볼 수 있는 형태로 변환
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))

        # AI 판독 시작!
        results = model(img)
        result = results[0]

        names_dict = model.names
        top_class = names_dict[result.probs.top1]
        confidence = float(result.probs.top1conf) * 100
        check_class = str(top_class).strip().lower()

        # ---- 결과 멘트 ----
        if check_class == "raw":
            message = "🧊 생고기입니다. 얌전히 기다리세요."
            tip = "아직 익지 않았어요. 조금 더 기다려 주세요!"
        elif check_class == "flip_now":
            message = "🔥 지금 당장 뒤집으세요! 골든 타임입니다!"
            tip = "한 면이 잘 익었어요. 지금이 뒤집을 타이밍!"
        elif check_class == "done":
            message = "😋 완벽하게 익었습니다! 젓가락을 드세요!"
            tip = "최적의 굽기입니다. 바로 드세요!"
        else:
            message = f"현재 상태: {top_class}"
            tip = "계속 판독 중..."

        # HTML에 결과를 JSON 형태로 돌려줌
        return jsonify({
            "status": "success",
            "class": check_class,
            "confidence": round(confidence, 1),
            "message": message,
            "tip": tip
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================
# 5. 서버 실행
#    - PORT 환경변수: Render 같은 서비스가 자동으로 포트를 정해줍니다
#    - 없으면 기본값 5001로 실행됩니다 (로컬 테스트용)
# ==========================================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    print(f"🚀 고기 판별 백엔드 서버 시작! 포트: {port}")
    app.run(host='0.0.0.0', port=port, debug=False)