import os
import urllib.request
import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import matplotlib.pyplot as plt
import base64
import tempfile
import json

st.set_page_config(page_title="眼球運動チェックアプリ", layout="centered")
st.title("👁️ 眼球運動（追従運動）テスト")

# MediaPipeモデルの読み込み（CPU明示＋自動フォールバック付き）
@st.cache_resource
def load_detector():
    model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
    model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1_000_000:
        model_path = os.path.join(tempfile.gettempdir(), "face_landmarker.task")
        if not os.path.exists(model_path) or os.path.getsize(model_path) < 1_000_000:
            req = urllib.request.Request(model_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp, open(model_path, "wb") as f:
                f.write(resp.read())

    with open(model_path, "rb") as f:
        model_bytes = f.read()

    base_options = python.BaseOptions(
        model_asset_buffer=model_bytes,
        delegate=python.BaseOptions.Delegate.CPU
    )
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)

detector = load_detector()

def analyze_frame(frame_bgr, target_x, elapsed_time):
    h, w, _ = frame_bgr.shape
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)

    normalized_eye_x = np.nan
    nose_x = np.nan

    if results.face_landmarks and len(results.face_landmarks) > 0:
        lm = results.face_landmarks[0]
        nose_x = lm[1].x * w
        face_width = abs(lm[454].x * w - lm[234].x * w)
        left_iris_x = np.mean([lm[i].x * w for i in range(468, 473)])
        right_iris_x = np.mean([lm[i].x * w for i in range(473, 478)])
        if face_width > 0:
            normalized_eye_x = (((left_iris_x + right_iris_x) / 2.0) - nose_x) / face_width

    return {'Time': elapsed_time, 'Target_X': target_x, 'Norm_Eye': normalized_eye_x, 'Head_X': nose_x, 'Width': w}

def show_analysis(df, calib_mean):
    screen_w = df['Width'].iloc[0]
    center_x = screen_w / 2.0
    df['Norm_Eye'] = df['Norm_Eye'].interpolate(method='linear').bfill().ffill()
    df['Head_X'] = df['Head_X'].interpolate(method='linear').bfill().ffill()

    if np.isnan(calib_mean):
        calib_mean = df['Norm_Eye'].iloc[:5].mean()

    df['Eye_X_Scaled_Raw'] = center_x + ((df['Norm_Eye'] - calib_mean) * (screen_w * 5.5))

    best_shift, best_corr = 0, -1
    for s in range(0, min(20, len(df) // 3)):
        shifted = df['Eye_X_Scaled_Raw'].shift(-s).ffill().bfill()
        if shifted.std() > 0 and df['Target_X'].std() > 0:
            corr = df['Target_X'].corr(shifted)
            if corr > best_corr:
                best_corr, best_shift = corr, s

    df['Eye_X_Scaled'] = df['Eye_X_Scaled_Raw'].shift(-best_shift).ffill().bfill()
    head_movement = df['Head_X'].std()
    sync_score = max(0, best_corr * 100)

    st.subheader("📈 分析結果")
    c1, c2, c3 = st.columns(3)
    c1.metric("的とのシンクロ率", f"{sync_score:.1f} %")
    c2.metric("顔の横ブレ量", f"{head_movement:.1f} px")
    c3.metric("解析フレーム数", f"{len(df)} 枚")

    if head_movement > 15:
        st.warning("⚠️ 顔が左右に動いています。顎を固定して目だけで追うと正確に測れます。")
    if sync_score > 65:
        st.success("✅ **診断示唆**: 的の動きに合わせて綺麗に目が動いています（正常な追従運動）。")
    else:
        st.info("ℹ️ **診断示唆**: 連動性がやや低いです。顔を動かさず目だけで追うよう意識してください。")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['Time'], df['Target_X'], label='Target (Red Ball)', color='red', linewidth=2)
    ax.plot(df['Time'], df['Eye_X_Scaled'], label='Your Eye Position', color='green', linestyle='--')
    ax.set_ylim(0, screen_w)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    csv = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 計測データをCSVでダウンロード", data=csv, file_name='eye_result.csv', mime='text/csv')

tab1, tab2 = st.tabs(["🎥 リアルタイムWebカメラテスト", "📁 録画動画でテスト"])

with tab1:
    st.write("1. 「カメラを起動」を押して緑色の人型枠に顔と肩を合わせる → 2. 「テスト開始」で黄色い丸(3秒)→赤い丸(12秒)を目だけで追う → 3. 完了後にデータをコピーして下に貼り付け")
    html_code = """
    <div style="text-align:center; font-family:sans-serif;">
        <div style="margin-bottom:8px;">
            <button id="prepBtn" style="padding:10px 16px; font-size:14px; background:#2e7d32; color:white; border:none; border-radius:6px; cursor:pointer;">📷 1. カメラを起動して位置を合わせる</button>
            <button id="startBtn" disabled style="padding:10px 16px; font-size:14px; background:#ccc; color:white; border:none; border-radius:6px; cursor:not-allowed;">▶ 2. テスト開始</button>
        </div>
        <div id="statusText" style="font-weight:bold; margin:8px 0;">まずは「1. カメラを起動して位置を合わせる」を押してください</div>
        <div style="position:relative; width:480px; height:360px; margin:0 auto; background:#000; border-radius:8px; overflow:hidden;">
            <video id="video" width="480" height="360" autoplay playsinline style="transform:scaleX(-1);"></video>
            <canvas id="overlay" width="480" height="360" style="position:absolute; top:0; left:0;"></canvas>
        </div>
        <div id="resultBox" style="display:none; margin-top:10px;">
            <button id="copyBtn" style="padding:8px 16px; background:#0068c9; color:white; border:none; border-radius:4px; cursor:pointer;">📋 計測データをコピー</button>
            <textarea id="outputData" rows="2" style="width:90%; margin-top:5px;"></textarea>
        </div>
    </div>
    <script>
    const prepBtn = document.getElementById('prepBtn'), startBtn = document.getElementById('startBtn');
    const video = document.getElementById('video'), overlay = document.getElementById('overlay'), ctx = overlay.getContext('2d');
    const statusText = document.getElementById('statusText'), resultBox = document.getElementById('resultBox'), outputData = document.getElementById('outputData'), copyBtn = document.getElementById('copyBtn');
    const cap = document.createElement('canvas'); cap.width = 320; cap.height = 240; const capCtx = cap.getContext('2d');
    let stream = null, isTesting = false, previewAnimId = null;

    // 緑色の人型（顔・肩）ガイド枠を描画する関数
    function drawGuideFrame() {
        ctx.save();
        ctx.strokeStyle = 'rgba(0, 255, 100, 0.75)';
        ctx.lineWidth = 3;
        ctx.setLineDash([8, 6]);

        // 顔の輪郭（中央の楕円）
        ctx.beginPath();
        ctx.ellipse(240, 165, 78, 105, 0, 0, Math.PI * 2);
        ctx.stroke();

        // 肩のライン（下部のアーチ）
        ctx.beginPath();
        ctx.moveTo(85, 360);
        ctx.quadraticCurveTo(100, 290, 185, 280);
        ctx.moveTo(395, 360);
        ctx.quadraticCurveTo(380, 290, 295, 280);
        ctx.stroke();

        // テスト開始前の案内テキスト
        if (!isTesting) {
            ctx.setLineDash([]);
            ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
            ctx.fillRect(105, 18, 270, 28);
            ctx.fillStyle = '#00ff64';
            ctx.font = 'bold 14px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('緑の枠に顔と肩を合わせてください', 240, 37);
        }
        ctx.restore();
    }

    // 待機中のプレビューループ
    function previewLoop() {
        if (isTesting) return;
        ctx.clearRect(0, 0, 480, 360);
        drawGuideFrame();
        previewAnimId = requestAnimationFrame(previewLoop);
    }

    // 1. カメラ起動＆位置合わせ
    prepBtn.onclick = async () => {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: { width: 480, height: 360 }, audio: false });
            video.srcObject = stream;
            isTesting = false;
            prepBtn.disabled = true;
            prepBtn.style.background = '#ccc';
            startBtn.disabled = false;
            startBtn.style.background = '#ff4b4b';
            startBtn.style.cursor = 'pointer';
            statusText.innerText = "緑の枠に顔を合わせたら「2. テスト開始」を押してください";
            previewLoop();
        } catch (e) { statusText.innerText = "カメラエラー: " + e.message; }
    };

    // 2. テスト開始
    startBtn.onclick = () => {
        if (!stream) return;
        isTesting = true;
        cancelAnimationFrame(previewAnimId);
        startBtn.disabled = true;
        startBtn.style.background = '#ccc';
        resultBox.style.display = 'none';

        const records = [], startTime = performance.now(); let lastCap = 0;
        function testLoop(now) {
            const elapsed = (now - startTime) / 1000.0;
            ctx.clearRect(0, 0, 480, 360);
            drawGuideFrame();

            if (elapsed < 3.0) {
                statusText.innerText = `【基準測定】黄色い丸を見てください (${(3.0 - elapsed).toFixed(1)}秒)`;
                ctx.beginPath(); ctx.arc(240, 180, 14, 0, Math.PI * 2); ctx.fillStyle = 'yellow'; ctx.fill();
                if (now - lastCap > 150) {
                    lastCap = now; capCtx.save(); capCtx.scale(-1, 1); capCtx.drawImage(video, -320, 0, 320, 240); capCtx.restore();
                    records.push({ phase: 'calib', t: elapsed, tx: 160, img: cap.toDataURL('image/jpeg', 0.6) });
                }
                requestAnimationFrame(testLoop);
            } else if (elapsed < 15.0) {
                const t = elapsed - 3.0;
                statusText.innerText = `【本番】赤い丸を目だけで追ってください (${(15.0 - elapsed).toFixed(1)}秒)`;
                const tx480 = 240 + 160 * Math.sin(2 * Math.PI * 0.12 * t);
                const tx320 = 160 + 106 * Math.sin(2 * Math.PI * 0.12 * t);
                ctx.beginPath(); ctx.arc(tx480, 180, 14, 0, Math.PI * 2); ctx.fillStyle = 'red'; ctx.fill();
                if (now - lastCap > 120) {
                    lastCap = now; capCtx.save(); capCtx.scale(-1, 1); capCtx.drawImage(video, -320, 0, 320, 240); capCtx.restore();
                    records.push({ phase: 'test', t: t, tx: tx320, img: cap.toDataURL('image/jpeg', 0.6) });
                }
                requestAnimationFrame(testLoop);
            } else {
                stream.getTracks().forEach(tr => tr.stop());
                isTesting = false;
                statusText.innerText = "完了！下の「計測データをコピー」を押してください。";
                outputData.value = JSON.stringify(records);
                resultBox.style.display = 'block';
                prepBtn.disabled = false;
                prepBtn.style.background = '#2e7d32';
            }
        }
        requestAnimationFrame(testLoop);
    };
    copyBtn.onclick = () => { outputData.select(); document.execCommand('copy'); copyBtn.innerText = "✅ コピー完了！"; };
    </script>
    """
    components.html(html_code, height=520)
    raw_json = st.text_area("📋 コピーした計測データをここに貼り付けてください", height=80)
    if st.button("📊 AIで黒目の動きを解析する", type="primary") and raw_json.strip():
        try:
            records = json.loads(raw_json)
            calib_vals, test_logs = [], []
            with st.spinner("MediaPipeで解析中..."):
                for r in records:
                    img_bytes = base64.b64decode(r['img'].split(",", 1)[1])
                    frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                    res = analyze_frame(frame, r['tx'], r['t'])
                    if r['phase'] == 'calib' and not np.isnan(res['Norm_Eye']):
                        calib_vals.append(res['Norm_Eye'])
                    elif r['phase'] == 'test':
                        test_logs.append(res)
            if len(test_logs) > 5:
                show_analysis(pd.DataFrame(test_logs), np.mean(calib_vals) if calib_vals else np.nan)
            else:
                st.error("解析データ不足です。もう一度お試しください。")
        except Exception as e:
            st.error(f"読み込みエラー: {e}")

with tab2:
    uploaded = st.file_uploader("動画ファイル（MP4/MOV）をアップロード", type=["mp4", "mov", "avi"])
    if uploaded is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded.read())
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        logs, idx = [], 0
        with st.spinner("動画を解析中..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % 3 == 0:
                    t = idx / fps
                    w = frame.shape[1]
                    logs.append(analyze_frame(frame, (w / 2.0) + (w * 0.31) * np.sin(2 * np.pi * 0.1 * t), t))
                idx += 1
        cap.release()
        if len(logs) > 5:
            show_analysis(pd.DataFrame(logs), np.nan)
        else:
            st.error("顔を検出できませんでした。")
