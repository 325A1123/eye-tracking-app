import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import time
import matplotlib.pyplot as plt
import threading

# ページ設定
st.set_page_config(page_title="眼球運動（追従運動）チェックアプリ", layout="centered")
st.title("👁️ 眼球運動（追従運動）テスト")
st.write("カメラを使って、動く的（赤い丸）を目だけで追えているかを計測・分析します。")

# MediaPipeモデルの読み込み（キャッシュ化して高速化）
@st.cache_resource
def load_detector():
    base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)

# 映像処理クラスの定義
class EyeTrackingProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = load_detector()
        self.lock = threading.Lock()
        self.phase = "waiting"  # waiting -> calibrating -> testing -> done
        self.start_time = None
        self.calibration_duration = 3.0
        self.test_duration = 15.0
        self.calibration_logs = []
        self.data_logs = []

    def reset(self):
        with self.lock:
            self.phase = "waiting"
            self.start_time = None
            self.calibration_logs = []
            self.data_logs = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        # 鏡のように左右反転
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        # MediaPipeで顔・虹彩を検出
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.detector.detect(mp_image)

        normalized_eye_x = None
        face_in_guide = False
        nose_x = None

        if results.face_landmarks and len(results.face_landmarks) > 0:
            lm = results.face_landmarks[0]
            nose_x = lm[1].x * w
            nose_y = lm[1].y * h

            face_left_x = lm[234].x * w
            face_right_x = lm[454].x * w
            face_width = abs(face_right_x - face_left_x)

            # ガイド枠内に顔があるか判定（640x480基準の比率で計算）
            if (w * 0.4 <= nose_x <= w * 0.6) and (h * 0.33 <= nose_y <= h * 0.63) and (face_width > w * 0.12):
                face_in_guide = True

            # 左右の虹彩（468-472, 473-477）のX座標平均
            left_iris_x = np.mean([lm[i].x * w for i in range(468, 473)])
            right_iris_x = np.mean([lm[i].x * w for i in range(473, 478)])
            avg_eye_x = (left_iris_x + right_iris_x) / 2.0

            if face_width > 0:
                normalized_eye_x = (avg_eye_x - nose_x) / face_width

        with self.lock:
            target_x, target_y = int(w / 2), int(h / 2)

            if self.phase == "waiting":
                # ガイド枠の描画
                color = (255, 255, 0) if face_in_guide else (0, 255, 0)
                cv2.rectangle(img, (int(w*0.34), int(h*0.2)), (int(w*0.66), int(h*0.75)), color, 3)
                msg = "OK! Starting..." if face_in_guide else "Align face in box"
                cv2.putText(img, msg, (int(w*0.32), 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                if face_in_guide and normalized_eye_x is not None:
                    self.phase = "calibrating"
                    self.start_time = time.time()

            elif self.phase == "calibrating":
                elapsed = time.time() - self.start_time
                if normalized_eye_x is not None:
                    self.calibration_logs.append(normalized_eye_x)

                # 黄色い丸を描画
                cv2.circle(img, (target_x, target_y), 15, (0, 255, 255), -1)
                cv2.circle(img, (target_x, target_y), 15, (255, 255, 255), 2)
                cv2.putText(img, "Look at YELLOW dot (3s)", (int(w*0.25), 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                if elapsed > self.calibration_duration:
                    if len(self.calibration_logs) > 5:
                        self.phase = "testing"
                        self.start_time = time.time()
                    else:
                        self.phase = "waiting"
                        self.calibration_logs = []

            elif self.phase == "testing":
                elapsed = time.time() - self.start_time
                if elapsed > self.test_duration:
                    self.phase = "done"
                else:
                    # 的の座標を計算（左右に往復）
                    amplitude = w * 0.31
                    target_x = int((w / 2) + amplitude * np.sin(2 * np.pi * 0.05 * elapsed))
                    
                    # 赤い丸を描画
                    cv2.circle(img, (target_x, target_y), 15, (0, 0, 255), -1)
                    cv2.circle(img, (target_x, target_y), 15, (255, 255, 255), 2)
                    cv2.putText(img, f"Follow RED dot: {int(self.test_duration - elapsed)}s", (int(w*0.25), 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    self.data_logs.append({
                        'Time': elapsed,
                        'Target_X': target_x,
                        'Norm_Eye': normalized_eye_x if normalized_eye_x is not None else np.nan,
                        'Head_X': nose_x if nose_x is not None else np.nan,
                        'Width': w
                    })

            elif self.phase == "done":
                cv2.putText(img, "Test Complete! Click 'Show Results' below.", (int(w*0.12), int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 説明書き
with st.expander("📋 テストの手順（クリックして開く）", expanded=True):
    st.markdown("""
    1. 下の **「START」** ボタンを押してカメラを起動します。
    2. 画面の **緑色の枠** に顔を合わせると自動でテストが始まります。
    3. 最初に **3秒間、真ん中の黄色い丸** をじっと見つめてください。
    4. 続いて **15秒間、左右に動く赤い丸** を **顔は動かさず目だけで** 追ってください。
    5. 完了したら、下の **「計測結果を分析・表示する」** ボタンを押してください。
    """)

# WebRTCカメラコンポーネントの起動
ctx = webrtc_streamer(
    key="eye-tracking",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=EyeTrackingProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    rtc_configuration={"iceServers": [{"urls": ["stun:://google.com"]}]}
)

# 操作ボタンと結果表示
col1, col2 = st.columns(2)
with col1:
    analyze_btn = st.button("📊 計測結果を分析・表示する", type="primary")
with col2:
    if st.button("🔄 テストをリセット"):
        if ctx.video_processor:
            ctx.video_processor.reset()
            st.info("リセットしました。もう一度枠に顔を合わせてください。")

if analyze_btn:
    if ctx.video_processor:
        with ctx.video_processor.lock:
            data_logs = list(ctx.video_processor.data_logs)
            calibration_logs = list(ctx.video_processor.calibration_logs)
            phase = ctx.video_processor.phase

        if len(data_logs) > 10 and len(calibration_logs) > 0:
            df = pd.DataFrame(data_logs)
            screen_w = df['Width'].iloc[0]
            center_x = screen_w / 2.0

            df['Norm_Eye'] = df['Norm_Eye'].interpolate(method='linear').bfill().ffill()
            df['Head_X'] = df['Head_X'].interpolate(method='linear').bfill().ffill()

            center_reference = np.mean(calibration_logs)
            df['Eye_Diff'] = df['Norm_Eye'] - center_reference
            
            # 左右反転しているため符号を調整
            df['Eye_X_Scaled_Raw'] = center_x + (df['Eye_Diff'] * (screen_w * 5.5))

            # タイムラグ自動補正
            best_shift = 0
            best_corr = -1
            max_shift = min(30, len(df) // 3)
            for s in range(0, max_shift):
                shifted_eye = df['Eye_X_Scaled_Raw'].shift(-s).ffill().bfill()
                if shifted_eye.std() > 0 and df['Target_X'].std() > 0:
                    corr = df['Target_X'].corr(shifted_eye)
                    if corr > best_corr:
                        best_corr = corr
                        best_shift = s

            df['Eye_X_Scaled'] = df['Eye_X_Scaled_Raw'].shift(-best_shift).ffill().bfill()

            head_movement = df['Head_X'].std()
            avg_frame_rate = len(df) / 15.0
            lag_seconds = best_shift / avg_frame_rate if avg_frame_rate > 0 else 0
            sync_score = max(0, best_corr * 100)

            st.subheader("📈 分析結果")
            m1, m2, m3 = st.columns(3)
            m1.metric("的とのシンクロ率", f"{sync_score:.1f} %")
            m2.metric("顔の横ブレ量", f"{head_movement:.1f} px", delta="10以下が理想", delta_color="off")
            m3.metric("処理速度", f"{avg_frame_rate:.1f} fps")

            if head_movement > 15:
                st.warning("⚠️ テスト中に顔（頭）が左右に動いています。顎を固定して目だけで追うとさらに正確に測れます。")

            if sync_score > 65:
                st.success("✅ **診断示唆**: 的の動きに合わせて綺麗に目が動いています（正常な追従運動）。")
            else:
                st.info("ℹ️ **診断示唆**: 連動性がやや低いです。顔を動かさず、目だけで追うように意識して再試行してください。")

            # グラフの描画
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df['Time'], df['Target_X'], label='Target (Red Ball)', color='red', linewidth=2)
            ax.plot(df['Time'], df['Eye_X_Scaled'], label='Your Eye Position', color='green', alpha=0.8, linestyle='--')
            ax.set_title('Eye Tracking Result (Normalized & Both Eyes)')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('X Position on Screen')
            ax.set_ylim(0, screen_w)
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # CSVダウンロードボタン
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 計測データをCSVでダウンロード",
                data=csv,
                file_name='eye_tracking_result.csv',
                mime='text/csv',
            )
        else:
            st.error("有効な計測データがまだありません。上のカメラを起動し、テストを完了させてからボタンを押してください。")
