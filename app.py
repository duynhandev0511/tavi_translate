import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
import time

# --- CẤU HÌNH VÀ TẢI MODEL (Không thay đổi) ---

MODEL_NAME = "FiveC/tay-to-viet-v2"

@st.cache_resource
def load_model():
    """Tải tokenizer và model từ Hugging Face."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        return tokenizer, model
    except Exception as e:
        st.error(f"Lỗi khi tải model: {e}. Vui lòng kiểm tra lại tên model '{MODEL_NAME}' và kết nối mạng.")
        return None, None

tokenizer, model = load_model()

# --- KHỞI TẠO SESSION STATE ---
if 'translated_text' not in st.session_state:
    st.session_state.translated_text = ""
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

# --- CSS ĐỂ THAY ĐỔI HOÀN TOÀN GIAO DIỆN ---
# Đây là phần quan trọng nhất để có được giao diện giống app di động
def load_css():
    st.markdown("""
        <style>
            /* Màu nền tổng thể */
            body {
                background-color: #F0F2F6; /* Màu nền xám nhạt như trong ảnh */
            }

            /* Ẩn header và footer mặc định của Streamlit */
            #MainMenu, footer {
                visibility: hidden;
            }
            
            /* Tùy chỉnh khối container chính để giống màn hình điện thoại */
            .block-container {
                max-width: 420px; /* Chiều rộng của một màn hình điện thoại tiêu chuẩn */
                padding-top: 2rem;
                padding-bottom: 2rem;
            }

            /* Tiêu đề "Translate" */
            h1 {
                text-align: center;
                font-weight: bold;
                color: #1a1a1a;
            }

            /* Hộp chứa cho ô nhập liệu và kết quả */
            .translate-box {
                background-color: white;
                border-radius: 1rem; /* Bo góc tròn */
                padding: 1.2rem;
                margin-bottom: 1rem;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05); /* Thêm bóng đổ nhẹ */
            }

            /* Tùy chỉnh ô text_area */
            .stTextArea textarea {
                border: none;
                background-color: white;
                min-height: 150px;
                color: #1a1a1a;
            }
            .stTextArea textarea:focus {
                outline: none !important;
                box-shadow: none !important;
                border: none !important;
            }

            /* Phần hiển thị kết quả dịch */
            .result-text {
                min-height: 150px;
                color: #1a1a1a;
                font-size: 1rem;
                line-height: 1.6;
            }
            
            /* Các icon và đếm ký tự */
            .bottom-bar {
                color: #888;
                font-size: 0.8rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-top: 0.5rem;
            }
            .bottom-bar .icons span {
                margin-left: 1rem;
                cursor: pointer;
            }

            /* Nút dịch */
            .stButton>button {
                border-radius: 0.75rem;
                background-color: #007AFF; /* Màu xanh dương hiện đại */
                color: white;
                font-weight: bold;
                height: 3rem;
                width: 100%; /* Chiếm toàn bộ chiều rộng */
            }
            .stButton>button:hover {
                background-color: #0056b3;
                color: white;
            }

        </style>
    """, unsafe_allow_html=True)

# --- XÂY DỰNG GIAO DIỆN NGƯỜI DÙNG (UI) ---

st.set_page_config(layout="centered", page_title="Dịch Tày-Việt")

load_css()

# Tiêu đề ứng dụng
st.title("Translate")

# Vùng chọn ngôn ngữ (mô phỏng)
lang_col1, lang_col2, lang_col3 = st.columns([0.4, 0.2, 0.4])
with lang_col1:
    st.info("🇹 Tày (Tay)")
with lang_col2:
    st.markdown("<p style='text-align: center; font-size: 24px; margin-top: 5px;'>🔄</p>", unsafe_allow_html=True)
with lang_col3:
    st.success("🇻🇳 Vietnamese")

st.write("") # Thêm một khoảng trống nhỏ

# --- Hộp nhập liệu ---
with st.container():
    st.markdown('<div class="translate-box">', unsafe_allow_html=True)
    input_text = st.text_area(
        "Nhập văn bản cần dịch", 
        key="input_text", 
        label_visibility="collapsed"
    )
    
    char_count = len(input_text)
    st.markdown(f"""
        <div class="bottom-bar">
            <span>{char_count} / 5.000</span>
            <div class="icons">
                <span>🔊</span>
                <span>🎤</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- Hộp kết quả ---
with st.container():
    st.markdown('<div class="translate-box">', unsafe_allow_html=True)
    
    # Hiển thị spinner hoặc kết quả
    if 'processing' in st.session_state and st.session_state.processing:
        # Tạo hiệu ứng loading giả để UI mượt hơn
        with st.spinner("Đang dịch..."):
            time.sleep(1) # Chờ 1 giây để spinner hiển thị rõ
    elif st.session_state.translated_text:
         st.markdown(f'<div class="result-text">{st.session_state.translated_text}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-text" style="color: #aaa;">Bản dịch sẽ xuất hiện ở đây...</div>', unsafe_allow_html=True)

    char_count_result = len(st.session_state.translated_text)
    st.markdown(f"""
        <div class="bottom-bar">
            <span>{char_count_result} / 5.000</span>
            <div class="icons">
                <span>🔊</span>
                <span>📋</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# --- Nút Dịch và Logic ---
if st.button("Dịch", use_container_width=True):
    if model and tokenizer and input_text:
        # Đặt cờ đang xử lý để hiển thị spinner
        st.session_state.processing = True
        st.rerun() # Chạy lại để hiển thị spinner
        
    elif not input_text:
        st.toast("🤔 Vui lòng nhập văn bản để dịch!")
    else:
        st.error("Model chưa sẵn sàng. Vui lòng làm mới trang.")

# Logic dịch thực sự được chạy sau khi rerun
if 'processing' in st.session_state and st.session_state.processing:
    # 1. Tokenize
    inputs = tokenizer(st.session_state.input_text, return_tensors="tf", max_length=512, truncation=True)
    
    # 2. Generate
    output_sequences = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=512,
        num_beams=5,
        early_stopping=True
    )
    
    # 3. Decode
    result = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    
    # 4. Lưu kết quả và tắt cờ xử lý
    st.session_state.translated_text = result
    st.session_state.processing = False
    
    # Chạy lại lần cuối để hiển thị kết quả
    st.rerun()
