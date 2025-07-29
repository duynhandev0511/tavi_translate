import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

# --- CẤU HÌNH VÀ TẢI MODEL ---

# ==============================================================================
# THAY THẾ 'FiveC/tay-to-viet-v2' BẰNG TÊN MODEL CỦA BẠN NẾU CẦN
# ==============================================================================
MODEL_NAME = "FiveC/tay-to-viet-v2"

# st.cache_resource giúp lưu trữ model vào cache, chỉ tải 1 lần duy nhất.
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

# Tải model và tokenizer
tokenizer, model = load_model()


# --- KHỞI TẠO SESSION STATE ---
# st.session_state giúp lưu trữ giá trị giữa các lần chạy lại của app
# (ví dụ: khi người dùng nhấn nút).

# Lưu trữ văn bản dịch gần nhất
if 'translated_text' not in st.session_state:
    st.session_state.translated_text = ""
# Lưu trữ lịch sử các bản dịch
if 'history' not in st.session_state:
    st.session_state.history = []


# --- XÂY DỰNG GIAO DIỆN NGƯỜI DÙNG (UI) THEO MẪU GOOGLE TRANSLATE ---

st.set_page_config(layout="wide", page_title="Dịch máy Tày - Việt", page_icon="🤖")

# --- Tiêu đề ---
st.markdown(
    """
    <style>
        /* Tùy chỉnh CSS để giảm khoảng trống trên cùng */
        .block-container {
            padding-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("🇻🇳 Dịch máy Tày - Việt 🇹")
st.caption("Mô phỏng giao diện Google Translate với model Tày-Việt từ Hugging Face.")


# --- Khu vực dịch chính ---
col1, col2 = st.columns(2, gap="medium")

# Cột Nhập liệu (Bên trái)
with col1:
    st.subheader("Tiếng Tày")
    input_text = st.text_area(
        "Nhập văn bản cần dịch...",
        height=250,
        key="input_text_area",
        label_visibility="collapsed"
    )
    
    char_count = len(input_text)
    st.caption(f"{char_count} / 5000 ký tự")

    # Nút dịch
    translate_button = st.button("Dịch sang Tiếng Việt", type="primary", use_container_width=True)


# Cột Kết quả (Bên phải)
with col2:
    st.subheader("Tiếng Việt")
    # Sử dụng container với viền để tạo box kết quả
    output_container = st.container(height=320, border=True)
    with output_container:
        # Hiển thị kết quả được lưu trong session_state
        st.markdown(st.session_state.translated_text)


# --- XỬ LÝ LOGIC DỊCH ---
if translate_button:
    # Chỉ thực hiện dịch nếu model đã tải và có văn bản nhập vào
    if model and tokenizer and input_text:
        # Hiển thị spinner trong box kết quả để báo đang xử lý
        with output_container:
            with st.spinner("🧠 Đang dịch..."):
                # 1. Tokenize câu đầu vào
                inputs = tokenizer(input_text, return_tensors="tf", max_length=512, truncation=True)
                
                # 2. Dùng model để sinh ra câu dịch
                output_sequences = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=512,  # Tăng max_length cho câu dài
                    num_beams=5,
                    early_stopping=True
                )
                
                # 3. Decode kết quả
                result = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
                
                # 4. Lưu kết quả vào session_state để hiển thị
                st.session_state.translated_text = result
                
                # 5. Thêm bản dịch mới vào đầu danh sách lịch sử
                st.session_state.history.insert(0, {
                    "source": input_text,
                    "translation": result
                })
                
                # Chạy lại script để cập nhật UI với kết quả mới
                st.rerun()
                
    elif not input_text:
        st.toast("🤔 Vui lòng nhập văn bản để dịch!")
    else:
        st.error("Model chưa sẵn sàng. Vui lòng kiểm tra lại lỗi ở trên và làm mới trang.")


# --- Khu vực Lịch sử dịch ---
st.divider()
st.header("Lịch sử")

if not st.session_state.history:
    st.info("Chưa có bản dịch nào trong lịch sử.")
else:
    # Hiển thị 5 bản dịch gần nhất
    for i, translation_item in enumerate(st.session_state.history[:5]):
        with st.container(border=True):
            st.text("Tày (Nguồn)")
            st.info(translation_item["source"])

            st.text("Việt (Dịch)")
            st.success(translation_item["translation"])
