import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

MODEL_NAME = "FiveC/tay-to-viet-v2"

def load_model():
try:
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
return tokenizer, model
except Exception as e:
# Nếu có lỗi, trả về None để xử lý ở giao diện
st.error(f"Lỗi khi tải model: {e}. Vui lòng kiểm tra lại tên model trong file app.py.")
return None, None
Tải model và tokenizer
tokenizer, model = load_model()
--- Xây dựng giao diện người dùng (UI) ---
st.set_page_config(page_title="Dịch máy Tày - Việt", page_icon="🤖")
st.title("🤖📝 Demo Dịch máy Tày - Việt")
st.write("Sản phẩm được xây dựng bằng Streamlit và triển khai trên Streamlit Community Cloud. Model được tải từ Hugging Face.")
Hộp văn bản để người dùng nhập liệu
input_text = st.text_area("Nhập câu tiếng Tày vào đây:", height=150, placeholder="Ví dụ: Pây kin khẩu.")
Nút để thực hiện dịch
if st.button("Dịch sang Tiếng Việt"):
# Kiểm tra xem model đã được tải thành công và có input chưa
if model is not None and tokenizer is not None and input_text:
# Hiển thị thông báo đang xử lý
with st.spinner("Đang dịch, vui lòng chờ..."):
# 1. Tokenize câu đầu vào
inputs = tokenizer(input_text, return_tensors="tf", max_length=512, truncation=True)
# 2. Dùng model để sinh ra câu dịch
        output_sequences = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=128,
            num_beams=5,
            early_stopping=True
        )
        
        # 3. Decode kết quả
        translated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        
        # 4. Hiển thị kết quả
        st.subheader("Kết quả dịch (Tiếng Việt):")
        st.success(translated_text)
        
elif not input_text:
    st.warning("Vui lòng nhập câu cần dịch.")
else:
    # Trường hợp model không tải được
    st.error("Model chưa sẵn sàng. Vui lòng kiểm tra lỗi ở trên.")
