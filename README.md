git clone https://github.com/Cunmo-dev/F5TTS.git
# ĐẾN THƯ MỤC
cd F5TTS
# Tạo môi trường ảo
python -m venv env

# Kích hoạt môi trường ảo trên PowerShell
.\env\Scripts\Activate.ps1

# 3. Cài đặt dependencies
pip install -r requirements.txt

# 4. Chạy ứng dụng
python app.py
