"""Phần dùng chung cho cả nhóm: cắt fold, bản kê trọng số, đo chồng lấn,
gộp kết quả thành bảng.

KHÔNG chứa model nào. Bốn model của bảng benchmark nằm trong
benchmark/<model>/cofseg/, mỗi thư mục một bản độc lập do một người phụ trách.
Thêm model mới thì thêm một thư mục trong benchmark/, không thêm vào đây.

Tách hẳn khỏi app/ (công cụ gán nhãn): thư viện KHÔNG BAO GIỜ import app/.
Chiều ngược lại thì được phép, nên xoá bên nào bên kia vẫn sống.
"""

__version__ = "0.1.0"
