"""Gộp kết quả của cả bốn model thành một bảng.

Việc CHẤM từng model nằm trong thư mục của người phụ trách
(benchmark/<model>/cofseg/evaluation/), mỗi người một bản. Ở đây chỉ còn phần
đọc các lần chấm đã xong và dựng bảng model x ruộng — thứ không thuộc về ai
trong bốn người.
"""

from . import folds

__all__ = ["folds"]
