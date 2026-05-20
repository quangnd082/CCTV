#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════╗
║          HỌC TIẾNG TRUNG TỪ CON SỐ 0 – GIA SƯ 1-1 RIÊNG TƯ            ║
║          Phiên bản: 1.0.0  |  Ngôn ngữ: Python 3 + PyQt5               ║
╚══════════════════════════════════════════════════════════════════════════╝

HƯỚNG DẪN CHẠY:
  1. Cài PyQt5:  pip install PyQt5
  2. Đặt file này và lessons_data.json vào cùng một thư mục
  3. Chạy: python chinese_tutor.py

Kiến trúc:
  - MainWindow: Cửa sổ chính chứa QTabWidget
  - Tab 1 (FoundationTab):   Giải thích Pinyin, Thanh điệu
  - Tab 2 (RoadmapTab):      Lộ trình 5 giai đoạn
  - Tab 3 (LessonTab):       Bài học chi tiết + Mini Quiz
  - LessonRenderer:          Chuyển dữ liệu JSON → HTML đẹp
  - DataManager:             Đọc/ghi file JSON, tạo mẫu nếu chưa có
"""

import sys
import os
import json
from typing import Optional

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QTextBrowser, QListWidget, QListWidgetItem,
    QPushButton, QScrollArea, QFrame, QButtonGroup, QRadioButton,
    QMessageBox, QSplitter, QSizePolicy, QGroupBox, QGridLayout,
    QSpacerItem
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QFontDatabase, QPalette, QColor, QPixmap

# ═══════════════════════════════════════════════════════════════════════════
#  HẰNG SỐ MÀU SẮC VÀ FONT CHỮ
# ═══════════════════════════════════════════════════════════════════════════

# Màu chủ đạo
C_BG_MAIN = "#F5F7FA"  # Nền trang chính – xám nhạt nhẹ mắt
C_BG_SIDEBAR = "#FFFFFF"  # Nền sidebar danh sách bài
C_BG_CONTENT = "#FFFFFF"  # Nền khu vực nội dung
C_PRIMARY = "#0078D7"  # Xanh công nghệ – nút, tiêu đề chính
C_PRIMARY_DARK = "#005A9E"  # Xanh đậm – hover
C_ACCENT = "#FF4D4F"  # Đỏ nhấn – cảnh báo, lỗi
C_SUCCESS = "#389E0D"  # Xanh lá – đúng, ví dụ
C_WARNING = "#FA8C16"  # Cam – lưu ý, mẹo nhớ
C_PURPLE = "#722ED1"  # Tím – phân tích ngữ âm
C_TEAL = "#13C2C2"  # Xanh lá nhạt – luyện đọc
C_TEXT_DARK = "#1A1A2E"  # Chữ tối
C_TEXT_MID = "#4A4A6A"  # Chữ trung tính
C_TEXT_LIGHT = "#8A8AAA"  # Chữ mờ
C_BORDER = "#E8EAF0"  # Đường viền nhạt
C_QUIZ_BG = "#F0F4FF"  # Nền khu Quiz
C_HANZI = "#C0392B"  # Màu hiển thị chữ Hán – đỏ truyền thống

# Tên file dữ liệu
DATA_FILE = "lessons_data_clau.json"

# ═══════════════════════════════════════════════════════════════════════════
#  DỮ LIỆU MẪU – Tự tạo khi chưa có file JSON
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_DATA = {
    "app_info": {
        "name": "Học Tiếng Trung Từ Con Số 0",
        "version": "1.0.0",
        "author": "Gia Sư Riêng 1-1",
        "description": "Dữ liệu bài học dành cho người Việt học tiếng Trung từ đầu"
    },
    "lessons": [
        {
            "id": "L01",
            "title": "Bài 1: Xin Chào! (你好)",
            "stage": "Giai đoạn 1 – Làm Quen Với Tiếng Trung",
            "vocabulary": [
                {
                    "hanzi": "你好",
                    "pinyin": "nǐ hǎo",
                    "tone": "Thanh 3 + Thanh 3 → Thanh 2 + Thanh 3 (biến điệu)",
                    "vietnamese_reading": "Nỉ hảo",
                    "meaning": "Xin chào / Chào bạn"
                },
                {
                    "hanzi": "你",
                    "pinyin": "nǐ",
                    "tone": "Thanh 3 ↘↗",
                    "vietnamese_reading": "Nỉ",
                    "meaning": "Bạn (ngôi thứ 2)"
                },
                {
                    "hanzi": "好",
                    "pinyin": "hǎo",
                    "tone": "Thanh 3 ↘↗",
                    "vietnamese_reading": "Hảo",
                    "meaning": "Tốt / Ổn / Được"
                },
                {
                    "hanzi": "我",
                    "pinyin": "wǒ",
                    "tone": "Thanh 3 ↘↗",
                    "vietnamese_reading": "Ủa (gần giống)",
                    "meaning": "Tôi (ngôi thứ 1)"
                },
                {
                    "hanzi": "谢谢",
                    "pinyin": "xiè xie",
                    "tone": "Thanh 4 + Thanh nhẹ",
                    "vietnamese_reading": "Xiê xiê (nhẹ chữ sau)",
                    "meaning": "Cảm ơn"
                },
                {
                    "hanzi": "再见",
                    "pinyin": "zài jiàn",
                    "tone": "Thanh 4 ↘ + Thanh 4 ↘",
                    "vietnamese_reading": "Zài chiên",
                    "meaning": "Tạm biệt"
                }
            ],
            "phonetic_analysis": (
                "【Phân Tích Âm Từng Chữ】\n\n"
                "🔤 NǏ (你 - Bạn):\n"
                "• Phụ âm đầu: N → Giống 'n' tiếng Việt, lưỡi chạm chân răng trên, hơi qua mũi\n"
                "• Nguyên âm: I → Môi KÉO NGANG tối đa, không tròn\n"
                "• Thanh 3: Xuống thấp rồi kéo lên (↘↗)\n\n"
                "🔤 HǍO (好 - Tốt):\n"
                "• Phụ âm H → KHÔNG giống 'h' Việt! Xuất phát từ CỔ HỌNG, nghe có tiếng xước\n"
                "• Nguyên âm AO: Mở miệng to nói 'A', từ từ khép môi thành 'O'\n"
                "• Thanh 3: Xuống rồi lên (↘↗)\n\n"
                "⚡ QUY TẮC BIẾN ĐIỆU QUAN TRỌNG:\n"
                "• Hai Thanh 3 đứng liền nhau → Chữ TRƯỚC đổi thành Thanh 2\n"
                "• 你好 → Thực đọc: NÍ HǍO (你 đổi Thanh 3→2)"
            ),
            "mouth_shape": (
                "【Hướng Dẫn Khẩu Hình Chi Tiết】\n\n"
                "👄 Âm N (trong 你):\n"
                "• Đầu lưỡi chạm chân răng cửa trên\n"
                "• Hơi thoát qua MŨI (thử bịt mũi → âm tắt = đúng!)\n"
                "• Môi hé mở tự nhiên\n\n"
                "👄 Âm H (trong 好):\n"
                "• Gốc lưỡi nâng lên phía vòm mềm\n"
                "• Hơi từ CỔ HỌNG, nghe có tiếng cọ xát\n"
                "• Thử hà hơi lên lòng bàn tay → Đó là cảm giác đúng!\n\n"
                "👄 Nguyên âm AO (trong 好):\n"
                "• Bắt đầu: Mở miệng rộng → 'A'\n"
                "• Kết thúc: Thu môi tròn lại → 'O'\n"
                "• Phải trượt dần dần, không đột ngột"
            ),
            "vietnamese_errors": (
                "【⚠️ Người Việt Hay Sai – Giáo Viên Cảnh Báo!】\n\n"
                "❌ Lỗi 1: Đọc 'hǎo' như 'hảo' tiếng Việt\n"
                "   → H tiếng Trung mạnh từ cổ họng, KHÁC hẳn H tiếng Việt\n"
                "   ✅ Đúng: Hà hơi mạnh từ cổ, có tiếng xước nhẹ\n\n"
                "❌ Lỗi 2: Đọc 你好 với cùng một Thanh 3\n"
                "   → Phải biến điệu: chữ đầu 你 đổi thành Thanh 2!\n"
                "   ✅ Đúng: NÍ hǎo (không phải NỈ hǎo)\n\n"
                "❌ Lỗi 3: Đọc 谢谢 cả hai chữ đều nặng\n"
                "   → Chữ 谢 thứ hai là THANH NHẸ, đọc lướt qua\n"
                "   ✅ Đúng: XIÊEEE-xie (nặng rồi nhạt)"
            ),
            "memory_tricks": (
                "【🧠 Mẹo Ghi Nhớ Siêu Hài – Không Quên Được!】\n\n"
                "🎯 你好 (Xin chào):\n"
                "• Nghe như 'NỈ HẢOO' = 'Này, Hào ơi!' (gọi tên người bạn tên Hào)\n"
                "• Câu nhớ: 'NỈ ơi, HẢOO, mày ổn không?' → Nǐ hǎo!\n\n"
                "🎯 谢谢 (Cảm ơn):\n"
                "• Nghe như 'XIÊEEE xiê' → Cảm ơn vì được ăn XIÊN QUE!\n"
                "• Câu nhớ: 'Ăn xiên que ngon quá, XIÊEEE XIÊEEE!'\n\n"
                "🎯 再见 (Tạm biệt):\n"
                "• Zài = lại/lần nữa; Jiàn = gặp → 'Gặp lại'\n"
                "• Nghe như 'ZÀI CHIÊN' → 'Zài chiên' = hẹn gặp lại!\n\n"
                "🎯 我 (Tôi):\n"
                "• Hình chữ: Người cầm GIÁO (戈) = 'Đó là TA đây!'\n"
                "• Nghe như 'ÙA' → 'ÙA ơi, đó là TÔI!'"
            ),
            "practical_examples": [
                {
                    "hanzi": "你好！我是小明。",
                    "pinyin": "Nǐ hǎo! Wǒ shì Xiǎo Míng.",
                    "vietnamese": "Xin chào! Tôi là Tiểu Minh.",
                    "note": "是 (shì) = là, Thanh 4 nặng"
                },
                {
                    "hanzi": "你好吗？",
                    "pinyin": "Nǐ hǎo ma?",
                    "vietnamese": "Bạn có khỏe không?",
                    "note": "吗 (ma) = trợ từ hỏi Yes/No, đọc thanh nhẹ"
                },
                {
                    "hanzi": "我很好，谢谢！",
                    "pinyin": "Wǒ hěn hǎo, xiè xie!",
                    "vietnamese": "Tôi rất khỏe, cảm ơn!",
                    "note": "很 (hěn) = rất, Thanh 3"
                },
                {
                    "hanzi": "再见！",
                    "pinyin": "Zài jiàn!",
                    "vietnamese": "Tạm biệt!",
                    "note": "Cả hai Thanh 4, đọc dứt khoát"
                }
            ],
            "reading_practice": {
                "level1_slow": [
                    {"text": "你 — 好", "pinyin": "nǐ — hǎo", "note": "Đọc từng chữ, dừng 1 giây"},
                    {"text": "谢 — 谢", "pinyin": "xiè — xie", "note": "Chữ đầu nặng, chữ sau nhạt"},
                    {"text": "再 — 见", "pinyin": "zài — jiàn", "note": "Cả hai Thanh 4, mạnh xuống"}
                ],
                "level2_tones": [
                    {"text": "Nǐ hǎo", "pinyin": "↗ ↘↗", "note": "你 đổi Thanh 2 (lên), 好 giữ Thanh 3 (xuống-lên)"},
                    {"text": "Wǒ hěn hǎo", "pinyin": "↘↗ ↘↗ ↘↗",
                     "note": "3 Thanh 3: 2 chữ đầu đổi Thanh 2, chữ cuối Thanh 3"},
                    {"text": "Xiè xie", "pinyin": "↘ (nhạt)", "note": "Nặng rồi lướt nhẹ"}
                ],
                "level3_natural": [
                    {"text": "你好！你好吗？", "pinyin": "Nǐ hǎo! Nǐ hǎo ma?", "note": "Tốc độ giao tiếp thực"},
                    {"text": "我很好，谢谢！再见！", "pinyin": "Wǒ hěn hǎo, xiè xie! Zài jiàn!",
                     "note": "Cả đoạn hội thoại liền mạch"}
                ]
            },
            "mini_quiz": [
                {
                    "question": "Khi đọc 你好 (nǐ hǎo), chữ 你 thay đổi thanh điệu như thế nào?",
                    "options": [
                        "A. Giữ nguyên Thanh 3",
                        "B. Đổi thành Thanh 1 (ngang bằng)",
                        "C. Đổi thành Thanh 2 (lên cao)",
                        "D. Đổi thành Thanh 4 (xuống mạnh)"
                    ],
                    "correct_answer": "C",
                    "explanation": (
                        "✅ Đáp án C ĐÚNG!\n\n"
                        "📚 Quy tắc BIẾN ĐIỆU 3→2:\n"
                        "Khi Thanh 3 đứng ngay trước Thanh 3 khác → chữ đầu đổi Thanh 2.\n\n"
                        "Vì sao? Hai Thanh 3 liên tiếp rất mệt miệng (cả hai xuống-lên), "
                        "nên người bản ngữ 'rút gọn': chữ đầu lên thẳng (Thanh 2), "
                        "chữ sau mới xuống-lên (Thanh 3).\n\n"
                        "🎯 Nhớ: NÍ hǎo (không phải NỈ hǎo)"
                    )
                },
                {
                    "question": "Âm H trong 好 (hǎo) có điểm gì khác âm H tiếng Việt?",
                    "options": [
                        "A. Hoàn toàn giống nhau",
                        "B. H tiếng Trung từ cổ họng, mạnh hơn, có tiếng xước",
                        "C. H tiếng Trung nhẹ hơn, không thở hơi",
                        "D. H tiếng Trung giống âm 'kh' hoàn toàn"
                    ],
                    "correct_answer": "B",
                    "explanation": (
                        "✅ Đáp án B ĐÚNG!\n\n"
                        "📚 H tiếng Trung là phụ âm CỔ HỌNG [x] trong IPA.\n"
                        "Gốc lưỡi nâng lên phía vòm mềm, hơi thoát từ cổ họng.\n\n"
                        "🔬 Kiểm tra: Hà hơi lên lòng bàn tay, nghe tiếng gió cọ xát "
                        "→ Đó là âm H đúng!\n\n"
                        "🎯 Mẹo: Hà hơi lên gương để lau = âm H tiếng Trung!"
                    )
                },
                {
                    "question": "谢谢 (xiè xie) được đọc đúng như thế nào?",
                    "options": [
                        "A. Xiê-Xiê (hai chữ đều nặng như nhau)",
                        "B. XIÊEEE-xie (chữ đầu Thanh 4, chữ sau thanh nhẹ lướt)",
                        "C. xiê-XIÊEEE (chữ sau nặng hơn)",
                        "D. Xiê xiê (cả hai Thanh 1 ngang)"
                    ],
                    "correct_answer": "B",
                    "explanation": (
                        "✅ Đáp án B ĐÚNG!\n\n"
                        "📚 谢谢 = Thanh 4 + Thanh nhẹ\n"
                        "• 谢 đầu: Thanh 4 → NẶNG, đi thẳng xuống\n"
                        "• 谢 sau: Thanh nhẹ → Nhạt, ngắn, không nhấn\n\n"
                        "Từ láy trong tiếng Trung thường có chữ thứ 2 đọc Thanh nhẹ.\n\n"
                        "🎯 Cảm giác: XIÊEEE-xie = Nặng rồi lướt, như phanh gấp!"
                    )
                },
                {
                    "question": "你好吗 (nǐ hǎo ma) có nghĩa là gì?",
                    "options": [
                        "A. Xin chào bạn!",
                        "B. Tôi rất khỏe",
                        "C. Bạn có khỏe không?",
                        "D. Tạm biệt bạn"
                    ],
                    "correct_answer": "C",
                    "explanation": (
                        "✅ Đáp án C ĐÚNG!\n\n"
                        "📚 Phân tích:\n"
                        "你(bạn) + 好(khỏe) + 吗(trợ từ hỏi) = Bạn có khỏe không?\n\n"
                        "吗 là trợ từ hỏi Yes/No – chỉ cần đặt cuối câu → câu hỏi!\n\n"
                        "🎯 Trả lời:\n"
                        "• 我很好！(Wǒ hěn hǎo) = Tôi rất khỏe!\n"
                        "• 还好 (Hái hǎo) = Bình thường thôi"
                    )
                }
            ]
        }
    ]
}


# ═══════════════════════════════════════════════════════════════════════════
#  QUẢN LÝ DỮ LIỆU (DataManager)
# ═══════════════════════════════════════════════════════════════════════════

class DataManager:
    """
    Chịu trách nhiệm đọc/ghi file JSON.
    Tự động tạo file mẫu nếu chưa tồn tại.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = self._load()

    def _load(self) -> dict:
        """Đọc file JSON. Nếu chưa có → tạo file mẫu."""
        if not os.path.exists(self.filepath):
            self._write_default()
            print(f"[DataManager] Đã tạo file mẫu: {self.filepath}")

        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[DataManager] Lỗi đọc file: {e} → Dùng dữ liệu mặc định")
            return DEFAULT_DATA

    def _write_default(self):
        """Ghi file JSON mẫu ra đĩa."""
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_DATA, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"[DataManager] Không thể ghi file: {e}")

    def get_lessons(self) -> list:
        """Trả về danh sách bài học."""
        return self.data.get("lessons", [])

    def get_lesson_by_id(self, lesson_id: str) -> Optional[dict]:
        """Lấy bài học theo ID."""
        for lesson in self.get_lessons():
            if lesson.get("id") == lesson_id:
                return lesson
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  LESSON RENDERER – Chuyển JSON → HTML đẹp
# ═══════════════════════════════════════════════════════════════════════════

class LessonRenderer:
    """
    Nhận dict bài học và trả về chuỗi HTML đầy đủ,
    sẵn sàng hiển thị trong QTextBrowser.
    """

    @staticmethod
    def render(lesson: dict) -> str:
        parts = []
        parts.append(LessonRenderer._html_header())
        parts.append(LessonRenderer._render_title(lesson))
        parts.append(LessonRenderer._render_vocabulary(lesson.get("vocabulary", [])))
        parts.append(LessonRenderer._render_section_box(
            "🔊 Phân Tích Ngữ Âm Chi Tiết",
            lesson.get("phonetic_analysis", ""),
            C_PURPLE, "#F9F0FF"
        ))
        parts.append(LessonRenderer._render_section_box(
            "👄 Hướng Dẫn Khẩu Hình",
            lesson.get("mouth_shape", ""),
            C_TEAL, "#E6FFFB"
        ))
        parts.append(LessonRenderer._render_section_box(
            "⚠️ Người Việt Hay Sai",
            lesson.get("vietnamese_errors", ""),
            C_ACCENT, "#FFF1F0"
        ))
        parts.append(LessonRenderer._render_section_box(
            "🧠 Mẹo Ghi Nhớ Siêu Hài",
            lesson.get("memory_tricks", ""),
            C_WARNING, "#FFF7E6"
        ))
        parts.append(LessonRenderer._render_examples(
            lesson.get("practical_examples", [])
        ))
        parts.append(LessonRenderer._render_reading_practice(
            lesson.get("reading_practice", {})
        ))
        parts.append(LessonRenderer._html_footer())
        return "".join(parts)

    @staticmethod
    def _html_header() -> str:
        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', 'Microsoft YaHei', 'Noto Sans CJK SC', sans-serif;
    background: {C_BG_MAIN};
    color: {C_TEXT_DARK};
    font-size: 15px;
    line-height: 1.7;
    padding: 0 4px;
  }}
  h1 {{ color: {C_PRIMARY}; font-size: 22px; margin-bottom: 6px; }}
  h2 {{ font-size: 17px; margin-bottom: 10px; }}
  h3 {{ font-size: 15px; color: {C_TEXT_MID}; }}
  .stage-badge {{
    display: inline-block;
    background: {C_PRIMARY};
    color: white;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 14px;
  }}
  .section-box {{
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 18px;
    border-left: 5px solid;
  }}
  .section-box pre, .section-box p {{
    white-space: pre-wrap;
    font-size: 14.5px;
    font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
    line-height: 1.8;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 18px;
  }}
  th {{
    background: {C_PRIMARY};
    color: white;
    padding: 9px 12px;
    text-align: left;
    font-size: 14px;
  }}
  td {{
    padding: 9px 12px;
    border-bottom: 1px solid {C_BORDER};
    font-size: 14px;
    vertical-align: top;
  }}
  tr:nth-child(even) td {{ background: #F8F9FC; }}
  .hanzi {{ font-size: 28px; color: {C_HANZI}; font-weight: bold; line-height: 1.3; }}
  .pinyin {{ font-size: 15px; color: {C_PRIMARY}; font-style: italic; }}
  .tone  {{ font-size: 13px; color: {C_PURPLE}; }}
  .vreading {{ font-size: 13px; color: {C_WARNING}; font-weight: 600; }}
  .meaning {{ font-size: 14px; color: {C_SUCCESS}; font-weight: 600; }}
  .example-block {{
    background: #F0FFF4;
    border: 1px solid #B7EB8F;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
  }}
  .example-hanzi {{ font-size: 20px; color: {C_HANZI}; font-weight: bold; }}
  .example-pinyin {{ font-size: 14px; color: {C_PRIMARY}; }}
  .example-viet {{ font-size: 14px; color: {C_TEXT_DARK}; }}
  .example-note {{ font-size: 12.5px; color: {C_TEXT_LIGHT}; margin-top: 4px; font-style: italic; }}
  .practice-block {{
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 10px 14px;
    border-radius: 8px;
    margin-bottom: 8px;
    background: #F0FBFF;
    border: 1px solid #BAE7FF;
  }}
  .practice-text {{ font-size: 18px; color: {C_HANZI}; font-weight: bold; }}
  .practice-pinyin {{ font-size: 14px; color: {C_PRIMARY}; }}
  .practice-note {{ font-size: 12.5px; color: {C_TEXT_MID}; font-style: italic; margin-top: 3px; }}
  .level-header {{
    font-size: 14px;
    font-weight: 700;
    color: white;
    padding: 5px 14px;
    border-radius: 6px;
    margin-bottom: 10px;
    display: inline-block;
  }}
  .divider {{
    height: 1px;
    background: linear-gradient(to right, {C_BORDER}, transparent);
    margin: 20px 0;
  }}
</style></head><body>
"""

    @staticmethod
    def _html_footer() -> str:
        return "</body></html>"

    @staticmethod
    def _render_title(lesson: dict) -> str:
        title = lesson.get("title", "Bài Học")
        stage = lesson.get("stage", "")
        return f"""
<div style="margin-bottom: 20px;">
  <div class="stage-badge">{stage}</div>
  <h1>{title}</h1>
  <div style="height:3px; background:linear-gradient(to right,{C_PRIMARY},{C_TEAL},{C_ACCENT}); border-radius:2px; margin-top:8px;"></div>
</div>
"""

    @staticmethod
    def _render_vocabulary(vocab_list: list) -> str:
        if not vocab_list:
            return ""
        rows = ""
        for v in vocab_list:
            hanzi = v.get("hanzi", "")
            pinyin = v.get("pinyin", "")
            tone = v.get("tone", "")
            vread = v.get("vietnamese_reading", "")
            meaning = v.get("meaning", "")
            rows += f"""
<tr>
  <td><span class="hanzi">{hanzi}</span></td>
  <td><span class="pinyin">{pinyin}</span></td>
  <td><span class="tone">{tone}</span></td>
  <td><span class="vreading">📣 {vread}</span></td>
  <td><span class="meaning">{meaning}</span></td>
</tr>"""

        return f"""
<div style="margin-bottom:18px;">
  <h2 style="color:{C_PRIMARY}; border-bottom:2px solid {C_PRIMARY}; padding-bottom:6px; margin-bottom:12px;">
    📚 Từ Mới Trong Bài
  </h2>
  <table>
    <tr>
      <th>Chữ Hán</th>
      <th>Pinyin</th>
      <th>Thanh Điệu</th>
      <th>Đọc Bồi (VN)</th>
      <th>Nghĩa</th>
    </tr>
    {rows}
  </table>
</div>
"""

    @staticmethod
    def _render_section_box(title: str, content: str, border_color: str, bg_color: str) -> str:
        if not content:
            return ""
        # Escape HTML cơ bản
        safe = (content.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))
        return f"""
<div class="section-box" style="border-color:{border_color}; background:{bg_color};">
  <h2 style="color:{border_color}; margin-bottom:10px;">{title}</h2>
  <pre style="color:{C_TEXT_DARK};">{safe}</pre>
</div>
"""

    @staticmethod
    def _render_examples(examples: list) -> str:
        if not examples:
            return ""
        items = ""
        for ex in examples:
            hanzi = ex.get("hanzi", "")
            pinyin = ex.get("pinyin", "")
            viet = ex.get("vietnamese", "")
            note = ex.get("note", "")
            note_html = f'<div class="example-note">💡 {note}</div>' if note else ""
            items += f"""
<div class="example-block">
  <div class="example-hanzi">{hanzi}</div>
  <div class="example-pinyin">{pinyin}</div>
  <div class="example-viet">🇻🇳 {viet}</div>
  {note_html}
</div>"""

        return f"""
<div style="margin-bottom:18px;">
  <h2 style="color:{C_SUCCESS}; border-bottom:2px solid {C_SUCCESS}; padding-bottom:6px; margin-bottom:12px;">
    💬 Câu Ví Dụ Thực Tế
  </h2>
  {items}
</div>
"""

    @staticmethod
    def _render_reading_practice(practice: dict) -> str:
        if not practice:
            return ""

        levels = [
            ("level1_slow", "🐢 Cấp 1: Đọc Chậm Từng Chữ", "#389E0D"),
            ("level2_tones", "🎵 Cấp 2: Đọc Chuẩn Thanh Điệu", C_WARNING),
            ("level3_natural", "🗣️ Cấp 3: Đọc Tự Nhiên Giao Tiếp", C_ACCENT),
        ]
        html = f"""
<div style="margin-bottom:18px;">
  <h2 style="color:{C_TEAL}; border-bottom:2px solid {C_TEAL}; padding-bottom:6px; margin-bottom:14px;">
    🎤 Bài Luyện Đọc – 3 Cấp Độ
  </h2>
"""
        for key, label, color in levels:
            items = practice.get(key, [])
            if not items:
                continue
            html += f'<div><span class="level-header" style="background:{color};">{label}</span></div>'
            for item in items:
                text = item.get("text", "")
                pinyin = item.get("pinyin", "")
                note = item.get("note", "")
                html += f"""
<div class="practice-block">
  <div>
    <div class="practice-text">{text}</div>
    <div class="practice-pinyin">{pinyin}</div>
    <div class="practice-note">📌 {note}</div>
  </div>
</div>"""
            html += '<div style="height:12px;"></div>'

        html += "</div>"
        return html


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 1: NỀN TẢNG CHO NGƯỜI MỚI
# ═══════════════════════════════════════════════════════════════════════════

class FoundationTab(QScrollArea):
    """Tab giải thích Pinyin, 4 Thanh điệu, hệ thống âm tiếng Trung."""

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(0)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setFrameShape(QFrame.NoFrame)
        browser.setStyleSheet("background: transparent; border: none;")
        browser.setHtml(self._build_html())
        browser.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout.addWidget(browser)
        self.setWidget(content)

    def _build_html(self) -> str:
        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  body {{
    font-family:'Segoe UI','Microsoft YaHei',sans-serif;
    background:{C_BG_MAIN}; color:{C_TEXT_DARK};
    font-size:15px; line-height:1.75; padding:4px;
  }}
  h1 {{ color:{C_PRIMARY}; font-size:24px; margin-bottom:6px; }}
  h2 {{ color:{C_PRIMARY}; font-size:18px; margin:20px 0 10px; }}
  h3 {{ color:{C_PURPLE}; font-size:15px; margin:14px 0 6px; }}
  .intro-box {{
    background:linear-gradient(135deg,{C_PRIMARY}22,{C_TEAL}22);
    border:2px solid {C_PRIMARY};
    border-radius:12px;
    padding:18px 24px;
    margin-bottom:24px;
  }}
  .tone-table {{ width:100%; border-collapse:collapse; margin:14px 0; }}
  .tone-table th {{
    background:{C_PRIMARY}; color:white;
    padding:10px 14px; font-size:14px;
  }}
  .tone-table td {{
    padding:10px 14px;
    border-bottom:1px solid {C_BORDER};
    font-size:14px; vertical-align:top;
  }}
  tr:nth-child(even) td {{ background:#F8F9FC; }}
  .tone1 {{ color:#0078D7; font-size:22px; font-weight:bold; }}
  .tone2 {{ color:#389E0D; font-size:22px; font-weight:bold; }}
  .tone3 {{ color:#722ED1; font-size:22px; font-weight:bold; }}
  .tone4 {{ color:#FF4D4F; font-size:22px; font-weight:bold; }}
  .toneN {{ color:#888;   font-size:22px; font-weight:bold; }}
  .card {{
    border-radius:10px; padding:16px 20px;
    margin-bottom:14px; border-left:5px solid;
  }}
  .diagram-ascii {{
    font-family:monospace; font-size:16px;
    background:#1A1A2E; color:#E0E0FF;
    padding:16px 20px; border-radius:8px;
    line-height:1.6; white-space:pre;
    margin:14px 0;
  }}
</style></head><body>

<div class="intro-box">
  <h1>☘️ Nền Tảng Cho Người Mới Bắt Đầu</h1>
  <p style="font-size:16px; color:{C_TEXT_MID};">
    Trước khi học bất cứ từ nào, hãy hiểu rõ <strong>3 trụ cột</strong> của tiếng Trung:
    <strong>Pinyin</strong>, <strong>Thanh điệu</strong> và <strong>Hệ thống âm tiết</strong>.
    Đọc kỹ phần này → bạn sẽ không bị lạc sau này!
  </p>
</div>

<!-- ────────────────────── PINYIN ────────────────────── -->
<h2>🔤 1. Pinyin Là Gì?</h2>
<div class="card" style="border-color:{C_PRIMARY}; background:#EFF6FF;">
  <p><strong>Pinyin (拼音)</strong> = hệ thống phiên âm La-tinh để đọc tiếng Trung.</p>
  <p style="margin-top:8px;">Tiếng Trung <strong>không có bảng chữ cái</strong> như tiếng Việt. 
  Thay vào đó, mỗi chữ Hán (汉字) có một âm đọc riêng, và Pinyin là cách ghi lại âm đó bằng chữ La-tinh.</p>
  <p style="margin-top:8px;">Ví dụ: <span style="color:{C_HANZI};font-size:20px;font-weight:bold;">你好</span> 
  → Pinyin: <span style="color:{C_PRIMARY};font-style:italic;font-size:16px;">nǐ hǎo</span> 
  → Nghĩa: <span style="color:{C_SUCCESS};font-weight:bold;">Xin chào</span></p>
  <p style="margin-top:8px; color:{C_WARNING};">
  ⚠️ Pinyin trông giống tiếng Anh/Việt nhưng cách đọc <strong>KHÁC HOÀN TOÀN</strong>! 
  Đừng đọc theo bản năng tiếng Việt!
  </p>
</div>

<h3>📐 Cấu trúc một âm tiết Pinyin:</h3>
<div class="diagram-ascii">  [Phụ âm đầu] + [Nguyên âm / Vần] + [Thanh điệu]

  Ví dụ:    m   +   ā   +   (thanh 1)  =  mā (妈 = mẹ)
            n   +   ǐ   +   (thanh 3)  =  nǐ (你 = bạn)
            h   +   ǎo  +   (thanh 3)  =  hǎo (好 = tốt)
            (không có)  +   ā   =  ā (啊 = à!)
</div>

<!-- ────────────────────── 4 THANH ĐIỆU ────────────────────── -->
<h2>🎵 2. Bốn Thanh Điệu – Điều Làm Tiếng Trung Đặc Biệt</h2>
<div class="card" style="border-color:{C_ACCENT}; background:#FFF1F0;">
  <p>Tiếng Trung là <strong>ngôn ngữ thanh điệu</strong> – cùng một Pinyin nhưng đọc khác thanh = nghĩa khác nhau!</p>
  <p style="margin-top:8px;">Nổi tiếng nhất là ví dụ chữ <strong>MA</strong>:</p>
  <table style="margin-top:10px;width:auto;">
    <tr>
      <td style="padding:6px 16px;"><span class="tone1">mā</span></td>
      <td style="padding:6px 16px;font-size:14px;">妈 = Mẹ</td>
    </tr>
    <tr style="background:#FFF;">
      <td style="padding:6px 16px;"><span class="tone2">má</span></td>
      <td style="padding:6px 16px;font-size:14px;">麻 = Cây gai dầu / Tê liệt</td>
    </tr>
    <tr>
      <td style="padding:6px 16px;"><span class="tone3">mǎ</span></td>
      <td style="padding:6px 16px;font-size:14px;">马 = Con ngựa</td>
    </tr>
    <tr style="background:#FFF;">
      <td style="padding:6px 16px;"><span class="tone4">mà</span></td>
      <td style="padding:6px 16px;font-size:14px;">骂 = Chửi rủa</td>
    </tr>
    <tr>
      <td style="padding:6px 16px;"><span class="toneN">ma</span></td>
      <td style="padding:6px 16px;font-size:14px;">吗 = Trợ từ hỏi (nhẹ)</td>
    </tr>
  </table>
  <p style="margin-top:10px; font-style:italic; color:{C_TEXT_LIGHT};">
  → Đọc sai thanh có thể nói "Tôi muốn mua CON NGỰA" thành "Tôi muốn CHỬI MẸ"! 😱
  </p>
</div>

<h3>📊 Bảng Chi Tiết 4 Thanh + Thanh Nhẹ:</h3>
<table class="tone-table">
  <tr><th>Ký Hiệu</th><th>Tên</th><th>Hình Dạng</th><th>Hình Dung Thực Tế</th><th>Ví Dụ</th></tr>
  <tr>
    <td><span class="tone1">ā (–)</span></td>
    <td><strong>Thanh 1</strong><br>Thanh Bằng<br>一声</td>
    <td style="font-size:20px;">—</td>
    <td>Bác sĩ bảo: "Há miệng ra... ÀÀÀ..." (kéo dài ngang, không lên không xuống)</td>
    <td><span style="color:{C_HANZI};font-size:18px;">妈</span> mā = mẹ<br>
        <span style="color:{C_HANZI};font-size:18px;">书</span> shū = sách</td>
  </tr>
  <tr>
    <td><span class="tone2">á (↗)</span></td>
    <td><strong>Thanh 2</strong><br>Thanh Sắc<br>二声</td>
    <td style="font-size:20px;">↗</td>
    <td>Hỏi lại khi không nghe rõ: "Hả?!" (giọng lên cao như ngạc nhiên)</td>
    <td><span style="color:{C_HANZI};font-size:18px;">麻</span> má = gai dầu<br>
        <span style="color:{C_HANZI};font-size:18px;">人</span> rén = người</td>
  </tr>
  <tr>
    <td><span class="tone3">ǎ (↘↗)</span></td>
    <td><strong>Thanh 3</strong><br>Thanh Hỏi<br>三声</td>
    <td style="font-size:20px;">↘↗</td>
    <td>Giọng người do dự: "Ờ... ờ... (xuống thấp rồi kéo lên) để tôi nghĩ đã..."</td>
    <td><span style="color:{C_HANZI};font-size:18px;">马</span> mǎ = ngựa<br>
        <span style="color:{C_HANZI};font-size:18px;">好</span> hǎo = tốt</td>
  </tr>
  <tr>
    <td><span class="tone4">à (↘)</span></td>
    <td><strong>Thanh 4</strong><br>Thanh Nặng<br>四声</td>
    <td style="font-size:20px;">↘</td>
    <td>Giận dữ, tức giận: "ĐỦ RỒI!" (rơi thẳng xuống mạnh, dứt khoát)</td>
    <td><span style="color:{C_HANZI};font-size:18px;">骂</span> mà = chửi<br>
        <span style="color:{C_HANZI};font-size:18px;">是</span> shì = là</td>
  </tr>
  <tr>
    <td><span class="toneN">a (nhẹ)</span></td>
    <td><strong>Thanh Nhẹ</strong><br>Thanh Nhạt<br>轻声</td>
    <td style="font-size:20px;">·</td>
    <td>Đọc lướt qua, nhạt, ngắn, không nhấn – như tiếng thở thêm vào</td>
    <td><span style="color:{C_HANZI};font-size:18px;">吗</span> ma = hỏi<br>
        <span style="color:{C_HANZI};font-size:18px;">谢谢</span> xiè·xie</td>
  </tr>
</table>

<!-- ────────────────────── VỊ TRÍ DẤU ────────────────────── -->
<h2>📍 3. Cách Đặt Dấu Thanh Trong Pinyin</h2>
<div class="card" style="border-color:{C_SUCCESS}; background:#F6FFED;">
  <p>Dấu thanh luôn đặt trên <strong>nguyên âm chính</strong> của âm tiết theo thứ tự ưu tiên:</p>
  <p style="margin-top:8px; font-size:15px;">
    <strong>a / e</strong> → Luôn đặt dấu ở đây nếu có<br>
    <strong>ou</strong> → Đặt ở o<br>
    <strong>Nguyên âm cuối</strong> → Đặt ở nguyên âm cuối cùng nếu không có quy tắc trên
  </p>
  <p style="margin-top:10px; font-size:14px; color:{C_TEXT_LIGHT};">
    Ví dụ: guó (国=nước) → dấu đặt ở o vì không có a/e<br>
    Ví dụ: hǎo (好=tốt) → dấu đặt ở a vì có a
  </p>
</div>

<!-- ────────────────────── ÂM KHÓ ────────────────────── -->
<h2>🔥 4. Những Âm Khó Nhất Với Người Việt</h2>
<table class="tone-table">
  <tr><th>Âm</th><th>Ví Dụ</th><th>Tại Sao Khó?</th><th>Mẹo Luyện</th></tr>
  <tr>
    <td><strong style="color:{C_ACCENT};">zh ch sh r</strong></td>
    <td>知吃师日</td>
    <td>Âm cuộn lưỡi – tiếng Việt không có!</td>
    <td>Cong đầu lưỡi ra sau như đang liếm vòm cứng</td>
  </tr>
  <tr>
    <td><strong style="color:{C_ACCENT};">z c s</strong></td>
    <td>字此丝</td>
    <td>Âm răng – nghe gần giống zh ch sh nhưng lưỡi phẳng</td>
    <td>Lưỡi phẳng, đầu lưỡi sau răng dưới</td>
  </tr>
  <tr>
    <td><strong style="color:{C_ACCENT};">ü</strong></td>
    <td>鱼女</td>
    <td>Nguyên âm môi tròn – tiếng Việt không có</td>
    <td>Nói "i" nhưng tròn môi như nói "u"</td>
  </tr>
  <tr>
    <td><strong style="color:{C_ACCENT};">b p d t</strong></td>
    <td>爸怕大他</td>
    <td>b/d không bật hơi; p/t bật hơi mạnh</td>
    <td>Để giấy trước miệng: p/t thổi bay giấy, b/d không</td>
  </tr>
</table>

<div style="height:20px;"></div>
</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 2: LỘ TRÌNH HỌC 5 GIAI ĐOẠN
# ═══════════════════════════════════════════════════════════════════════════

class RoadmapTab(QScrollArea):
    """Tab hiển thị lộ trình học 5 giai đoạn."""

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 20, 20, 20)

        browser = QTextBrowser()
        browser.setFrameShape(QFrame.NoFrame)
        browser.setStyleSheet("background: transparent; border: none;")
        browser.setHtml(self._build_html())

        layout.addWidget(browser)
        self.setWidget(content)

    def _build_html(self) -> str:
        stages = [
            {
                "num": "1",
                "title": "Pinyin & Thanh Điệu",
                "duration": "2-4 tuần",
                "color": C_PRIMARY,
                "icon": "🔤",
                "goals": [
                    "Thuộc 4 thanh điệu và thanh nhẹ",
                    "Đọc được tất cả phụ âm đầu (b, p, m, f, d, t, n, l, g, k, h, j, q, x, zh, ch, sh, r, z, c, s, y, w)",
                    "Đọc được tất cả nguyên âm đơn và kép (a, o, e, i, u, ü, ai, ei, ao, ou...)",
                    "Hiểu quy tắc biến điệu 3→2 và biến điệu của 不/一",
                    "Luyện cơ bắp miệng với âm cuộn lưỡi (zh, ch, sh, r)"
                ],
                "resources": "Bảng Pinyin đầy đủ, video phát âm chuẩn, ứng dụng này"
            },
            {
                "num": "2",
                "title": "Từ Vựng Cơ Bản 500 Từ",
                "duration": "2-3 tháng",
                "color": C_SUCCESS,
                "icon": "📚",
                "goals": [
                    "Học 500 từ vựng HSK 1-2 (cấp độ cơ bản nhất)",
                    "Nắm chắc 100 từ siêu phổ biến nhất (好/是/不/有/我/你/他...)",
                    "Học số đếm 0-100, ngày tháng, thời gian",
                    "Học màu sắc, đồ vật trong nhà, thức ăn cơ bản",
                    "Bắt đầu nhận biết chữ Hán qua nét và bộ thủ"
                ],
                "resources": "Anki flashcard, ứng dụng này, HSK 1-2 wordlist"
            },
            {
                "num": "3",
                "title": "Ngữ Pháp & Mẫu Câu",
                "duration": "2-3 tháng",
                "color": C_PURPLE,
                "icon": "📝",
                "goals": [
                    "Cấu trúc câu cơ bản: Chủ ngữ + Vị ngữ + Tân ngữ",
                    "Câu phủ định: 不 + động từ; 没有 + động từ",
                    "Câu hỏi: 吗? (Yes/No), 什么/哪/谁/哪儿 (WH-questions)",
                    "Trợ từ thời gian: 了 (rồi), 过 (đã từng), 着 (đang)",
                    "Chỉ thị: 这/那 (này/kia), lượng từ (一个, 两本...)",
                    "20 mẫu câu giao tiếp cơ bản"
                ],
                "resources": "Sách 'Hán ngữ 300 câu', ứng dụng này"
            },
            {
                "num": "4",
                "title": "Hội Thoại Thực Tế",
                "duration": "3-6 tháng",
                "color": C_WARNING,
                "icon": "💬",
                "goals": [
                    "Tự giới thiệu bản thân hoàn chỉnh (tên, tuổi, nghề, quê hương)",
                    "Mua bán ở chợ/shop: hỏi giá, trả giá, thanh toán",
                    "Gọi đồ ăn ở nhà hàng, hỏi đường",
                    "Nói chuyện về gia đình, sở thích, kế hoạch",
                    "Nghe và hiểu hội thoại tốc độ bình thường (50-60%)",
                    "Xem phim hoạt hình Trung Quốc với phụ đề Pinyin"
                ],
                "resources": "YouTube: Easy Chinese, ứng dụng HelloChinese, bạn ngôn ngữ"
            },
            {
                "num": "5",
                "title": "Đọc Viết & Thi HSK",
                "duration": "6-12 tháng",
                "color": C_ACCENT,
                "icon": "🏆",
                "goals": [
                    "Nhận biết và viết được 300 chữ Hán cơ bản (HSK 1-2)",
                    "Đọc hiểu đoạn văn ngắn không cần Pinyin",
                    "Hiểu bộ thủ (部首) để đoán nghĩa chữ mới",
                    "Viết đoạn văn tự giới thiệu (100-200 chữ)",
                    "Thi đậu HSK 3 (tương đương A2-B1 châu Âu)",
                    "Theo dõi mạng xã hội WeChat/Weibo, xem phim không phụ đề"
                ],
                "resources": "Sách luyện thi HSK, thầy cô bản ngữ, du lịch Trung Quốc!"
            }
        ]

        cards_html = ""
        for s in stages:
            goals_html = "".join(
                f'<li style="margin-bottom:5px; font-size:14px;">{g}</li>'
                for g in s["goals"]
            )
            cards_html += f"""
<div style="
  border-radius:12px;
  border-left:6px solid {s['color']};
  background:white;
  padding:20px 24px;
  margin-bottom:18px;
  box-shadow:0 2px 8px rgba(0,0,0,0.07);
">
  <div style="display:flex; align-items:center; margin-bottom:12px;">
    <div style="
      background:{s['color']};
      color:white;
      width:40px; height:40px;
      border-radius:50%;
      display:flex; align-items:center; justify-content:center;
      font-size:18px; font-weight:bold;
      text-align:center; line-height:40px;
      min-width:40px;
      margin-right:14px;
    ">{s['num']}</div>
    <div>
      <div style="font-size:20px;">{s['icon']}&nbsp;
        <strong style="color:{s['color']};">{s['title']}</strong>
      </div>
      <div style="font-size:13px; color:{C_TEXT_LIGHT};">⏱️ Thời gian ước tính: <strong>{s['duration']}</strong></div>
    </div>
  </div>
  <ul style="padding-left:22px; color:{C_TEXT_MID};">
    {goals_html}
  </ul>
  <div style="
    margin-top:12px;
    padding:8px 14px;
    background:{s['color']}18;
    border-radius:6px;
    font-size:13px; color:{C_TEXT_MID};
  ">
    📖 <strong>Tài nguyên:</strong> {s['resources']}
  </div>
</div>
"""

        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  body {{
    font-family:'Segoe UI','Microsoft YaHei',sans-serif;
    background:{C_BG_MAIN}; color:{C_TEXT_DARK};
    font-size:15px; line-height:1.7; padding:4px;
  }}
  h1 {{ color:{C_PRIMARY}; font-size:24px; margin-bottom:8px; }}
  li {{ line-height:1.7; }}
</style></head><body>

<div style="
  background:linear-gradient(135deg, {C_PRIMARY}, {C_TEAL});
  color:white; border-radius:14px;
  padding:22px 28px; margin-bottom:26px;
">
  <h1 style="color:white; font-size:24px; margin-bottom:8px;">🗺️ Lộ Trình Học Tiếng Trung 5 Giai Đoạn</h1>
  <p style="font-size:15px; opacity:0.95; margin:0;">
    Từ <strong>con số 0</strong> đến <strong>giao tiếp tự nhiên</strong> – Mỗi giai đoạn xây nền cho giai đoạn sau.
    Đừng vội nhảy cóc – học chắc từng bước là nhanh nhất!
  </p>
</div>

<div style="
  background:#FFFBE6; border:2px solid {C_WARNING};
  border-radius:10px; padding:14px 20px; margin-bottom:22px;
">
  <strong>💡 Lời khuyên từ giáo viên:</strong> Trung bình người Việt cần 
  <strong>1-2 năm học nghiêm túc</strong> (1-2 tiếng/ngày) để đạt HSK 3. 
  Nếu học 30 phút mỗi ngày và kiên trì, bạn sẽ cảm nhận được sự tiến bộ rõ ràng sau 
  <strong>3-6 tháng đầu tiên</strong>.
</div>

{cards_html}

<div style="
  background:{C_SUCCESS}15; border:2px solid {C_SUCCESS};
  border-radius:10px; padding:16px 20px; margin-top:8px;
">
  <h3 style="color:{C_SUCCESS}; margin-bottom:8px;">🌟 Bí Quyết Thành Công</h3>
  <ul style="padding-left:20px; font-size:14px; color:{C_TEXT_MID};">
    <li><strong>Nghe mỗi ngày</strong> – 15-30 phút nhạc/podcast Trung dù không hiểu</li>
    <li><strong>Nói to từng chữ</strong> – Không chỉ nhìn trong đầu, phải dùng miệng!</li>
    <li><strong>Lặp lại có khoảng cách</strong> – Ôn lại sau 1 ngày, 3 ngày, 7 ngày, 1 tháng</li>
    <li><strong>Đừng sợ sai</strong> – Người Trung thích khi người nước ngoài cố nói tiếng Trung</li>
    <li><strong>Tìm bạn ngôn ngữ</strong> – Apps như HelloTalk, Tandem, iTalki</li>
  </ul>
</div>

<div style="height:20px;"></div>
</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 3: BÀI HỌC CHI TIẾT + MINI QUIZ
# ═══════════════════════════════════════════════════════════════════════════

class LessonTab(QWidget):
    """
    Tab chính: Sidebar chọn bài học + Nội dung bài + Mini Quiz.
    Layout: QSplitter (sidebar | nội dung)
    """

    def __init__(self, data_manager: DataManager):
        super().__init__()
        self.data_manager = data_manager
        self.current_lesson = None
        self.quiz_groups = []  # Danh sách QButtonGroup cho từng câu hỏi
        self._setup_ui()
        self._load_lesson_list()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)

        # ── Sidebar ──────────────────────────────────────
        sidebar = QWidget()
        sidebar.setStyleSheet(f"background: {C_BG_SIDEBAR};")
        sidebar.setMinimumWidth(200)
        sidebar.setMaximumWidth(260)
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(12, 16, 12, 16)
        sb_layout.setSpacing(10)

        lbl = QLabel("📖 Danh Sách Bài Học")
        lbl.setStyleSheet(f"""
            font-size: 14px; font-weight: bold;
            color: {C_PRIMARY}; padding-bottom: 6px;
            border-bottom: 2px solid {C_PRIMARY};
        """)
        sb_layout.addWidget(lbl)

        self.lesson_list = QListWidget()
        self.lesson_list.setStyleSheet(f"""
            QListWidget {{
                border: 1px solid {C_BORDER};
                border-radius: 8px;
                font-size: 13px;
                background: white;
                outline: none;
            }}
            QListWidget::item {{
                padding: 10px 12px;
                border-bottom: 1px solid {C_BORDER};
                color: {C_TEXT_DARK};
            }}
            QListWidget::item:selected {{
                background: {C_PRIMARY};
                color: white;
                border-radius: 6px;
            }}
            QListWidget::item:hover:!selected {{
                background: {C_PRIMARY}18;
            }}
        """)
        self.lesson_list.currentRowChanged.connect(self._on_lesson_selected)
        sb_layout.addWidget(self.lesson_list)

        # Nút reload
        btn_reload = QPushButton("🔄 Tải Lại Dữ Liệu")
        btn_reload.setStyleSheet(f"""
            QPushButton {{
                background: {C_PRIMARY}; color: white;
                border-radius: 6px; padding: 8px;
                font-size: 13px; font-weight: bold; border: none;
            }}
            QPushButton:hover {{ background: {C_PRIMARY_DARK}; }}
        """)
        btn_reload.clicked.connect(self._reload_data)
        sb_layout.addWidget(btn_reload)

        # ── Khu vực nội dung (có scroll) ─────────────────
        self.content_scroll = QScrollArea()
        self.content_scroll.setWidgetResizable(True)
        self.content_scroll.setFrameShape(QFrame.NoFrame)
        self.content_scroll.setStyleSheet(f"background: {C_BG_MAIN};")

        self.content_widget = QWidget()
        self.content_widget.setStyleSheet(f"background: {C_BG_MAIN};")
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(18, 18, 18, 18)
        self.content_layout.setSpacing(0)

        self.content_scroll.setWidget(self.content_widget)

        splitter.addWidget(sidebar)
        splitter.addWidget(self.content_scroll)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

    def _load_lesson_list(self):
        """Đổ danh sách bài học vào QListWidget."""
        self.lesson_list.clear()
        for lesson in self.data_manager.get_lessons():
            item = QListWidgetItem(lesson.get("title", "Không rõ tên"))
            item.setData(Qt.UserRole, lesson.get("id"))
            self.lesson_list.addItem(item)

        if self.lesson_list.count() > 0:
            self.lesson_list.setCurrentRow(0)

    def _on_lesson_selected(self, row: int):
        """Xử lý khi người dùng chọn bài học."""
        if row < 0:
            return
        item = self.lesson_list.item(row)
        if not item:
            return
        lesson_id = item.data(Qt.UserRole)
        lesson = self.data_manager.get_lesson_by_id(lesson_id)
        if lesson:
            self.current_lesson = lesson
            self._render_lesson(lesson)

    def _render_lesson(self, lesson: dict):
        """Xóa nội dung cũ và hiển thị bài học mới."""
        # Xóa widget cũ
        while self.content_layout.count():
            child = self.content_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.quiz_groups.clear()

        # ── Phần HTML nội dung bài ──────────────────────
        html_content = LessonRenderer.render(lesson)

        lesson_browser = QTextBrowser()
        lesson_browser.setHtml(html_content)
        lesson_browser.setFrameShape(QFrame.NoFrame)
        lesson_browser.setStyleSheet(f"""
            QTextBrowser {{
                background: {C_BG_MAIN};
                border: none;
                font-size: 15px;
            }}
        """)
        lesson_browser.document().setDefaultFont(
            QFont("Segoe UI", 10)
        )
        self.content_layout.addWidget(lesson_browser)

        # ── Khu vực Mini Quiz ───────────────────────────
        quiz_data = lesson.get("mini_quiz", [])
        if quiz_data:
            self.content_layout.addWidget(self._build_quiz_widget(quiz_data))

        self.content_layout.addSpacerItem(
            QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

    def _build_quiz_widget(self, quiz_data: list) -> QWidget:
        """Tạo widget Mini Quiz tương tác."""
        container = QGroupBox("🎯 MINI QUIZ – Kiểm Tra Kiến Thức")
        container.setStyleSheet(f"""
            QGroupBox {{
                font-size: 16px;
                font-weight: bold;
                color: {C_PRIMARY};
                border: 2px solid {C_PRIMARY};
                border-radius: 12px;
                margin-top: 24px;
                padding-top: 14px;
                background: {C_QUIZ_BG};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                padding: 0 12px;
                background: {C_QUIZ_BG};
            }}
        """)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(16)

        self.quiz_groups = []

        for idx, q in enumerate(quiz_data):
            q_widget = self._build_question_widget(idx + 1, q)
            layout.addWidget(q_widget)

        # Nút nộp bài
        btn_submit = QPushButton("📝  Nộp Bài Cho Giáo Viên 1-1  ✔️")
        btn_submit.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {C_PRIMARY}, stop:1 {C_TEAL});
                color: white;
                border-radius: 10px;
                padding: 13px 20px;
                font-size: 15px;
                font-weight: bold;
                border: none;
                margin-top: 8px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {C_PRIMARY_DARK}, stop:1 {C_PRIMARY});
            }}
            QPushButton:pressed {{
                padding-top: 15px; padding-bottom: 11px;
            }}
        """)
        btn_submit.setMinimumHeight(48)
        btn_submit.clicked.connect(
            lambda: self._submit_quiz(quiz_data)
        )
        layout.addWidget(btn_submit)

        return container

    def _build_question_widget(self, num: int, q_data: dict) -> QWidget:
        """Tạo widget cho một câu hỏi trắc nghiệm."""
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background: white;
                border: 1px solid {C_BORDER};
                border-radius: 10px;
                padding: 4px;
            }}
        """)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        # Câu hỏi
        q_label = QLabel(f"Câu {num}: {q_data.get('question', '')}")
        q_label.setStyleSheet(f"""
            font-size: 14px;
            font-weight: bold;
            color: {C_TEXT_DARK};
            padding: 4px 0;
        """)
        q_label.setWordWrap(True)
        layout.addWidget(q_label)

        # Các lựa chọn
        btn_group = QButtonGroup(frame)
        btn_group.setExclusive(True)

        options = q_data.get("options", [])
        for opt in options:
            rb = QRadioButton(opt)
            rb.setStyleSheet(f"""
                QRadioButton {{
                    font-size: 13.5px;
                    color: {C_TEXT_DARK};
                    padding: 4px 0;
                    spacing: 8px;
                }}
                QRadioButton::indicator {{
                    width: 16px; height: 16px;
                }}
                QRadioButton:hover {{
                    color: {C_PRIMARY};
                }}
            """)
            btn_group.addButton(rb)
            layout.addWidget(rb)

        self.quiz_groups.append(btn_group)
        return frame

    def _submit_quiz(self, quiz_data: list):
        """Chấm điểm và hiển thị kết quả chi tiết."""
        score = 0
        results = []

        for idx, (q_data, btn_group) in enumerate(
                zip(quiz_data, self.quiz_groups)
        ):
            selected = btn_group.checkedButton()
            if selected is None:
                results.append({
                    "num": idx + 1,
                    "answered": False,
                    "correct": False,
                    "explanation": q_data.get("explanation", "")
                })
                continue

            # Lấy ký tự đầu tiên (A/B/C/D) từ nội dung RadioButton
            answer_text = selected.text()
            chosen_letter = answer_text[0].upper() if answer_text else ""
            correct_letter = q_data.get("correct_answer", "").upper()
            is_correct = (chosen_letter == correct_letter)

            if is_correct:
                score += 1

            results.append({
                "num": idx + 1,
                "question": q_data.get("question", ""),
                "answered": True,
                "correct": is_correct,
                "chosen": answer_text,
                "correct_answer": correct_letter,
                "explanation": q_data.get("explanation", "")
            })

        total = len(quiz_data)
        unanswered = sum(1 for r in results if not r["answered"])

        if unanswered > 0:
            msg = QMessageBox(self)
            msg.setWindowTitle("⚠️ Chưa Trả Lời Hết!")
            msg.setText(
                f"Bạn còn <b>{unanswered} câu chưa trả lời</b>.<br>"
                f"Hãy chọn đáp án cho tất cả câu trước khi nộp bài nhé!"
            )
            msg.setIcon(QMessageBox.Warning)
            msg.exec_()
            return

        # Hiển thị kết quả
        self._show_results_dialog(score, total, results)

    def _show_results_dialog(self, score: int, total: int, results: list):
        """Hiển thị hộp thoại kết quả chi tiết."""
        percentage = int(score / total * 100) if total > 0 else 0

        # Đánh giá xếp loại
        if percentage == 100:
            grade = "🏆 XUẤT SẮC! Hoàn hảo tuyệt đối!"
            grade_color = C_SUCCESS
        elif percentage >= 75:
            grade = "🌟 Tốt! Nắm vững kiến thức rồi!"
            grade_color = C_PRIMARY
        elif percentage >= 50:
            grade = "📚 Khá! Ôn lại phần sai nhé!"
            grade_color = C_WARNING
        else:
            grade = "💪 Cần cố gắng thêm! Đọc lại bài học!"
            grade_color = C_ACCENT

        # Xây dựng HTML kết quả
        result_lines = ""
        for r in results:
            icon = "✅" if r["correct"] else "❌"
            bg = "#F6FFED" if r["correct"] else "#FFF1F0"
            border = C_SUCCESS if r["correct"] else C_ACCENT
            expl = r.get("explanation", "").replace("\n", "<br>")
            result_lines += f"""
<div style="
  border:1px solid {border}; border-radius:8px;
  padding:12px 16px; margin-bottom:12px; background:{bg};
">
  <div style="font-size:14px;font-weight:bold;margin-bottom:6px;">
    {icon} Câu {r['num']}: {r['question']}
  </div>
  <div style="font-size:13px;color:#555;margin-bottom:8px;">
    Bạn chọn: <strong>{r.get('chosen', '')}</strong>
  </div>
  <div style="
    background:white; border-radius:6px;
    padding:10px 14px; font-size:13px; color:#333;
    border-left:4px solid {border};
  ">{expl}</div>
</div>
"""

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  body {{
    font-family:'Segoe UI','Microsoft YaHei',sans-serif;
    font-size:14px; color:{C_TEXT_DARK}; line-height:1.6;
  }}
  .score-box {{
    background:linear-gradient(135deg,{C_PRIMARY}22,{C_TEAL}22);
    border:2px solid {C_PRIMARY}; border-radius:12px;
    padding:16px 20px; margin-bottom:18px; text-align:center;
  }}
</style></head><body>
<div class="score-box">
  <div style="font-size:40px; font-weight:bold; color:{grade_color};">
    {score}/{total}
  </div>
  <div style="font-size:22px; color:{grade_color}; margin:6px 0;">{grade}</div>
  <div style="font-size:14px; color:{C_TEXT_MID};">
    Điểm số: <strong>{percentage}%</strong>
  </div>
</div>
<div style="font-size:16px;font-weight:bold;color:{C_PRIMARY};margin-bottom:12px;">
  📋 Giải Thích Chi Tiết Từng Câu:
</div>
{result_lines}
</body></html>"""

        # Tạo dialog tùy chỉnh
        dialog = QMessageBox(self)
        dialog.setWindowTitle("📊 Kết Quả Kiểm Tra")
        dialog.setTextFormat(Qt.RichText)

        # Dùng QTextBrowser trong dialog để hiển thị HTML đẹp
        result_browser = QTextBrowser()
        result_browser.setHtml(html)
        result_browser.setMinimumSize(640, 500)
        result_browser.setStyleSheet(f"""
            QTextBrowser {{
                border: none;
                background: white;
                font-size: 14px;
            }}
        """)

        # Tạo QDialog tùy chỉnh thay vì QMessageBox (để hiển thị HTML phức tạp)
        from PyQt5.QtWidgets import QDialog, QDialogButtonBox
        custom_dialog = QDialog(self)
        custom_dialog.setWindowTitle("📊 Kết Quả Kiểm Tra – Giáo Viên Chấm Điểm")
        custom_dialog.setMinimumSize(680, 580)
        custom_dialog.setStyleSheet("background: white;")

        d_layout = QVBoxLayout(custom_dialog)
        d_layout.setContentsMargins(0, 0, 0, 12)
        d_layout.addWidget(result_browser)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok)
        btn_box.button(QDialogButtonBox.Ok).setText("✔️  Đóng & Học Tiếp")
        btn_box.button(QDialogButtonBox.Ok).setStyleSheet(f"""
            QPushButton {{
                background: {C_PRIMARY}; color: white;
                border-radius: 8px; padding: 8px 24px;
                font-size: 14px; font-weight: bold; border: none;
            }}
            QPushButton:hover {{ background: {C_PRIMARY_DARK}; }}
        """)
        btn_box.accepted.connect(custom_dialog.accept)
        d_layout.addWidget(btn_box, 0, Qt.AlignCenter)

        custom_dialog.exec_()

    def _reload_data(self):
        """Tải lại dữ liệu từ file JSON."""
        self.data_manager.data = self.data_manager._load()
        self._load_lesson_list()
        QMessageBox.information(
            self, "✅ Đã Tải Lại",
            f"Dữ liệu từ <b>{DATA_FILE}</b> đã được tải lại thành công!\n"
            f"Số bài học: {len(self.data_manager.get_lessons())}"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  CỬA SỔ CHÍNH (MainWindow)
# ═══════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    """Cửa sổ ứng dụng chính."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("📖 Học Tiếng Trung Từ Con Số 0 – Gia Sư 1-1 Riêng Tư")
        self.setMinimumSize(1060, 720)
        self.resize(1200, 800)

        # Tìm file dữ liệu trong cùng thư mục với script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, DATA_FILE)

        # Khởi tạo DataManager (tự tạo file nếu chưa có)
        self.data_manager = DataManager(data_path)

        self._setup_style()
        self._setup_ui()

    def _setup_style(self):
        """Thiết lập stylesheet toàn ứng dụng."""
        self.setStyleSheet(f"""
            QMainWindow {{
                background: {C_BG_MAIN};
            }}
            QTabWidget::pane {{
                border: none;
                background: {C_BG_MAIN};
            }}
            QTabWidget::tab-bar {{
                alignment: left;
            }}
            QTabBar::tab {{
                background: #E8EAF0;
                color: {C_TEXT_MID};
                padding: 12px 22px;
                font-size: 14px;
                font-weight: 600;
                border: none;
                border-bottom: 3px solid transparent;
                margin-right: 2px;
            }}
            QTabBar::tab:selected {{
                background: white;
                color: {C_PRIMARY};
                border-bottom: 3px solid {C_PRIMARY};
            }}
            QTabBar::tab:hover:!selected {{
                background: #DDE2F0;
                color: {C_PRIMARY};
            }}
            QScrollBar:vertical {{
                border: none;
                background: {C_BG_MAIN};
                width: 8px;
                margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: {C_PRIMARY}60;
                border-radius: 4px;
                min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {C_PRIMARY};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
            QScrollBar:horizontal {{
                height: 8px;
                background: {C_BG_MAIN};
                border: none;
            }}
            QScrollBar::handle:horizontal {{
                background: {C_PRIMARY}60;
                border-radius: 4px;
            }}
            QToolTip {{
                background: {C_TEXT_DARK};
                color: white;
                border: none;
                padding: 6px 10px;
                border-radius: 6px;
                font-size: 13px;
            }}
        """)

    def _setup_ui(self):
        """Xây dựng giao diện chính."""
        # Header banner
        header = QWidget()
        header.setStyleSheet(f"""
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 {C_PRIMARY}, stop:0.5 #0095E8, stop:1 {C_TEAL});
            min-height: 58px;
            max-height: 58px;
        """)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(20, 0, 20, 0)

        title_lbl = QLabel("🀄 Học Tiếng Trung Từ Con Số 0")
        title_lbl.setStyleSheet("""
            color: white;
            font-size: 20px;
            font-weight: bold;
            font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
        """)

        subtitle_lbl = QLabel("Gia Sư Riêng 1-1 · Cực Kỳ Chậm · Cực Kỳ Chi Tiết")
        subtitle_lbl.setStyleSheet("""
            color: rgba(255,255,255,0.85);
            font-size: 13px;
            font-family: 'Segoe UI', sans-serif;
        """)

        h_layout.addWidget(title_lbl)
        h_layout.addStretch()
        h_layout.addWidget(subtitle_lbl)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        tab1 = FoundationTab()
        tab2 = RoadmapTab()
        tab3 = LessonTab(self.data_manager)

        self.tabs.addTab(tab1, "☘️  Nền Tảng Cho Người Mới")
        self.tabs.addTab(tab2, "🗺️  Lộ Trình Học Đường Dài")
        self.tabs.addTab(tab3, "📖  Bài Học Chi Tiết")

        # Ghép layout tổng
        central = QWidget()
        central.setStyleSheet(f"background: {C_BG_MAIN};")
        c_layout = QVBoxLayout(central)
        c_layout.setContentsMargins(0, 0, 0, 0)
        c_layout.setSpacing(0)
        c_layout.addWidget(header)
        c_layout.addWidget(self.tabs)

        # Thanh trạng thái
        self.statusBar().setStyleSheet(f"""
            QStatusBar {{
                background: {C_BG_SIDEBAR};
                border-top: 1px solid {C_BORDER};
                color: {C_TEXT_LIGHT};
                font-size: 12px;
                padding: 3px 12px;
            }}
        """)
        lesson_count = len(self.data_manager.get_lessons())
        self.statusBar().showMessage(
            f"✅ Đã tải {lesson_count} bài học từ {DATA_FILE}  |  "
            f"Phiên bản 1.0.0  |  Dữ liệu có thể chỉnh sửa trong file lessons_data.json"
        )

        self.setCentralWidget(central)

        # Mặc định mở Tab 3 (Bài học)
        self.tabs.setCurrentIndex(2)


# ═══════════════════════════════════════════════════════════════════════════
#  ĐIỂM KHỞI CHẠY
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # Kích hoạt DPI cao cho màn hình Retina/4K
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("Học Tiếng Trung Từ Con Số 0")
    app.setOrganizationName("GiaSuTiengTrung")

    # Font mặc định hỗ trợ chữ Hán
    default_font = QFont()
    default_font.setFamily("Segoe UI")
    default_font.setPointSize(10)
    app.setFont(default_font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()