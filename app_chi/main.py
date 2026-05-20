# =========================================================
# APP HỌC TIẾNG TRUNG TỪ CON SỐ 0 - PYQT5
# File: main.py
#
# Cài:
#   pip install PyQt5
#
# Chạy:
#   python main.py
# =========================================================

import sys
import os
import json

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextBrowser,
    QListWidget,
    QPushButton,
    QMessageBox,
    QTabWidget,
    QScrollArea,
    QFrame,
    QGroupBox,
    QRadioButton,
    QButtonGroup
)

# =========================================================
# FILE JSON
# =========================================================

JSON_FILE = "lessons_data.json"


# =========================================================
# TẠO FILE JSON MẪU
# =========================================================

def create_sample_json():

    sample_data = {
        "roadmap": [
            {
                "stage_id": 1,
                "title": "Pinyin",
                "description": "Học phát âm, thanh điệu, khẩu hình.",
                "details": [
                    "Học 4 thanh điệu",
                    "Luyện khẩu hình",
                    "Nghe phát âm chuẩn",
                    "Phân biệt bật hơi"
                ]
            },
            {
                "stage_id": 2,
                "title": "Ghép âm",
                "description": "Ghép thanh mẫu và vận mẫu.",
                "details": [
                    "b + a = ba",
                    "zh + ang = zhang",
                    "Luyện phản xạ ghép âm"
                ]
            },
            {
                "stage_id": 3,
                "title": "Từ vựng",
                "description": "Học từ giao tiếp cơ bản.",
                "details": [
                    "Chào hỏi",
                    "Gia đình",
                    "Ăn uống",
                    "Số đếm"
                ]
            }
        ],

        "pinyin": {

            "initials": [

                {
                    "hanzi": "b",
                    "name": "Thanh mẫu b",
                    "vietnamese_reading": "p nhẹ",
                    "mouth_shape": "Hai môi khép rồi bật nhẹ.",
                    "warning": "Không bật hơi mạnh."
                },

                {
                    "hanzi": "p",
                    "name": "Thanh mẫu p",
                    "vietnamese_reading": "phờ bật hơi",
                    "mouth_shape": "Bật hơi mạnh ra ngoài.",
                    "warning": "Đặt tay trước miệng sẽ thấy hơi."
                },

                {
                    "hanzi": "m",
                    "name": "Thanh mẫu m",
                    "vietnamese_reading": "mờ",
                    "mouth_shape": "Khép môi nhẹ.",
                    "warning": "Âm rung mũi."
                },

                {
                    "hanzi": "h",
                    "name": "Thanh mẫu h",
                    "vietnamese_reading": "h nhẹ cổ họng",
                    "mouth_shape": "Đẩy hơi từ cổ họng.",
                    "warning": "Không đọc quá nặng."
                }
            ],

            "finals": [

                {
                    "hanzi": "a",
                    "name": "Vận mẫu a",
                    "vietnamese_reading": "a",
                    "mouth_shape": "Mở miệng rộng.",
                    "warning": "Âm vang."
                },

                {
                    "hanzi": "i",
                    "name": "Vận mẫu i",
                    "vietnamese_reading": "i",
                    "mouth_shape": "Kéo ngang môi.",
                    "warning": "Không đọc quá ngắn."
                },

                {
                    "hanzi": "ao",
                    "name": "Vận mẫu ao",
                    "vietnamese_reading": "ao",
                    "mouth_shape": "Mở rộng rồi tròn môi.",
                    "warning": "Không đọc thành au."
                },

                {
                    "hanzi": "ü",
                    "name": "Vận mẫu ü",
                    "vietnamese_reading": "uy chu môi",
                    "mouth_shape": "Chu môi tròn.",
                    "warning": "Người Việt hay đọc sai."
                }
            ]
        },

        "lessons": [

            {
                "id": 1,

                "title": "BÀI 1: XIN CHÀO! (你好)",

                "stage": "Giai đoạn 3 - Từ vựng nền tảng",

                "vocabulary": [

                    {
                        "hanzi": "你",
                        "pinyin": "nǐ",
                        "tone": "Thanh 3 ↘↗",
                        "vietnamese_reading": "nỉ",
                        "meaning": "Bạn"
                    },

                    {
                        "hanzi": "好",
                        "pinyin": "hǎo",
                        "tone": "Thanh 3 ↘↗",
                        "vietnamese_reading": "hảo",
                        "meaning": "Tốt"
                    },

                    {
                        "hanzi": "你好",
                        "pinyin": "nǐ hǎo",
                        "tone": "Thanh 3 + Thanh 3",
                        "vietnamese_reading": "nỉ hảo",
                        "meaning": "Xin chào"
                    }
                ],

                "phonetic_analysis": """
<h3>PHÂN TÍCH ÂM</h3>

<b>nǐ</b><br>
- n = đầu lưỡi chạm nướu trên<br>
- i = âm i kéo dài<br>
- Thanh 3: xuống rồi lên ↘↗<br><br>

<b>hǎo</b><br>
- h = hơi nhẹ từ cổ họng<br>
- ao = ao nhưng tròn môi hơn
""",

                "mouth_shape": """
<h3>KHẨU HÌNH</h3>

- Miệng mở tự nhiên.<br>
- Âm h dùng cổ họng nhẹ.<br>
- Thanh 3 phải hạ xuống rồi kéo lên.
""",

                "vietnamese_errors": """
<h3>NGƯỜI VIỆT HAY SAI</h3>

❌ Sai: Ní hào<br>
✅ Đúng: Nỉ hảo<br><br>

❌ Sai: đọc h quá nặng<br>
✅ Đúng: bật hơi nhẹ
""",

                "memory_tricks": """
<h3>MẸO GHI NHỚ</h3>

你 = bạn<br><br>

好 gồm:
- 女 = con gái
- 子 = đứa trẻ

→ ý nghĩa tốt đẹp.
""",

                "practical_examples": [

                    {
                        "hanzi": "你好！",
                        "pinyin": "Nǐ hǎo!",
                        "meaning": "Xin chào!",
                        "special_note": "Hai thanh 3 đứng cạnh nhau."
                    },

                    {
                        "hanzi": "你好吗？",
                        "pinyin": "Nǐ hǎo ma?",
                        "meaning": "Bạn khỏe không?",
                        "special_note": "吗 đọc nhẹ."
                    }
                ],

                "reading_practice": {

                    "slow": "nǐ ... hǎo ...",

                    "tone": "nǐ hǎo ↘↗ ↘↗",

                    "natural": "Ní hảo!"
                },

                "mini_quiz": [

                    {
                        "question": "你好 nghĩa là gì?",

                        "options": [
                            "Xin chào",
                            "Tạm biệt",
                            "Cảm ơn",
                            "Xin lỗi"
                        ],

                        "correct_answer": "Xin chào",

                        "explanation": "你好 = xin chào."
                    },

                    {
                        "question": "好 mang thanh mấy?",

                        "options": [
                            "Thanh 1",
                            "Thanh 2",
                            "Thanh 3",
                            "Thanh 4"
                        ],

                        "correct_answer": "Thanh 3",

                        "explanation": "hǎo là thanh 3 ↘↗."
                    }
                ]
            }
        ]
    }

    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=4)


# =========================================================
# LOAD JSON
# =========================================================

def load_data():

    if not os.path.exists(JSON_FILE):
        create_sample_json()

    try:
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception as e:

        QMessageBox.critical(
            None,
            "Lỗi JSON",
            str(e)
        )

        return {}


# =========================================================
# APP
# =========================================================

class ChineseApp(QMainWindow):

    def __init__(self):
        super().__init__()

        self.data = load_data()

        self.lessons = self.data.get("lessons", [])

        self.current_quiz = []

        self.setWindowTitle("☘️ Học Tiếng Trung Từ Con Số 0")

        self.resize(1500, 950)

        self.setup_ui()

    # =====================================================
    # UI
    # =====================================================

    def setup_ui(self):

        tabs = QTabWidget()

        tabs.setStyleSheet("""
            QTabBar::tab {
                background:#EAF2FF;
                padding:12px;
                margin:4px;
                border-radius:8px;
                font-size:15px;
                font-weight:bold;
            }

            QTabBar::tab:selected {
                background:#0078D7;
                color:white;
            }
        """)

        tabs.addTab(
            self.create_foundation_tab(),
            "☘️ Nền Tảng"
        )

        tabs.addTab(
            self.create_pinyin_tab(),
            "🔤 Thanh Mẫu & Vận Mẫu"
        )

        tabs.addTab(
            self.create_roadmap_tab(),
            "🗺️ Lộ Trình"
        )

        tabs.addTab(
            self.create_lesson_tab(),
            "📖 Bài Học"
        )

        self.setCentralWidget(tabs)

    # =====================================================
    # TAB FOUNDATION
    # =====================================================

    def create_foundation_tab(self):

        widget = QWidget()

        layout = QVBoxLayout()

        browser = QTextBrowser()

        html = """
        <div style='padding:20px;font-family:Segoe UI;'>

        <h1 style='color:#0078D7;'>
        Pinyin là gì?
        </h1>

        <p style='font-size:16px;line-height:1.9;'>

        Pinyin là hệ thống phiên âm giúp người nước ngoài đọc được tiếng Trung.

        </p>

        <h2 style='color:#FF4D4F;'>
        4 Thanh Điệu
        </h2>

        <table border='1' cellpadding='10'
        cellspacing='0'
        style='border-collapse:collapse;
               width:100%;
               font-size:16px;'>

        <tr style='background:#0078D7;color:white;'>

            <th>Thanh</th>
            <th>Ký hiệu</th>
            <th>Mô tả</th>

        </tr>

        <tr>
            <td>Thanh 1</td>
            <td>¯</td>
            <td>Giọng ngang ngang</td>
        </tr>

        <tr>
            <td>Thanh 2</td>
            <td>ˊ</td>
            <td>Đi lên như hỏi</td>
        </tr>

        <tr>
            <td>Thanh 3</td>
            <td>ˇ</td>
            <td>Xuống rồi lên ↘↗</td>
        </tr>

        <tr>
            <td>Thanh 4</td>
            <td>ˋ</td>
            <td>Dứt mạnh xuống</td>
        </tr>

        </table>

        </div>
        """

        browser.setHtml(html)

        layout.addWidget(browser)

        widget.setLayout(layout)

        return widget

    # =====================================================
    # TAB PINYIN
    # =====================================================

    def create_pinyin_tab(self):

        widget = QWidget()

        layout = QVBoxLayout()

        browser = QTextBrowser()

        pinyin_data = self.data.get("pinyin", {})

        initials = pinyin_data.get("initials", [])

        finals = pinyin_data.get("finals", [])

        html = """
        <div style='font-family:Segoe UI;padding:20px;'>

        <h1 style='color:#0078D7;'>
        🔤 THANH MẪU
        </h1>
        """

        # =============================================
        # INITIALS
        # =============================================

        for item in initials:

            html += f"""
            <div style='background:#F5F7FA;
                        padding:18px;
                        margin-bottom:15px;
                        border-radius:12px;
                        border:1px solid #D9E2EC;'>

                <div style='font-size:42px;
                            font-weight:bold;
                            color:#0078D7;'>

                    {item.get('hanzi', '')}

                </div>

                <div style='margin-top:10px;font-size:18px;'>

                    {item.get('name', '')}

                </div>

                <div style='margin-top:10px;'>

                    <b>Đọc gần giống:</b>
                    {item.get('vietnamese_reading', '')}

                </div>

                <div style='margin-top:10px;'>

                    <b>Khẩu hình:</b>
                    {item.get('mouth_shape', '')}

                </div>

                <div style='margin-top:10px;color:#FF4D4F;'>

                    ⚠️ {item.get('warning', '')}

                </div>

            </div>
            """

        # =============================================
        # FINALS
        # =============================================

        html += """
        <h1 style='color:#0078D7;margin-top:30px;'>
        🔠 VẬN MẪU
        </h1>
        """

        for item in finals:

            html += f"""
            <div style='background:#FFF7E6;
                        padding:18px;
                        margin-bottom:15px;
                        border-radius:12px;
                        border:1px solid #D9E2EC;'>

                <div style='font-size:42px;
                            font-weight:bold;
                            color:#FF4D4F;'>

                    {item.get('hanzi', '')}

                </div>

                <div style='margin-top:10px;font-size:18px;'>

                    {item.get('name', '')}

                </div>

                <div style='margin-top:10px;'>

                    <b>Đọc gần giống:</b>
                    {item.get('vietnamese_reading', '')}

                </div>

                <div style='margin-top:10px;'>

                    <b>Khẩu hình:</b>
                    {item.get('mouth_shape', '')}

                </div>

                <div style='margin-top:10px;color:#FF4D4F;'>

                    ⚠️ {item.get('warning', '')}

                </div>

            </div>
            """

        html += "</div>"

        browser.setHtml(html)

        layout.addWidget(browser)

        widget.setLayout(layout)

        return widget

    # =====================================================
    # TAB ROADMAP
    # =====================================================

    def create_roadmap_tab(self):

        widget = QWidget()

        layout = QVBoxLayout()

        browser = QTextBrowser()

        roadmap_data = self.data.get("roadmap", [])

        html = """
        <div style='padding:20px;font-family:Segoe UI;'>

        <h1 style='color:#0078D7;'>
        🗺️ LỘ TRÌNH HỌC
        </h1>
        """

        for item in roadmap_data:

            html += f"""
            <div style='background:white;
                        padding:18px;
                        margin-bottom:15px;
                        border-radius:12px;
                        border:1px solid #D9E2EC;'>

                <h2 style='color:#0078D7;'>

                    Giai đoạn {item.get('stage_id', '')}
                    :
                    {item.get('title', '')}

                </h2>

                <p style='font-size:16px;'>

                    {item.get('description', '')}

                </p>

                <ul>
            """

            for detail in item.get("details", []):

                html += f"<li>{detail}</li>"

            html += """
                </ul>
            </div>
            """

        html += "</div>"

        browser.setHtml(html)

        layout.addWidget(browser)

        widget.setLayout(layout)

        return widget

    # =====================================================
    # TAB LESSON
    # =====================================================

    def create_lesson_tab(self):

        widget = QWidget()

        main_layout = QHBoxLayout()

        # =============================================
        # LIST LESSON
        # =============================================

        self.lesson_list = QListWidget()

        self.lesson_list.setMaximumWidth(320)

        self.lesson_list.setStyleSheet("""
            QListWidget {
                background:white;
                font-size:16px;
                border:1px solid #D9E2EC;
            }

            QListWidget::item {
                padding:12px;
                margin:4px;
            }

            QListWidget::item:selected {
                background:#0078D7;
                color:white;
                border-radius:8px;
            }
        """)

        for lesson in self.lessons:
            self.lesson_list.addItem(
                lesson.get("title", "Không có tiêu đề")
            )

        self.lesson_list.currentRowChanged.connect(
            self.display_lesson
        )

        # =============================================
        # RIGHT SIDE
        # =============================================

        right_layout = QVBoxLayout()

        self.lesson_browser = QTextBrowser()

        self.lesson_browser.setStyleSheet("""
            QTextBrowser {
                background:white;
                border:1px solid #D9E2EC;
                padding:20px;
                font-size:16px;
            }
        """)

        scroll = QScrollArea()

        scroll.setWidgetResizable(True)

        scroll.setWidget(self.lesson_browser)

        right_layout.addWidget(scroll)

        # =============================================
        # QUIZ AREA
        # =============================================

        self.quiz_layout = QVBoxLayout()

        quiz_frame = QFrame()

        quiz_frame.setLayout(self.quiz_layout)

        quiz_scroll = QScrollArea()

        quiz_scroll.setWidgetResizable(True)

        quiz_scroll.setWidget(quiz_frame)

        quiz_scroll.setMinimumHeight(300)

        right_layout.addWidget(quiz_scroll)

        submit_btn = QPushButton(
            "📨 Nộp bài cho giáo viên 1-1"
        )

        submit_btn.setStyleSheet("""
            QPushButton {
                background:#FF4D4F;
                color:white;
                padding:14px;
                font-size:16px;
                border:none;
                border-radius:10px;
                font-weight:bold;
            }

            QPushButton:hover {
                background:#ff7875;
            }
        """)

        submit_btn.clicked.connect(
            self.submit_quiz
        )

        right_layout.addWidget(submit_btn)

        main_layout.addWidget(self.lesson_list)

        main_layout.addLayout(right_layout)

        widget.setLayout(main_layout)

        if self.lessons:
            self.lesson_list.setCurrentRow(0)

        return widget

    # =====================================================
    # DISPLAY LESSON
    # =====================================================

    def display_lesson(self, index):

        if index < 0 or index >= len(self.lessons):
            return

        lesson = self.lessons[index]

        html = f"""
        <div style='font-family:Segoe UI;'>

        <h1 style='color:#0078D7;'>
        {lesson.get('title', '')}
        </h1>

        <h3 style='color:#FF4D4F;'>
        {lesson.get('stage', '')}
        </h3>

        <hr>

        <h2 style='color:#0078D7;'>
        📚 TỪ VỰNG
        </h2>

        <table border='1'
        cellpadding='10'
        cellspacing='0'
        style='border-collapse:collapse;
               width:100%;
               font-size:16px;'>

        <tr style='background:#0078D7;color:white;'>

            <th>Hán tự</th>
            <th>Pinyin</th>
            <th>Thanh</th>
            <th>Đọc bồi</th>
            <th>Nghĩa</th>

        </tr>
        """

        # =============================================
        # VOCAB
        # =============================================

        for word in lesson.get("vocabulary", []):

            html += f"""
            <tr>

                <td style='font-size:28px;font-weight:bold;'>
                    {word.get('hanzi', '')}
                </td>

                <td>{word.get('pinyin', '')}</td>

                <td>{word.get('tone', '')}</td>

                <td>{word.get('vietnamese_reading', '')}</td>

                <td>{word.get('meaning', '')}</td>

            </tr>
            """

        html += "</table><br>"

        # =============================================
        # CONTENT
        # =============================================

        html += f"""
        <div style='background:#F5F7FA;
                    padding:15px;
                    border-radius:12px;'>

            {lesson.get('phonetic_analysis', '')}

        </div>

        <br>

        <div style='background:#FFF7E6;
                    padding:15px;
                    border-radius:12px;'>

            {lesson.get('mouth_shape', '')}

        </div>

        <br>

        <div style='background:#FFF1F0;
                    padding:15px;
                    border-radius:12px;'>

            {lesson.get('vietnamese_errors', '')}

        </div>

        <br>

        <div style='background:#F6FFED;
                    padding:15px;
                    border-radius:12px;'>

            {lesson.get('memory_tricks', '')}

        </div>
        """

        # =============================================
        # EXAMPLES
        # =============================================

        html += """
        <br>

        <h2 style='color:#0078D7;'>
        💬 VÍ DỤ THỰC TẾ
        </h2>
        """

        for ex in lesson.get("practical_examples", []):

            html += f"""
            <div style='border:1px solid #D9E2EC;
                        padding:15px;
                        margin-bottom:12px;
                        border-radius:10px;'>

                <div style='font-size:28px;font-weight:bold;'>

                    {ex.get('hanzi', '')}

                </div>

                <div style='color:#0078D7;
                            font-size:18px;'>

                    {ex.get('pinyin', '')}

                </div>

                <div style='margin-top:10px;'>

                    Nghĩa:
                    {ex.get('meaning', '')}

                </div>

                <div style='margin-top:10px;
                            color:#FF4D4F;'>

                    ⚠️
                    {ex.get('special_note', '')}

                </div>

            </div>
            """

        # =============================================
        # READING
        # =============================================

        practice = lesson.get(
            "reading_practice",
            {}
        )

        slow = practice.get(
            "slow",
            "Chưa có dữ liệu."
        )

        tone = practice.get(
            "tone",
            "Chưa có dữ liệu."
        )

        natural = practice.get(
            "natural",
            "Chưa có dữ liệu."
        )

        html += f"""
        <h2 style='color:#0078D7;'>
        🎧 LUYỆN ĐỌC
        </h2>

        <div style='background:#EAF2FF;
                    padding:18px;
                    border-radius:12px;
                    line-height:2;'>

            <b>1️⃣ Đọc chậm:</b><br>
            {slow}<br><br>

            <b>2️⃣ Đọc thanh điệu:</b><br>
            {tone}<br><br>

            <b>3️⃣ Đọc tự nhiên:</b><br>
            {natural}

        </div>
        """

        self.lesson_browser.setHtml(html)

        self.load_quiz(
            lesson.get("mini_quiz", [])
        )

    # =====================================================
    # LOAD QUIZ
    # =====================================================

    def load_quiz(self, quiz_data):

        self.current_quiz = []

        while self.quiz_layout.count():

            item = self.quiz_layout.takeAt(0)

            widget = item.widget()

            if widget:
                widget.deleteLater()

        for idx, q in enumerate(quiz_data):

            group_box = QGroupBox(
                f"Câu {idx+1}: {q.get('question', '')}"
            )

            vbox = QVBoxLayout()

            button_group = QButtonGroup()

            for option in q.get("options", []):

                radio = QRadioButton(option)

                vbox.addWidget(radio)

                button_group.addButton(radio)

            group_box.setLayout(vbox)

            self.quiz_layout.addWidget(group_box)

            self.current_quiz.append({

                "data": q,

                "group": button_group
            })

    # =====================================================
    # SUBMIT QUIZ
    # =====================================================

    def submit_quiz(self):

        total = len(self.current_quiz)

        score = 0

        result_html = ""

        for idx, item in enumerate(self.current_quiz):

            data = item["data"]

            group = item["group"]

            selected = group.checkedButton()

            if not selected:

                result_html += f"""
                ⚠️ Câu {idx+1}: Chưa chọn đáp án.<br><br>
                """

                continue

            answer = selected.text()

            correct = data.get(
                "correct_answer",
                ""
            )

            if answer == correct:

                score += 1

                result_html += f"""
                ✅ Câu {idx+1}: Chính xác!<br>
                🧠 {data.get('explanation', '')}<br><br>
                """

            else:

                result_html += f"""
                ❌ Câu {idx+1}: Sai rồi.<br>
                👉 Đáp án đúng:
                {correct}<br>

                🧠 {data.get('explanation', '')}<br><br>
                """

        msg = QMessageBox(self)

        msg.setWindowTitle(
            "🎓 Giáo viên 1-1 chấm bài"
        )

        msg.setTextFormat(Qt.RichText)

        msg.setText(f"""
        <h2>KẾT QUẢ</h2>

        <p style='font-size:18px;'>
        Điểm:
        <b style='color:#0078D7;'>
        {score}/{total}
        </b>
        </p>

        <hr>

        {result_html}
        """)

        msg.exec_()


# =========================================================
# MAIN
# =========================================================

def main():

    app = QApplication(sys.argv)

    app.setFont(
        QFont("Segoe UI", 11)
    )

    app.setStyleSheet("""
        QMainWindow {
            background:#F5F7FA;
        }

        QWidget {
            background:#F5F7FA;
        }
    """)

    window = ChineseApp()

    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()