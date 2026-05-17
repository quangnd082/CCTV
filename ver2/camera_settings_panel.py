from typing import List

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGridLayout,
    QHBoxLayout,
    QCheckBox,
    QLabel,
    QGroupBox,
    QFrame,
    QPushButton,
)


class CameraSettingsPanel(QWidget):
    """Chọn camera — chỉ áp dụng khi bấm Xác nhận (tránh xung đột luồng RTSP)."""

    apply_requested = pyqtSignal(list)

    def __init__(self, camera_infos, max_active: int = 4, parent=None, logger=None):
        super().__init__(parent)
        self._camera_infos = list(camera_infos)
        self._max_active = max(1, int(max_active))
        self.logger = logger
        self._checkboxes: List[QCheckBox] = []
        self._updating = False
        self._build_ui()
        self._load_initial_selection()
        self._refresh_lock_state()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = QLabel("Chọn camera hiển thị trên màn hình Monitor")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #e8e8e8;")
        root.addWidget(title)

        hint = QLabel(
            f"Khi vào tab này, tất cả luồng camera trên Monitor sẽ được dừng.\n"
            f"Chọn tối đa {self._max_active} camera, sau đó bấm «Xác nhận» để mở lại luồng."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #a8c8e8; font-size: 12px;")
        root.addWidget(hint)

        self.label_counter = QLabel()
        self.label_counter.setStyleSheet("color: #7fd3ff; font-size: 13px; font-weight: bold;")
        root.addWidget(self.label_counter)

        group = QGroupBox(f"Danh sách camera ({len(self._camera_infos)})")
        group.setStyleSheet(
            "QGroupBox { color: #ddd; font-weight: bold; border: 1px solid #444; "
            "border-radius: 6px; margin-top: 8px; padding-top: 12px; }"
        )
        grid = QGridLayout(group)
        grid.setSpacing(8)

        for i, ci in enumerate(self._camera_infos):
            row = i // 2
            col = (i % 2) * 3

            frame = QFrame()
            frame.setStyleSheet(
                "QFrame { background-color: #1a1a1a; border: 1px solid #333; border-radius: 6px; }"
            )
            cell = QGridLayout(frame)
            cell.setContentsMargins(10, 8, 10, 8)

            cb = QCheckBox("Use")
            cb.setStyleSheet("font-weight: bold; color: #7fd3ff;")
            cb.setProperty("camera_name", ci.camera_name)
            cb.stateChanged.connect(self._on_checkbox_changed)

            name_lbl = QLabel(ci.camera_name)
            name_lbl.setStyleSheet("font-size: 14px; font-weight: bold; color: #f0f0f0;")

            src = str(ci.camera_src)
            if len(src) > 56:
                src = src[:53] + "..."
            src_lbl = QLabel(src)
            src_lbl.setStyleSheet("font-size: 11px; color: #888;")
            src_lbl.setWordWrap(True)

            cell.addWidget(cb, 0, 0, 1, 1, Qt.AlignTop)
            cell.addWidget(name_lbl, 0, 1, 1, 2)
            cell.addWidget(src_lbl, 1, 1, 1, 2)

            self._checkboxes.append(cb)
            grid.addWidget(frame, row, col, 1, 3)

        root.addWidget(group)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_apply = QPushButton("Xác nhận và áp dụng")
        self.btn_apply.setMinimumHeight(40)
        self.btn_apply.setStyleSheet(
            "QPushButton { background-color: #1a6fb5; color: white; font-weight: bold; "
            "font-size: 14px; padding: 8px 24px; border-radius: 6px; }"
            "QPushButton:hover { background-color: #2288dd; }"
            "QPushButton:pressed { background-color: #0d4f82; }"
        )
        self.btn_apply.clicked.connect(self._on_apply_clicked)
        btn_row.addWidget(self.btn_apply)
        btn_row.addStretch(1)
        root.addLayout(btn_row)
        root.addStretch(1)

    def _load_initial_selection(self) -> None:
        self._updating = True
        started = 0
        for cb, ci in zip(self._checkboxes, self._camera_infos):
            flags = getattr(ci, "enable_flags", None) or {}
            use = int(flags.get("use_camera", 0)) == 1
            checked = use and started < self._max_active
            if checked:
                started += 1
            cb.setChecked(checked)
        self._updating = False

    def load_selection(self, camera_names: List[str]) -> None:
        """Đồng bộ checkbox theo bộ camera đang áp dụng (khi mở tab Settings)."""
        name_set = set(camera_names or [])
        self._updating = True
        for cb in self._checkboxes:
            cb.setChecked(cb.property("camera_name") in name_set)
        self._updating = False
        self._refresh_lock_state()

    def get_selected_names(self) -> List[str]:
        names = []
        for cb in self._checkboxes:
            if cb.isChecked():
                names.append(cb.property("camera_name"))
        return names

    def _selected_count(self) -> int:
        return sum(1 for cb in self._checkboxes if cb.isChecked())

    def _refresh_lock_state(self) -> None:
        n = self._selected_count()
        at_limit = n >= self._max_active
        for cb in self._checkboxes:
            if cb.isChecked():
                cb.setEnabled(True)
            else:
                cb.setEnabled(not at_limit)
        self.label_counter.setText(f"Đã chọn (chưa áp dụng): {n}/{self._max_active}")

    def _on_checkbox_changed(self, _state: int) -> None:
        if self._updating:
            return

        cb = self.sender()
        if cb is None or not isinstance(cb, QCheckBox):
            return

        if cb.isChecked() and self._selected_count() > self._max_active:
            self._updating = True
            cb.setChecked(False)
            self._updating = False

        self._refresh_lock_state()

    def _on_apply_clicked(self) -> None:
        selected = self.get_selected_names()
        if self.logger:
            self.logger.info(f"[CAM] apply requested: {selected}")
        self.apply_requested.emit(selected)
