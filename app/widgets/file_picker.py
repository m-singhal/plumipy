from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLineEdit,
    QPushButton, QLabel, QFileDialog
)
from PyQt6.QtCore import pyqtSignal


class FilePicker(QWidget):
    """Browse button + path display + optional hint label."""

    path_changed = pyqtSignal(str)

    def __init__(
        self,
        placeholder="Not selected",
        hint="",
        file_filter="All Files (*)",
        optional=False,
        parent=None,
    ):
        super().__init__(parent)
        self._filter = file_filter
        self._optional = optional

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        row = QHBoxLayout()
        row.setSpacing(8)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(placeholder)
        self.path_edit.textChanged.connect(self.path_changed)
        row.addWidget(self.path_edit)

        browse = QPushButton("Browse")
        browse.setObjectName("browse_btn")
        browse.setFixedWidth(80)
        browse.clicked.connect(self._browse)
        row.addWidget(browse)

        if optional:
            opt_label = QLabel("optional")
            opt_label.setObjectName("hint_label")
            row.addWidget(opt_label)

        layout.addLayout(row)

        if hint:
            h = QLabel(hint)
            h.setObjectName("hint_label")
            h.setWordWrap(True)
            layout.addWidget(h)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select file", "", self._filter,
            options=QFileDialog.Option.DontUseNativeDialog
        )
        if path:
            self.path_edit.setText(path)

    def path(self) -> str:
        return self.path_edit.text().strip()

    def set_path(self, p: str):
        self.path_edit.setText(p)

    def clear(self):
        self.path_edit.clear()
