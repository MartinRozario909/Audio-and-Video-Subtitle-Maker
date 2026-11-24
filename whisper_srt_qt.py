"""Audio and Video Subtitle Maker — PySide6 interface.

Same engine as whisper_srt.py (Whisper + optional NLLB / Hy-MT2 translation),
with a modern dark/light UI, drag-and-drop, live progress and Cancel.

Run:  python whisper_srt_qt.py   (or double-click run.bat)
"""

import html
import math
import os
import re
import subprocess
import sys
import threading

# Importing whisper_srt sets HF_HOME to <app>\models\hf before anything else.
from whisper_srt import (  # noqa: E402
    APP_DIR, MODELS_DIR, WHISPER_DIR, LANGUAGE_NAMES, NAME_TO_CODE, OUTPUT_FORMATS,
    AUTO, OUT_SAME, MODEL_AUTO, MODELS, TR_WHISPER, TRANSLATOR_CHOICES,
    DEFAULT_TRANSLATOR, DEFAULT_TRANSCRIBE_MODEL, DEFAULT_TRANSLATE_MODEL,
    TRANSLATORS, lang_name, free_gpu,
)

from PySide6.QtCore import (QEasingCurve, QObject, QPointF, QPropertyAnimation, QRectF, QSize,
                            Qt, QThread, Property, Signal, Slot)
from PySide6.QtGui import (QColor, QIcon, QLinearGradient, QPainter, QPainterPath, QPen,
                           QPixmap)
from PySide6.QtWidgets import (QAbstractButton, QApplication, QComboBox, QFileDialog, QFrame,
                               QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
                               QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
                               QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

MEDIA_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma",
              ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v"}
MEDIA_FILTER = ("Audio / Video (" + " ".join("*" + e for e in sorted(MEDIA_EXTS)) + ");;"
                "All files (*.*)")
ASSETS_DIR = os.path.join(APP_DIR, "assets")

# ---------------------------------------------------------------- theme
THEMES = {
    "dark": {
        "bg": "#121417", "card": "#1C1F24", "border": "#2A2E35", "border_strong": "#3A3F48",
        "text": "#E8EAED", "muted": "#9AA0A6", "input": "#24282E", "input_hover": "#2B3037",
        "input_soft": "#191C20", "log_bg": "#15171B",
        "accent": "#7C5CFF", "accent_hover": "#8E72FF", "accent_disabled": "#4A3D8C",
        "danger": "#E5484D", "danger_hover": "#EC5D62",
        "ok": "#3DD68C", "ok_bg": "rgba(61,214,140,0.14)", "bad_bg": "rgba(229,72,77,0.14)",
        "toggle_off": "#3A3F48",
    },
    "light": {
        "bg": "#EEF0F4", "card": "#FFFFFF", "border": "#E1E4EA", "border_strong": "#C9CED8",
        "text": "#1D2026", "muted": "#6B7280", "input": "#F6F7F9", "input_hover": "#EEF0F3",
        "input_soft": "#FAFBFC", "log_bg": "#F8F9FB",
        "accent": "#7C5CFF", "accent_hover": "#6A48F5", "accent_disabled": "#C5B8FF",
        "danger": "#E5484D", "danger_hover": "#D63C41",
        "ok": "#1F9D55", "ok_bg": "rgba(31,157,85,0.12)", "bad_bg": "rgba(229,72,77,0.10)",
        "toggle_off": "#C9CED8",
    },
}
T = dict(THEMES["dark"])  # current palette, read by custom-painted widgets


def chevron_svg(color: str) -> str:
    os.makedirs(ASSETS_DIR, exist_ok=True)
    path = os.path.join(ASSETS_DIR, f"chevron_{color.lstrip('#')}.svg")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write('<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" '
                    'viewBox="0 0 12 12"><path d="M2.5 4.5 L6 8 L9.5 4.5" fill="none" '
                    f'stroke="{color}" stroke-width="1.6" stroke-linecap="round" '
                    'stroke-linejoin="round"/></svg>')
    return path.replace("\\", "/")


QSS = """
QWidget { color: %(text)s; font-family: "Segoe UI Variable Text", "Segoe UI"; font-size: 10pt; }
QWidget#Root, QScrollArea#Scroll { background: %(bg)s; border: 0; }
QFrame#Card { background: %(card)s; border: 1px solid %(border)s; border-radius: 12px; }
QFrame#Card QLabel { background: transparent; }
QLabel#Title { font-size: 15pt; font-weight: 600; }
QLabel#CardTitle { font-size: 10.5pt; font-weight: 600; }
QLabel#Muted { color: %(muted)s; font-size: 9pt; }
QLabel#Field { color: %(text)s; }
QLabel#Arrow { color: %(muted)s; font-size: 13pt; padding: 0 4px; }
QLabel#Pill { background: %(input)s; border: 1px solid %(border)s; border-radius: 11px;
              padding: 3px 11px; font-size: 9pt; font-weight: 600; }
QFrame#DropZone { background: %(input_soft)s; border: 1.5px dashed %(border_strong)s;
                  border-radius: 10px; }
QFrame#DropZone[hover="true"] { border-color: %(accent)s; }
QLabel#FileBadge { border: 1.5px solid %(accent)s; border-radius: 6px; color: %(accent)s;
                   font-weight: 700; font-size: 8pt; }
QLabel#FileName { font-size: 13pt; }
QLabel#Duration { font-size: 13pt; color: %(muted)s; }
QComboBox, QLineEdit { background: %(input)s; border: 1px solid %(border)s; border-radius: 6px;
                       padding: 6px 10px; min-height: 20px; }
QComboBox:hover, QLineEdit:hover { border-color: %(border_strong)s; }
QComboBox:focus, QLineEdit:focus { border-color: %(accent)s; }
QComboBox:disabled { color: %(muted)s; }
QComboBox::drop-down { border: 0; width: 26px; }
QComboBox::down-arrow { image: url(%(chevron)s); width: 12px; height: 12px; }
QComboBox QAbstractItemView { background: %(card)s; border: 1px solid %(border)s; outline: 0;
                              selection-background-color: %(accent)s; selection-color: white;
                              padding: 4px; }
QPushButton { background: %(input)s; border: 1px solid %(border)s; border-radius: 6px;
              padding: 7px 16px; }
QPushButton:hover { background: %(input_hover)s; border-color: %(border_strong)s; }
QPushButton:disabled { color: %(muted)s; }
QPushButton#Primary { background: %(accent)s; border: 0; color: white; font-weight: 600;
                      padding: 9px 22px; }
QPushButton#Primary:hover { background: %(accent_hover)s; }
QPushButton#Primary:disabled { background: %(accent_disabled)s; color: rgba(255,255,255,0.7); }
QPushButton#Danger { background: %(danger)s; border: 0; color: white; font-weight: 600;
                     padding: 9px 22px; }
QPushButton#Danger:hover { background: %(danger_hover)s; }
QPushButton#Danger:disabled { color: rgba(255,255,255,0.7); }
QPushButton#LogHeader { background: transparent; border: 0; text-align: left;
                        font-weight: 600; font-size: 10.5pt; padding: 2px 0; }
QLabel#ChipOk { background: %(ok_bg)s; color: %(ok)s; border-radius: 4px; padding: 2px 8px;
                font-size: 9pt; }
QLabel#ChipBad { background: %(bad_bg)s; color: %(danger)s; border-radius: 4px;
                 padding: 2px 8px; font-size: 9pt; }
QLabel#ChipInfo { background: %(input)s; color: %(muted)s; border-radius: 4px;
                  padding: 2px 8px; font-size: 9pt; }
QLabel#Error { color: %(danger)s; font-size: 9pt; }
QProgressBar { background: %(input)s; border: 0; border-radius: 3px; min-height: 6px;
               max-height: 6px; }
QProgressBar::chunk { background: %(accent)s; border-radius: 3px; }
QPlainTextEdit#Log { background: %(log_bg)s; border: 1px solid %(border)s; border-radius: 8px;
                     font-family: "Cascadia Mono", Consolas; font-size: 9pt; padding: 6px; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: %(border_strong)s; border-radius: 3px;
                              min-height: 24px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QToolTip { background: %(card)s; color: %(text)s; border: 1px solid %(border)s; }
"""


def build_qss(t: dict) -> str:
    return QSS % dict(t, chevron=chevron_svg(t["muted"]))


def repolish(w: QWidget):
    w.style().unpolish(w)
    w.style().polish(w)
    w.update()


def blend(a: QColor, b: QColor, f: float) -> QColor:
    return QColor(round(a.red() + (b.red() - a.red()) * f),
                  round(a.green() + (b.green() - a.green()) * f),
                  round(a.blue() + (b.blue() - a.blue()) * f))


def waveform_pixmap(size: int = 36) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    grad = QLinearGradient(0, 0, size, 0)
    grad.setColorAt(0, QColor("#7C5CFF"))
    grad.setColorAt(1, QColor("#C29BFF"))
    heights = [0.25, 0.55, 0.85, 0.45, 1.0, 0.6, 0.9, 0.4, 0.2]
    n = len(heights)
    bw = size / (n * 1.9)
    gap = (size - n * bw) / (n - 1)
    p.setPen(Qt.NoPen)
    p.setBrush(grad)
    for i, h in enumerate(heights):
        bh = h * size * 0.9
        p.drawRoundedRect(QRectF(i * (bw + gap), (size - bh) / 2, bw, bh), bw / 2, bw / 2)
    p.end()
    return pm


def fmt_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------- widgets
class ToggleSwitch(QAbstractButton):
    """iOS-style switch with optional text, painted with the current theme."""

    def __init__(self, text: str = "", checked: bool = False, parent=None):
        super().__init__(parent)
        self.setText(text)
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.PointingHandCursor)
        self._offset = 1.0 if checked else 0.0
        self._anim = QPropertyAnimation(self, b"offset", self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self.toggled.connect(self._animate)

    def _animate(self, on: bool):
        self._anim.stop()
        self._anim.setStartValue(self._offset)
        self._anim.setEndValue(1.0 if on else 0.0)
        self._anim.start()

    def _get_offset(self):
        return self._offset

    def _set_offset(self, v):
        self._offset = v
        self.update()

    offset = Property(float, _get_offset, _set_offset)

    def sizeHint(self):
        w = 40
        if self.text():
            w += 8 + self.fontMetrics().horizontalAdvance(self.text())
        return QSize(w, 24)

    def hitButton(self, pos):
        return self.rect().contains(pos)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        track = QRectF(0, (self.height() - 22) / 2, 40, 22)
        color = blend(QColor(T["toggle_off"]), QColor(T["accent"]), self._offset)
        if not self.isEnabled():
            color.setAlpha(110)
        p.setPen(Qt.NoPen)
        p.setBrush(color)
        p.drawRoundedRect(track, 11, 11)
        p.setBrush(QColor("#FFFFFF"))
        p.drawEllipse(QRectF(3 + self._offset * 18, track.y() + 3, 16, 16))
        if self.text():
            p.setPen(QColor(T["text"]))
            p.drawText(QRectF(48, 0, self.width() - 48, self.height()),
                       Qt.AlignVCenter | Qt.AlignLeft, self.text())
        p.end()


class ThemeSwitch(ToggleSwitch):
    """Larger switch for the header: moon in the knob for dark, sun for light."""

    W, H, KNOB = 52, 28, 22

    def sizeHint(self):
        return QSize(self.W, self.H)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        track = QRectF(0, (self.height() - self.H) / 2, self.W, self.H)
        dark_track, light_track = QColor("#2B2F45"), QColor("#FFD666")
        p.setPen(Qt.NoPen)
        p.setBrush(blend(dark_track, light_track, self._offset))
        p.drawRoundedRect(track, self.H / 2, self.H / 2)

        pad = (self.H - self.KNOB) / 2
        kx = pad + self._offset * (self.W - self.KNOB - 2 * pad)
        knob = QRectF(kx, track.y() + pad, self.KNOB, self.KNOB)
        p.setBrush(QColor("#FFFFFF"))
        p.drawEllipse(knob)

        c = knob.center()
        if self._offset >= 0.5:  # sun
            sun = QColor("#F5A524")
            p.setBrush(sun)
            p.drawEllipse(c, 4.0, 4.0)
            pen = QPen(sun, 1.6)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            for i in range(8):
                a = math.radians(i * 45)
                p.drawLine(QPointF(c.x() + 6.2 * math.cos(a), c.y() + 6.2 * math.sin(a)),
                           QPointF(c.x() + 8.4 * math.cos(a), c.y() + 8.4 * math.sin(a)))
        else:  # moon: circle minus an offset circle
            moon = QPainterPath()
            moon.addEllipse(c, 6.5, 6.5)
            bite = QPainterPath()
            bite.addEllipse(QPointF(c.x() + 3.6, c.y() - 3.0), 5.6, 5.6)
            p.setBrush(QColor("#5B4BC4"))
            p.drawPath(moon.subtracted(bite))
        p.end()


class DropZone(QFrame):
    fileDropped = Signal(str)
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DropZone")
        self.setAcceptDrops(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setProperty("hover", False)

    def _set_hover(self, on: bool):
        self.setProperty("hover", on)
        repolish(self)

    def enterEvent(self, e):
        self._set_hover(True)
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._set_hover(False)
        super().leaveEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit()

    def dragEnterEvent(self, e):
        urls = e.mimeData().urls()
        if urls and urls[0].isLocalFile():
            e.acceptProposedAction()
            self._set_hover(True)

    def dragLeaveEvent(self, e):
        self._set_hover(False)

    def dropEvent(self, e):
        self._set_hover(False)
        urls = e.mimeData().urls()
        if urls and urls[0].isLocalFile():
            self.fileDropped.emit(os.path.normpath(urls[0].toLocalFile()))


def card(title: str = ""):
    frame = QFrame()
    frame.setObjectName("Card")
    frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)  # never squashed
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(16, 10, 16, 12)
    lay.setSpacing(6)
    if title:
        t = QLabel(title)
        t.setObjectName("CardTitle")
        lay.addWidget(t)
    return frame, lay


def label(text: str, name: str = "Field") -> QLabel:
    lb = QLabel(text)
    lb.setObjectName(name)
    return lb


# ---------------------------------------------------------------- engine / job
class Cancelled(Exception):
    pass


class Engine:
    """Keeps loaded models between runs; only one big model on the GPU at a time."""

    def __init__(self):
        self.whisper_models = {}
        self.translator = None

    def unload_whisper(self):
        self.whisper_models.clear()
        free_gpu()

    def unload_translator(self):
        if self.translator is not None:
            self.translator.unload()
            self.translator = None
            free_gpu()


TS_RE = re.compile(r"\[(?:(\d+):)?(\d+):(\d+\.\d+) --> (?:(\d+):)?(\d+):(\d+\.\d+)\]")


class JobWriter:
    """Replaces stdout/stderr in the job: forwards lines, tracks progress, honours Cancel."""

    def __init__(self, job):
        self.job = job

    def write(self, text):
        if self.job.cancel_requested:
            raise Cancelled()
        if "\r" in text:  # tqdm redraws; keep only the finished bar
            if "100%" in text:
                self.job.log.emit(text.replace("\r", "").strip())
            return
        for line in text.splitlines():
            if not line.strip():
                continue
            self.job.log.emit(line.rstrip())
            m = TS_RE.search(line)
            if m and self.job.duration:
                end = int(m.group(4) or 0) * 3600 + int(m.group(5)) * 60 + float(m.group(6))
                self.job.report_transcribe(end)

    def flush(self):
        pass


class Job(QObject):
    log = Signal(str)
    progress = Signal(float, str)   # fraction 0..1, status text
    step = Signal(int)              # index into the step list
    done = Signal(list)
    failed = Signal(str)
    cancelled = Signal()
    finished = Signal()

    def __init__(self, engine: Engine, params: dict):
        super().__init__()
        self.engine = engine
        self.p = params
        self.cancel_requested = False
        self.duration = 0.0
        self.transcribe_share = 0.97

    def report_transcribe(self, seconds: float):
        f = min(1.0, seconds / self.duration) * self.transcribe_share
        self.progress.emit(f, f"Transcribing… {fmt_time(seconds)} / {fmt_time(self.duration)}")

    def check_cancel(self):
        if self.cancel_requested:
            raise Cancelled()

    @Slot()
    def run(self):
        p, eng = self.p, self.engine
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = JobWriter(self)
        try:
            self.step.emit(0)
            self.progress.emit(0.0, "Loading libraries…")
            import torch
            import whisper

            device = "cuda" if torch.cuda.is_available() else "cpu"

            # 1. Whisper
            eng.unload_translator()
            if p["model"] not in eng.whisper_models:
                eng.unload_whisper()
                self.progress.emit(0.0, f"Loading Whisper '{p['model']}'…")
                self.log.emit(f"Loading Whisper '{p['model']}' on {device} "
                              "(first time downloads it)…")
                eng.whisper_models[p["model"]] = whisper.load_model(
                    p["model"], device=device, download_root=WHISPER_DIR)
            model = eng.whisper_models[p["model"]]
            self.check_cancel()

            self.progress.emit(0.0, "Reading audio…")
            audio = whisper.load_audio(p["path"])
            self.duration = len(audio) / whisper.audio.SAMPLE_RATE
            self.transcribe_share = 0.85 if p["post_translator"] else 0.97

            self.log.emit(f"Task: {p['task']} | input: {p['src'] or 'auto'} | "
                          f"model: {p['model']} | device: {device}")
            self.progress.emit(0.0, f"Transcribing… 00:00 / {fmt_time(self.duration)}")
            result = model.transcribe(audio, task=p["task"], language=p["src"],
                                      fp16=(device == "cuda"), verbose=True)
            del model, audio
            self.check_cancel()

            detected = result.get("language") or p["src"] or "xx"
            self.log.emit(f"Detected language: {lang_name(detected)}")
            segments = result["segments"]

            # 2. Optional line-by-line translation
            tgt, tr_name = p["tgt"], p["post_translator"]
            if p["task"] == "translate":
                out_lang = "en"
            elif tr_name and detected != tgt:
                self.step.emit(1)
                cls = TRANSLATORS[tr_name]
                if detected not in cls.codes:
                    raise RuntimeError(f"{tr_name} does not support the detected language "
                                       f"({lang_name(detected)}).")
                eng.unload_whisper()
                if eng.translator is None or eng.translator.name != tr_name:
                    eng.unload_translator()
                    self.progress.emit(self.transcribe_share, f"Loading {tr_name}…")
                    self.log.emit(f"Loading translator {tr_name} (first time downloads it)…")
                    eng.translator = cls()
                    eng.translator.load(device)
                self.check_cancel()
                share = self.transcribe_share
                tr_re = re.compile(r"Translated (\d+)/(\d+)")

                def tr_log(msg):
                    self.check_cancel()
                    self.log.emit(msg)
                    m = tr_re.search(msg)
                    if m:
                        a, b = int(m.group(1)), int(m.group(2))
                        self.progress.emit(share + (0.98 - share) * a / max(b, 1),
                                           f"Translating… {a}/{b} lines")

                self.log.emit(f"Translating {lang_name(detected)} → {lang_name(tgt)}…")
                self.progress.emit(share, "Translating…")
                texts = eng.translator.translate([s["text"] for s in segments],
                                                 detected, tgt, log=tr_log)
                segments = [dict(s, text=t) for s, t in zip(segments, texts)]
                out_lang = tgt
            else:
                if tr_name:
                    self.log.emit("Input is already in the output language; "
                                  "no translation needed.")
                out_lang = detected

            # 3. Save
            self.step.emit(2)
            self.progress.emit(0.99, "Saving…")
            base = os.path.splitext(os.path.basename(p["path"]))[0]
            saved = []
            for key in p["formats"]:
                lbl, writer, _default = OUTPUT_FORMATS[key]
                out_path = os.path.join(p["out_dir"], f"{base}.{out_lang}.{key}")
                n = writer(segments, out_path)
                self.log.emit(f"{lbl}: {n} lines")
                saved.append(out_path)
            self.progress.emit(1.0, "Done")
            self.done.emit(saved)
        except Cancelled:
            self.cancelled.emit()
        except Exception as e:  # report anything to the GUI
            self.failed.emit(f"{type(e).__name__}: {e}")
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            self.finished.emit()


# ---------------------------------------------------------------- main window
class MainWindow(QMainWindow):
    gpuDetected = Signal(str)
    durationFound = Signal(str, str)  # path, text

    def __init__(self):
        super().__init__()
        self.engine = Engine()
        self.thread = None
        self.job = None
        self.file_path = ""
        self.last_output = None
        self.steps = ["Transcribe", "Translate", "Save"]
        self.current_step = -1

        self.setWindowTitle("Audio and Video Subtitle Maker")
        self.setWindowIcon(QIcon(waveform_pixmap(64)))
        self.resize(980, 800)
        self.setMinimumSize(860, 640)

        # Content sits in a scroll area so small screens scroll instead of squashing cards
        scroll = QScrollArea()
        scroll.setObjectName("Scroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        self.setCentralWidget(scroll)
        root = QWidget()
        root.setObjectName("Root")
        scroll.setWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(20, 14, 20, 16)
        main.setSpacing(10)

        main.addLayout(self._build_header())
        main.addWidget(self._build_source())
        main.addWidget(self._build_languages())
        main.addWidget(self._build_engine())
        main.addWidget(self._build_output())
        main.addWidget(self._build_actions())
        self.log_card = self._build_log()
        main.addWidget(self.log_card)
        self.spacer = QWidget()
        main.addWidget(self.spacer)
        self.main_layout = main
        self.set_log_expanded(False)

        # signals
        self.in_combo.currentTextChanged.connect(self.update_translator_ui)
        self.out_combo.currentTextChanged.connect(self.on_output_or_model_change)
        self.model_combo.currentTextChanged.connect(self.on_output_or_model_change)
        self.tr_combo.currentTextChanged.connect(self.update_translator_ui)
        self.gpuDetected.connect(self.gpu_pill.setText)
        self.durationFound.connect(self._on_duration)

        self.apply_theme("dark")
        self.update_translator_ui()
        self.update_start_state()
        self.set_steps(-1)
        threading.Thread(target=self._detect_gpu, daemon=True).start()

    # ---------- build ----------
    def _build_header(self):
        row = QHBoxLayout()
        row.setSpacing(12)
        icon = QLabel()
        icon.setPixmap(waveform_pixmap(36))
        row.addWidget(icon)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(label("Audio and Video Subtitle Maker", "Title"))
        titles.addWidget(label("Speech → subtitles, offline", "Muted"))
        row.addLayout(titles)
        row.addStretch(1)
        self.theme_toggle = ThemeSwitch()
        self.theme_toggle.setToolTip("Switch to light mode")
        self.theme_toggle.toggled.connect(lambda on: self.apply_theme("light" if on else "dark"))
        self.theme_toggle.toggled.connect(lambda on: self.theme_toggle.setToolTip(
            "Switch to dark mode" if on else "Switch to light mode"))
        row.addWidget(self.theme_toggle)
        self.gpu_pill = label("Detecting GPU…", "Pill")
        row.addWidget(self.gpu_pill)
        return row

    def _build_source(self):
        frame, lay = card("Source")
        self.drop = DropZone()
        dz = QHBoxLayout(self.drop)
        dz.setContentsMargins(16, 12, 18, 12)
        dz.setSpacing(14)
        self.badge = label("+", "FileBadge")
        self.badge.setFixedSize(40, 48)
        self.badge.setAlignment(Qt.AlignCenter)
        dz.addWidget(self.badge)
        texts = QVBoxLayout()
        texts.setSpacing(2)
        self.file_name = label("Drop audio or video here, or click to browse", "FileName")
        texts.addWidget(self.file_name)
        texts.addWidget(label("Accepted: mp3, mp4, wav, m4a, flac, ogg, mkv, mov, webm …",
                              "Muted"))
        dz.addLayout(texts, 1)
        self.duration_lbl = label("", "Duration")
        dz.addWidget(self.duration_lbl)
        self.drop.clicked.connect(self.browse_file)
        self.drop.fileDropped.connect(self.set_file)
        lay.addWidget(self.drop)
        return frame

    def _build_languages(self):
        frame, lay = card("Languages")
        g = QGridLayout()
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(4)
        g.addWidget(label("Spoken language"), 0, 0)
        g.addWidget(label("Subtitle language"), 0, 2)
        self.in_combo = QComboBox()
        self.in_combo.addItems([AUTO] + LANGUAGE_NAMES)
        self.in_combo.setMaxVisibleItems(16)
        g.addWidget(self.in_combo, 1, 0)
        g.addWidget(label("→", "Arrow"), 1, 1)
        self.out_combo = QComboBox()
        self.out_combo.addItems([OUT_SAME] + LANGUAGE_NAMES)
        self.out_combo.setCurrentText("English")
        self.out_combo.setMaxVisibleItems(16)
        g.addWidget(self.out_combo, 1, 2)
        g.setColumnStretch(0, 1)
        g.setColumnStretch(2, 1)
        lay.addLayout(g)
        return frame

    def _build_engine(self):
        frame, lay = card("Engine")
        g = QGridLayout()
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(4)
        g.addWidget(label("Whisper model"), 0, 0)
        g.addWidget(label("Translator"), 0, 2)
        self.model_combo = QComboBox()
        self.model_combo.addItems(MODELS)
        g.addWidget(self.model_combo, 1, 0)
        g.addWidget(label("→", "Arrow"), 1, 1)
        self.tr_combo = QComboBox()
        self.tr_combo.addItems(TRANSLATOR_CHOICES)
        g.addWidget(self.tr_combo, 1, 2)
        self.model_hint = label("turbo is fastest but cannot translate", "Muted")
        g.addWidget(self.model_hint, 2, 0)
        chip_row = QHBoxLayout()
        self.tr_chip = label("", "ChipInfo")
        chip_row.addWidget(self.tr_chip)
        chip_row.addStretch(1)
        g.addLayout(chip_row, 2, 2)
        g.setColumnStretch(0, 1)
        g.setColumnStretch(2, 1)
        lay.addLayout(g)
        return frame

    def _build_output(self):
        frame, lay = card("Output")
        g = QGridLayout()
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(10)
        g.addWidget(label("Save to"), 0, 0)
        self.outdir_edit = QLineEdit()
        self.outdir_edit.setPlaceholderText(f"{APP_DIR} (default)")
        g.addWidget(self.outdir_edit, 0, 1)
        self.outdir_browse = QPushButton("Browse")
        self.outdir_browse.clicked.connect(self.browse_outdir)
        g.addWidget(self.outdir_browse, 0, 2)
        g.addWidget(label("Formats"), 1, 0)
        fmts = QHBoxLayout()
        fmts.setSpacing(22)
        self.format_toggles = {}
        for key, (_lbl, _writer, default) in OUTPUT_FORMATS.items():
            sw = ToggleSwitch(key.upper(), checked=default)
            sw.toggled.connect(self.update_start_state)
            fmts.addWidget(sw)
            self.format_toggles[key] = sw
        fmts.addSpacing(8)
        self.format_error = label("Pick at least one format", "Error")
        fmts.addWidget(self.format_error)
        fmts.addStretch(1)
        g.addLayout(fmts, 1, 1, 1, 2)
        g.setColumnStretch(1, 1)
        lay.addLayout(g)
        lay.addWidget(label(f"Models stored in {MODELS_DIR}", "Muted"))
        return frame

    def _build_actions(self):
        frame, lay = card()
        row = QHBoxLayout()
        row.setSpacing(10)
        self.start_btn = QPushButton("▶  Start")
        self.start_btn.setObjectName("Primary")
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.setMinimumWidth(110)
        self.start_btn.clicked.connect(self.start_or_cancel)
        row.addWidget(self.start_btn)
        self.open_btn = QPushButton("Open folder")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self.open_folder)
        row.addWidget(self.open_btn)
        row.addSpacing(12)
        prog = QVBoxLayout()
        prog.setSpacing(6)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        prog.addWidget(self.progress)
        status_row = QHBoxLayout()
        self.status_lbl = label("Ready", "Muted")
        status_row.addWidget(self.status_lbl)
        status_row.addStretch(1)
        self.steps_lbl = QLabel()
        self.steps_lbl.setTextFormat(Qt.RichText)
        status_row.addWidget(self.steps_lbl)
        prog.addLayout(status_row)
        row.addLayout(prog, 1)
        lay.addLayout(row)
        return frame

    def _build_log(self):
        frame, lay = card()
        self.log_btn = QPushButton("Log  ▸")
        self.log_btn.setObjectName("LogHeader")
        self.log_btn.setCursor(Qt.PointingHandCursor)
        self.log_btn.clicked.connect(lambda: self.set_log_expanded(not self.log_view.isVisible()))
        lay.addWidget(self.log_btn)
        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("Log")
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(140)
        lay.addWidget(self.log_view)
        return frame

    # ---------- theme ----------
    def apply_theme(self, name: str):
        T.clear()
        T.update(THEMES[name])
        QApplication.instance().setStyleSheet(build_qss(T))
        self.set_steps(self.current_step)
        for w in self.findChildren(ToggleSwitch):
            w.update()

    # ---------- small helpers ----------
    def set_log_expanded(self, on: bool):
        self.log_view.setVisible(on)
        self.log_card.setSizePolicy(QSizePolicy.Preferred,
                                    QSizePolicy.Expanding if on else QSizePolicy.Fixed)
        self.log_btn.setText("Log  ▾" if on else "Log  ▸")
        self.main_layout.setStretchFactor(self.log_card, 1 if on else 0)
        self.main_layout.setStretchFactor(self.spacer, 0 if on else 1)

    def set_steps(self, current: int):
        self.current_step = current
        needs_tr = self.needs_post_translation()
        names = [s for s in self.steps if needs_tr or s != "Translate"]
        idx_map = {s: i for i, s in enumerate(self.steps)}
        parts = []
        for s in names:
            i = idx_map[s]
            if i == current:
                parts.append(f'<span style="color:{T["accent"]}; font-weight:600">{s}</span>')
            elif i < current:
                parts.append(f'<span style="color:{T["text"]}">{s}</span>')
            else:
                parts.append(f'<span style="color:{T["muted"]}">{s}</span>')
        self.steps_lbl.setText(f' <span style="color:{T["muted"]}">→</span> '.join(parts))

    def append_log(self, text: str, kind: str = ""):
        color = {"ok": T["ok"], "err": T["danger"], "note": T["accent"]}.get(kind)
        safe = html.escape(text)
        self.log_view.appendHtml(f'<span style="color:{color}">{safe}</span>' if color
                                 else f"<span>{safe}</span>")

    def _detect_gpu(self):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=5,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            name = out.stdout.strip().splitlines()[0]
            for junk in ("NVIDIA ", "GeForce ", " Laptop GPU"):
                name = name.replace(junk, "")
            self.gpuDetected.emit(f"{name} · CUDA")
        except Exception:
            self.gpuDetected.emit("CPU only")

    def _probe_duration(self, path: str):
        try:
            out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                  "-of", "default=nw=1:nk=1", path],
                                 capture_output=True, text=True, timeout=10,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.durationFound.emit(path, fmt_time(float(out.stdout.strip())))
        except Exception:
            self.durationFound.emit(path, "")

    def _on_duration(self, path, text):
        if path == self.file_path:
            self.duration_lbl.setText(text)

    # ---------- file / folder ----------
    def browse_file(self):
        if self.job:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Choose audio or video", "", MEDIA_FILTER)
        if path:
            self.set_file(os.path.normpath(path))

    def set_file(self, path: str):
        if self.job:
            return
        if not os.path.isfile(path):
            return
        ext = os.path.splitext(path)[1].lower()
        if ext not in MEDIA_EXTS:
            QMessageBox.warning(self, "Audio and Video Subtitle Maker",
                                f"'{ext or 'this file'}' is not a supported audio/video type.")
            return
        self.file_path = path
        self.file_name.setText(os.path.basename(path))
        self.file_name.setToolTip(path)
        self.badge.setText(ext.lstrip(".")[:4])
        self.duration_lbl.setText("…")
        threading.Thread(target=self._probe_duration, args=(path,), daemon=True).start()

    def browse_outdir(self):
        start = self.outdir_edit.text().strip() or APP_DIR
        path = QFileDialog.getExistingDirectory(self, "Choose output folder", start)
        if path:
            self.outdir_edit.setText(os.path.normpath(path))

    def open_folder(self):
        if self.last_output and os.path.exists(self.last_output):
            os.startfile(os.path.dirname(self.last_output))

    # ---------- translation choices ----------
    def target_code(self):
        name = self.out_combo.currentText()
        return None if name == OUT_SAME else NAME_TO_CODE[name]

    def source_code(self):
        name = self.in_combo.currentText()
        return None if name == AUTO else NAME_TO_CODE[name]

    def needs_translation(self):
        tgt = self.target_code()
        return tgt is not None and self.source_code() != tgt

    def needs_post_translation(self):
        return self.needs_translation() and self.tr_combo.currentText() != TR_WHISPER

    def on_output_or_model_change(self, *_):
        """Whisper can't translate with turbo or into non-English: switch translator."""
        tgt = self.target_code()
        if self.tr_combo.currentText() == TR_WHISPER and tgt is not None:
            if self.model_combo.currentText() == "turbo" or tgt != "en":
                self.tr_combo.setCurrentText(DEFAULT_TRANSLATOR)
        self.update_translator_ui()

    def translator_problem(self):
        if not self.needs_translation():
            return None
        tgt, src, tr = self.target_code(), self.source_code(), self.tr_combo.currentText()
        if tr == TR_WHISPER:
            return None if tgt == "en" else "Whisper only translates into English"
        cls = TRANSLATORS[tr]
        if tgt not in cls.codes:
            return f"{tr} does not support {lang_name(tgt)}"
        if src is not None and src not in cls.codes:
            return f"{tr} does not support {lang_name(src)}"
        return None

    def _set_chip(self, text: str, kind: str):
        self.tr_chip.setText(text)
        self.tr_chip.setObjectName({"ok": "ChipOk", "bad": "ChipBad"}.get(kind, "ChipInfo"))
        repolish(self.tr_chip)

    def update_translator_ui(self, *_):
        model = self.model_combo.currentText()
        self.model_hint.setText({
            "turbo": "turbo is fastest but cannot translate",
            MODEL_AUTO: "Auto: turbo to transcribe, medium for Whisper translation",
        }.get(model, "turbo is fastest but cannot translate"))
        if not self.needs_translation():
            self.tr_combo.setEnabled(False)
            self._set_chip("No translation needed", "info")
        else:
            self.tr_combo.setEnabled(not self.job)
            problem = self.translator_problem()
            if problem:
                self._set_chip(f"✗ {problem}", "bad")
            elif self.tr_combo.currentText() == TR_WHISPER:
                self._set_chip("✓ Uses Whisper 'medium' (turbo can't translate)", "ok")
            else:
                n = len(TRANSLATORS[self.tr_combo.currentText()].codes)
                self._set_chip(f"✓ {n} languages", "ok")
        if not self.job:
            self.set_steps(-1)
        self.update_start_state()

    def update_start_state(self, *_):
        has_fmt = any(t.isChecked() for t in self.format_toggles.values())
        self.format_error.setVisible(not has_fmt)
        if not self.job:
            self.start_btn.setEnabled(has_fmt and self.translator_problem() is None)

    # ---------- run ----------
    def set_running(self, running: bool):
        for w in (self.in_combo, self.out_combo, self.model_combo, self.outdir_edit,
                  self.outdir_browse, *self.format_toggles.values()):
            w.setEnabled(not running)
        if running:
            self.tr_combo.setEnabled(False)
            self.start_btn.setText("Cancel")
            self.start_btn.setObjectName("Danger")
            self.start_btn.setEnabled(True)
        else:
            self.start_btn.setText("▶  Start")
            self.start_btn.setObjectName("Primary")
        repolish(self.start_btn)

    def start_or_cancel(self):
        if self.job:
            self.job.cancel_requested = True
            self.start_btn.setEnabled(False)
            self.start_btn.setText("Cancelling…")
            self.status_lbl.setText("Cancelling… (stops after the current segment)")
            return
        self.start()

    def start(self):
        if not self.file_path or not os.path.isfile(self.file_path):
            QMessageBox.warning(self, "Audio and Video Subtitle Maker", "Choose an audio or video file first.")
            return
        formats = [k for k, t in self.format_toggles.items() if t.isChecked()]
        if not formats:
            return
        problem = self.translator_problem()
        if problem:
            QMessageBox.warning(self, "Audio and Video Subtitle Maker", problem + ".")
            return

        src, tgt, tr = self.source_code(), self.target_code(), self.tr_combo.currentText()
        model = self.model_combo.currentText()
        notes = []
        if self.needs_translation() and tr == TR_WHISPER:
            task, post = "translate", None
            if model in (MODEL_AUTO, "turbo"):
                if model == "turbo":
                    notes.append("Note: 'turbo' cannot translate. "
                                 f"Using '{DEFAULT_TRANSLATE_MODEL}' instead.")
                model = DEFAULT_TRANSLATE_MODEL
        else:
            task = "transcribe"
            post = tr if self.needs_translation() else None
            if model == MODEL_AUTO:
                model = DEFAULT_TRANSCRIBE_MODEL

        out_dir = self.outdir_edit.text().strip() or APP_DIR
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, "Audio and Video Subtitle Maker",
                                 f"Cannot use output folder:\n{out_dir}\n\n{e}")
            return

        self.log_view.clear()
        self.set_log_expanded(True)
        for n in notes:
            self.append_log(n, "note")
        self.progress.setValue(0)
        self.open_btn.setEnabled(False)

        params = dict(path=self.file_path, src=src, tgt=tgt, task=task, model=model,
                      post_translator=post, out_dir=out_dir, formats=formats)
        self.thread = QThread(self)
        self.job = Job(self.engine, params)
        self.job.moveToThread(self.thread)
        self.thread.started.connect(self.job.run)
        self.job.log.connect(self.append_log)
        self.job.progress.connect(self.on_progress)
        self.job.step.connect(self.set_steps)
        self.job.done.connect(self.on_done)
        self.job.failed.connect(self.on_failed)
        self.job.cancelled.connect(self.on_cancelled)
        self.job.finished.connect(self.on_job_finished)
        self.set_running(True)
        self.set_steps(0)
        self.thread.start()

    def on_progress(self, frac: float, text: str):
        self.progress.setValue(int(frac * 1000))
        self.status_lbl.setText(f"{int(frac * 100)}%  ·  {text}")

    def on_done(self, saved: list):
        self.last_output = saved[0]
        self.open_btn.setEnabled(True)
        for path in saved:
            self.append_log(f"Saved: {path}", "ok")
        self.status_lbl.setText(f"Done  ·  {len(saved)} file(s) saved")

    def on_failed(self, msg: str):
        self.append_log(f"ERROR: {msg}", "err")
        self.status_lbl.setText("Failed")
        QMessageBox.critical(self, "Audio and Video Subtitle Maker", msg)

    def on_cancelled(self):
        self.append_log("Cancelled.", "err")
        self.status_lbl.setText("Cancelled")
        self.progress.setValue(0)

    def on_job_finished(self):
        self.thread.quit()
        self.thread.wait()
        self.job.deleteLater()
        self.thread.deleteLater()
        self.job = None
        self.thread = None
        self.set_running(False)
        self.update_translator_ui()
        self.set_steps(-1 if self.status_lbl.text() in ("Cancelled", "Failed") else 3)

    def closeEvent(self, event):
        if self.job:
            ans = QMessageBox.question(self, "Audio and Video Subtitle Maker",
                                       "A job is still running. Quit anyway?")
            if ans != QMessageBox.Yes:
                event.ignore()
                return
            os._exit(0)  # the GPU job can't be interrupted mid-segment
        event.accept()


def main():
    if sys.platform == "win32":
        try:  # own taskbar icon instead of python.exe's
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("WhisperSRTMaker")
        except Exception:
            pass
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
