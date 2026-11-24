"""Whisper SRT maker.

Pick an audio or video file, choose the spoken language and the subtitle
language, and get .srt/.vtt/.txt/.tsv files.

Translation:
  * Whisper itself can only translate into English (and not with 'turbo').
  * For any other case a separate translator (NLLB-200 or Hy-MT2) translates
    each subtitle line after Whisper transcribes, keeping the timestamps.

All models are stored under <this folder>\\models, not on C:.

Run:  python whisper_srt.py   (or double-click run.bat)
"""

import gc
import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_DIR, "models")
WHISPER_DIR = os.path.join(MODELS_DIR, "whisper")

# Must be set before huggingface_hub / transformers are imported anywhere.
os.environ["HF_HOME"] = os.path.join(MODELS_DIR, "hf")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from translators import TRANSLATORS  # noqa: E402  (needs HF_HOME set first)

# Same language list Whisper uses (code -> name). Importing whisper.tokenizer
# is cheap; the heavy torch import is deferred to the worker thread.
try:
    from whisper.tokenizer import LANGUAGES
except ImportError:
    LANGUAGES = {"en": "english", "es": "spanish"}

NAME_TO_CODE = {name.title(): code for code, name in LANGUAGES.items()}
LANGUAGE_NAMES = sorted(NAME_TO_CODE)

AUTO = "Auto-detect"
OUT_SAME = "Same as input"
MODEL_AUTO = "Auto"
MODELS = [MODEL_AUTO, "tiny", "base", "small", "medium", "turbo", "large-v3"]

TR_WHISPER = "Whisper (English only)"
TRANSLATOR_CHOICES = [TR_WHISPER] + list(TRANSLATORS)
DEFAULT_TRANSLATOR = "NLLB-200 600M"

# turbo is not trained for translation; medium fits in 8 GB VRAM.
DEFAULT_TRANSCRIBE_MODEL = "turbo"
DEFAULT_TRANSLATE_MODEL = "medium"

MEDIA_TYPES = [
    ("Audio / Video", "*.mp3 *.wav *.m4a *.flac *.ogg *.opus *.aac *.wma "
                      "*.mp4 *.mkv *.mov *.avi *.webm *.flv *.wmv *.m4v"),
    ("All files", "*.*"),
]


def lang_name(code: str) -> str:
    return LANGUAGES.get(code, code).title()


def srt_timestamp(seconds: float) -> str:
    ms = max(0, round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments, path: str) -> int:
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            text = seg["text"].strip()
            if not text:
                continue
            count += 1
            f.write(f"{count}\n")
            f.write(f"{srt_timestamp(seg['start'])} --> {srt_timestamp(seg['end'])}\n")
            f.write(f"{text}\n\n")
    return count


def write_txt(segments, path: str) -> int:
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            text = seg["text"].strip()
            if not text:
                continue
            count += 1
            f.write(f"{text}\n")
    return count


def write_vtt(segments, path: str) -> int:
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            text = seg["text"].strip()
            if not text:
                continue
            count += 1
            start = srt_timestamp(seg["start"]).replace(",", ".")
            end = srt_timestamp(seg["end"]).replace(",", ".")
            f.write(f"{start} --> {end}\n{text}\n\n")
    return count


def write_tsv(segments, path: str) -> int:
    # Same layout as Whisper's own TSV: start/end in integer milliseconds
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        f.write("start\tend\ttext\n")
        for seg in segments:
            text = seg["text"].strip().replace("\t", " ")
            if not text:
                continue
            count += 1
            f.write(f"{round(seg['start'] * 1000)}\t{round(seg['end'] * 1000)}\t{text}\n")
    return count


# Output formats: key -> (checkbox label, writer, on by default)
OUTPUT_FORMATS = {
    "srt": ("Subtitles (.srt)", write_srt, True),
    "vtt": ("Web subtitles (.vtt)", write_vtt, False),
    "txt": ("Text (.txt)", write_txt, False),
    "tsv": ("Table (.tsv)", write_tsv, False),
}


class QueueWriter:
    """File-like object that forwards printed text to the GUI log queue."""

    def __init__(self, q: queue.Queue):
        self.q = q

    def write(self, text):
        # tqdm download bars redraw with '\r'; only log the finished bar
        if "\r" in text:
            if "100%" in text:
                self.q.put(("log", text.replace("\r", "").strip()))
            return
        if text.strip():
            self.q.put(("log", text.rstrip()))

    def flush(self):
        pass


def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.q: queue.Queue = queue.Queue()
        self.whisper_models = {}  # name -> loaded Whisper model
        self.translator = None    # loaded BaseTranslator, or None
        self.last_output = None
        self.running = False

        root.title("Whisper SRT Maker")
        root.minsize(680, 480)

        frm = ttk.Frame(root, padding=12)
        frm.pack(fill="both", expand=True)
        frm.columnconfigure(1, weight=1)

        # Input file
        ttk.Label(frm, text="Input file:").grid(row=0, column=0, sticky="w", pady=4)
        self.file_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.file_var).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(frm, text="Browse…", command=self.browse).grid(row=0, column=2)

        # Input language
        ttk.Label(frm, text="Input language:").grid(row=1, column=0, sticky="w", pady=4)
        self.in_var = tk.StringVar(value=AUTO)
        ttk.Combobox(frm, textvariable=self.in_var, state="readonly",
                     values=[AUTO] + LANGUAGE_NAMES).grid(row=1, column=1, sticky="w", padx=6)

        # Output language
        ttk.Label(frm, text="Output language:").grid(row=2, column=0, sticky="w", pady=4)
        self.out_var = tk.StringVar(value="English")
        ttk.Combobox(frm, textvariable=self.out_var, state="readonly",
                     values=[OUT_SAME] + LANGUAGE_NAMES).grid(row=2, column=1, sticky="w", padx=6)

        # Whisper model
        ttk.Label(frm, text="Whisper model:").grid(row=3, column=0, sticky="w", pady=4)
        self.model_var = tk.StringVar(value=MODEL_AUTO)
        ttk.Combobox(frm, textvariable=self.model_var, state="readonly",
                     values=MODELS).grid(row=3, column=1, sticky="w", padx=6)

        # Translator (only shown when the output language differs from the input)
        self.tr_label = ttk.Label(frm, text="Translator:")
        self.tr_label.grid(row=4, column=0, sticky="w", pady=4)
        self.tr_frame = ttk.Frame(frm)
        self.tr_frame.grid(row=4, column=1, columnspan=2, sticky="w", padx=6)
        self.tr_var = tk.StringVar(value=TR_WHISPER)
        ttk.Combobox(self.tr_frame, textvariable=self.tr_var, state="readonly", width=24,
                     values=TRANSLATOR_CHOICES).pack(side="left")
        self.tr_hint = ttk.Label(self.tr_frame, text="", foreground="gray")
        self.tr_hint.pack(side="left", padx=8)

        # Output folder (empty = folder this program is in)
        ttk.Label(frm, text="Output folder:").grid(row=5, column=0, sticky="w", pady=4)
        self.outdir_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.outdir_var).grid(row=5, column=1, sticky="ew", padx=6)
        ttk.Button(frm, text="Browse…", command=self.browse_outdir).grid(row=5, column=2)
        ttk.Label(frm, text=f"Empty = {APP_DIR}   |   Models stored in {MODELS_DIR}",
                  foreground="gray").grid(row=6, column=1, columnspan=2, sticky="w", padx=6)

        # Output files (at least one must be ticked)
        ttk.Label(frm, text="Output files:").grid(row=7, column=0, sticky="w", pady=4)
        fmt_frame = ttk.Frame(frm)
        fmt_frame.grid(row=7, column=1, columnspan=2, sticky="w", padx=6)
        self.format_vars = {}
        for key, (label, _writer, default) in OUTPUT_FORMATS.items():
            var = tk.BooleanVar(value=default)
            ttk.Checkbutton(fmt_frame, text=label, variable=var,
                            command=self.update_start_state).pack(side="left", padx=(0, 12))
            self.format_vars[key] = var

        # Buttons
        btns = ttk.Frame(frm)
        btns.grid(row=8, column=0, columnspan=3, sticky="w", pady=8)
        self.start_btn = ttk.Button(btns, text="Start", command=self.start)
        self.start_btn.pack(side="left")
        self.open_btn = ttk.Button(btns, text="Open folder", command=self.open_folder,
                                   state="disabled")
        self.open_btn.pack(side="left", padx=6)
        self.progress = ttk.Progressbar(btns, mode="indeterminate", length=200)
        self.progress.pack(side="left", padx=6)

        # Log
        log_frame = ttk.Frame(frm)
        log_frame.grid(row=9, column=0, columnspan=3, sticky="nsew")
        frm.rowconfigure(9, weight=1)
        self.log = tk.Text(log_frame, height=14, wrap="word", state="disabled")
        sb = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # React to choices that affect translation
        self.in_var.trace_add("write", lambda *_: self.update_translator_ui())
        self.out_var.trace_add("write", lambda *_: self.on_output_or_model_change())
        self.model_var.trace_add("write", lambda *_: self.on_output_or_model_change())
        self.tr_var.trace_add("write", lambda *_: self.update_translator_ui())
        self.update_translator_ui()

        self.root.after(100, self.poll_queue)

    # ---------- Translation choices ----------
    def target_code(self):
        name = self.out_var.get()
        return None if name == OUT_SAME else NAME_TO_CODE[name]

    def source_code(self):
        name = self.in_var.get()
        return None if name == AUTO else NAME_TO_CODE[name]

    def on_output_or_model_change(self):
        """Whisper can't translate with turbo or into non-English: switch translator."""
        tgt = self.target_code()
        if self.tr_var.get() == TR_WHISPER and tgt is not None:
            if self.model_var.get() == "turbo" or tgt != "en":
                self.tr_var.set(DEFAULT_TRANSLATOR)
        self.update_translator_ui()

    def translator_problem(self):
        """Return a reason the current choice can't work, or None."""
        tgt, src, tr = self.target_code(), self.source_code(), self.tr_var.get()
        if tgt is None or (src is not None and src == tgt):
            return None
        if tr == TR_WHISPER:
            return None if tgt == "en" else "Whisper only translates into English"
        cls = TRANSLATORS[tr]
        if tgt not in cls.codes:
            return f"{tr} does not support {lang_name(tgt)}"
        if src is not None and src not in cls.codes:
            return f"{tr} does not support {lang_name(src)}"
        return None

    def update_translator_ui(self):
        tgt, src = self.target_code(), self.source_code()
        needs_translation = tgt is not None and src != tgt
        if needs_translation:
            self.tr_label.grid()
            self.tr_frame.grid()
            problem = self.translator_problem()
            if problem:
                self.tr_hint.configure(text=f"✗ {problem}", foreground="red")
            elif self.tr_var.get() == TR_WHISPER:
                self.tr_hint.configure(text="Uses Whisper 'medium' (turbo can't translate)",
                                       foreground="gray")
            else:
                n = len(TRANSLATORS[self.tr_var.get()].codes)
                self.tr_hint.configure(text=f"✓ {n} languages, translates each subtitle line",
                                       foreground="gray")
        else:
            self.tr_label.grid_remove()
            self.tr_frame.grid_remove()

    # ---------- UI helpers ----------
    def browse(self):
        path = filedialog.askopenfilename(title="Choose audio or video", filetypes=MEDIA_TYPES)
        if path:
            self.file_var.set(os.path.normpath(path))

    def browse_outdir(self):
        path = filedialog.askdirectory(title="Choose output folder",
                                       initialdir=self.outdir_var.get() or APP_DIR)
        if path:
            self.outdir_var.set(os.path.normpath(path))

    def selected_formats(self):
        return [key for key, var in self.format_vars.items() if var.get()]

    def update_start_state(self):
        ok = bool(self.selected_formats()) and not self.running
        self.start_btn.configure(state="normal" if ok else "disabled")

    def append_log(self, text: str):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def open_folder(self):
        if self.last_output and os.path.exists(self.last_output):
            os.startfile(os.path.dirname(self.last_output))

    def poll_queue(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self.append_log(payload)
                elif kind == "done":
                    self.finish(ok=True, output=payload)
                elif kind == "error":
                    self.finish(ok=False, output=payload)
        except queue.Empty:
            pass
        self.root.after(100, self.poll_queue)

    def finish(self, ok: bool, output):
        self.progress.stop()
        self.running = False
        self.update_start_state()
        if ok:
            self.last_output = output[0]
            self.open_btn.configure(state="normal")
            for p in output:
                self.append_log(f"Saved: {p}")
        else:
            self.append_log(f"\nERROR: {output}")
            messagebox.showerror("Whisper SRT Maker", output)

    # ---------- Work ----------
    def start(self):
        path = self.file_var.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showwarning("Whisper SRT Maker", "Choose an existing audio or video file.")
            return

        formats = self.selected_formats()
        if not formats:
            messagebox.showwarning("Whisper SRT Maker", "Tick at least one output file type.")
            return

        problem = self.translator_problem()
        if problem:
            messagebox.showwarning("Whisper SRT Maker", problem + ".")
            return

        src, tgt, tr = self.source_code(), self.target_code(), self.tr_var.get()
        model_name = self.model_var.get()
        notes = []

        if tgt is not None and src != tgt and tr == TR_WHISPER:
            # Whisper's own translation (English only)
            task, post_translator = "translate", None
            if model_name == MODEL_AUTO:
                model_name = DEFAULT_TRANSLATE_MODEL
            elif model_name == "turbo":
                model_name = DEFAULT_TRANSLATE_MODEL
                notes.append(f"Note: 'turbo' cannot translate. Using '{model_name}' instead.")
        else:
            # Transcribe in the original language; translate lines afterwards if needed
            task = "transcribe"
            post_translator = tr if tgt is not None and src != tgt else None
            if model_name == MODEL_AUTO:
                model_name = DEFAULT_TRANSCRIBE_MODEL

        out_dir = self.outdir_var.get().strip() or APP_DIR
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as e:
            messagebox.showerror("Whisper SRT Maker", f"Cannot use output folder:\n{out_dir}\n\n{e}")
            return

        self.running = True
        self.update_start_state()
        self.open_btn.configure(state="disabled")
        self.progress.start(12)
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        for n in notes:
            self.append_log(n)

        threading.Thread(target=self.worker,
                         args=(path, src, tgt, task, model_name, post_translator,
                               out_dir, formats),
                         daemon=True).start()

    def unload_whisper(self):
        self.whisper_models.clear()
        free_gpu()

    def unload_translator(self):
        if self.translator is not None:
            self.translator.unload()
            self.translator = None
            free_gpu()

    def worker(self, path, src, tgt, task, model_name, post_translator, out_dir, formats):
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = QueueWriter(self.q)
        sys.stderr = QueueWriter(self.q)
        try:
            print("Loading libraries…")
            import torch
            import whisper

            device = "cuda" if torch.cuda.is_available() else "cpu"

            # --- 1. Whisper ---
            self.unload_translator()  # only one big model on the GPU at a time
            if model_name not in self.whisper_models:
                self.unload_whisper()
                print(f"Loading Whisper '{model_name}' on {device} (first time downloads it)…")
                self.whisper_models[model_name] = whisper.load_model(
                    model_name, device=device, download_root=WHISPER_DIR)
            model = self.whisper_models[model_name]

            print(f"Task: {task} | input: {src or 'auto'} | model: {model_name} | device: {device}")
            print("Working… lines appear as they are recognised.\n")
            result = model.transcribe(path, task=task, language=src,
                                      fp16=(device == "cuda"), verbose=True)
            del model

            detected = result.get("language") or src or "xx"
            print(f"\nDetected language: {lang_name(detected)}")
            segments = result["segments"]

            # --- 2. Optional line-by-line translation ---
            if task == "translate":
                out_lang = "en"
            elif post_translator and detected != tgt:
                cls = TRANSLATORS[post_translator]
                if detected not in cls.codes:
                    raise RuntimeError(f"{post_translator} does not support the detected "
                                       f"language ({lang_name(detected)}).")
                self.unload_whisper()
                if self.translator is None or self.translator.name != post_translator:
                    self.unload_translator()
                    print(f"\nLoading translator {post_translator} (first time downloads it)…")
                    self.translator = cls()
                    self.translator.load(device)
                print(f"Translating {lang_name(detected)} → {lang_name(tgt)}…")
                texts = self.translator.translate([s["text"] for s in segments],
                                                  detected, tgt, log=print)
                segments = [dict(s, text=t) for s, t in zip(segments, texts)]
                out_lang = tgt
            else:
                if post_translator:
                    print("Input is already in the output language; no translation needed.")
                out_lang = detected

            # --- 3. Write files ---
            base = os.path.splitext(os.path.basename(path))[0]
            saved = []
            for key in formats:
                label, writer, _default = OUTPUT_FORMATS[key]
                out_path = os.path.join(out_dir, f"{base}.{out_lang}.{key}")
                n = writer(segments, out_path)
                print(f"{label}: {n} lines")
                saved.append(out_path)
            self.q.put(("done", saved))
        except Exception as e:  # report anything to the GUI
            self.q.put(("error", f"{type(e).__name__}: {e}"))
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


def main():
    if sys.platform == "win32":
        try:  # sharp text on high-DPI screens
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
