"""Subtitle-line translators used after Whisper transcribes.

Each translator takes a list of text lines plus Whisper language codes
(e.g. "es", "en") and returns the translated lines in the same order,
so the original timestamps can be kept.

Models are downloaded by Hugging Face into HF_HOME, which whisper_srt.py
points at E:\\Whisper\\models\\hf before anything is imported.
"""

import gc

# Whisper language code -> NLLB-200 (FLORES-200) code
NLLB_CODES = {
    "en": "eng_Latn", "zh": "zho_Hans", "de": "deu_Latn", "es": "spa_Latn",
    "ru": "rus_Cyrl", "ko": "kor_Hang", "fr": "fra_Latn", "ja": "jpn_Jpan",
    "pt": "por_Latn", "tr": "tur_Latn", "pl": "pol_Latn", "ca": "cat_Latn",
    "nl": "nld_Latn", "ar": "arb_Arab", "sv": "swe_Latn", "it": "ita_Latn",
    "id": "ind_Latn", "hi": "hin_Deva", "fi": "fin_Latn", "vi": "vie_Latn",
    "he": "heb_Hebr", "uk": "ukr_Cyrl", "el": "ell_Grek", "ms": "zsm_Latn",
    "cs": "ces_Latn", "ro": "ron_Latn", "da": "dan_Latn", "hu": "hun_Latn",
    "ta": "tam_Taml", "no": "nob_Latn", "th": "tha_Thai", "ur": "urd_Arab",
    "hr": "hrv_Latn", "bg": "bul_Cyrl", "lt": "lit_Latn", "mi": "mri_Latn",
    "ml": "mal_Mlym", "cy": "cym_Latn", "sk": "slk_Latn", "te": "tel_Telu",
    "fa": "pes_Arab", "lv": "lvs_Latn", "bn": "ben_Beng", "sr": "srp_Cyrl",
    "az": "azj_Latn", "sl": "slv_Latn", "kn": "kan_Knda", "et": "est_Latn",
    "mk": "mkd_Cyrl", "eu": "eus_Latn", "is": "isl_Latn", "hy": "hye_Armn",
    "ne": "npi_Deva", "mn": "khk_Cyrl", "bs": "bos_Latn", "kk": "kaz_Cyrl",
    "sq": "als_Latn", "sw": "swh_Latn", "gl": "glg_Latn", "mr": "mar_Deva",
    "pa": "pan_Guru", "si": "sin_Sinh", "km": "khm_Khmr", "sn": "sna_Latn",
    "yo": "yor_Latn", "so": "som_Latn", "af": "afr_Latn", "oc": "oci_Latn",
    "ka": "kat_Geor", "be": "bel_Cyrl", "tg": "tgk_Cyrl", "sd": "snd_Arab",
    "gu": "guj_Gujr", "am": "amh_Ethi", "yi": "ydd_Hebr", "lo": "lao_Laoo",
    "uz": "uzn_Latn", "fo": "fao_Latn", "ht": "hat_Latn", "ps": "pbt_Arab",
    "tk": "tuk_Latn", "mt": "mlt_Latn", "lb": "ltz_Latn", "my": "mya_Mymr",
    "bo": "bod_Tibt", "tl": "tgl_Latn", "as": "asm_Beng", "tt": "tat_Cyrl",
    "ln": "lin_Latn", "ha": "hau_Latn", "ba": "bak_Cyrl", "jw": "jav_Latn",
    "su": "sun_Latn", "yue": "yue_Hant",
}

# Whisper language code -> full English name used in Hy-MT2 prompts
HYMT_NAMES = {
    "zh": "Chinese", "en": "English", "fr": "French", "pt": "Portuguese",
    "es": "Spanish", "ja": "Japanese", "tr": "Turkish", "ru": "Russian",
    "ar": "Arabic", "ko": "Korean", "th": "Thai", "it": "Italian",
    "de": "German", "vi": "Vietnamese", "ms": "Malay", "id": "Indonesian",
    "tl": "Filipino", "hi": "Hindi", "pl": "Polish", "cs": "Czech",
    "nl": "Dutch", "km": "Khmer", "my": "Burmese", "fa": "Persian",
    "gu": "Gujarati", "ur": "Urdu", "te": "Telugu", "mr": "Marathi",
    "he": "Hebrew", "bn": "Bengali", "ta": "Tamil", "uk": "Ukrainian",
    "bo": "Tibetan", "kk": "Kazakh", "mn": "Mongolian", "yue": "Cantonese",
}


class BaseTranslator:
    name = ""
    model_id = ""
    batch_size = 16
    codes: dict = {}

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cpu"

    def supports(self, lang_code: str) -> bool:
        return lang_code in self.codes

    def load(self, device: str):
        raise NotImplementedError

    def _translate_batch(self, lines, src, tgt):
        raise NotImplementedError

    def translate(self, lines, src, tgt, log=print):
        """Translate lines; empty lines stay empty. src/tgt are Whisper codes."""
        out = [""] * len(lines)
        todo = [(i, t.strip()) for i, t in enumerate(lines) if t.strip()]
        for start in range(0, len(todo), self.batch_size):
            chunk = todo[start:start + self.batch_size]
            results = self._translate_batch([t for _, t in chunk], src, tgt)
            for (i, _), res in zip(chunk, results):
                out[i] = res.strip()
            log(f"Translated {min(start + self.batch_size, len(todo))}/{len(todo)} lines")
        return out

    def unload(self):
        self.model = None
        self.tokenizer = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class NLLBTranslator(BaseTranslator):
    name = "NLLB-200 600M"
    model_id = "facebook/nllb-200-distilled-600M"
    batch_size = 16
    codes = NLLB_CODES

    def load(self, device: str):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, dtype=dtype).to(device)
        self.model.eval()

    def _translate_batch(self, lines, src, tgt):
        import torch

        tok = self.tokenizer
        tok.src_lang = self.codes[src]
        inputs = tok(lines, return_tensors="pt", padding=True, truncation=True,
                     max_length=256).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                forced_bos_token_id=tok.convert_tokens_to_ids(self.codes[tgt]),
                max_new_tokens=256,
                num_beams=4,
            )
        return tok.batch_decode(out, skip_special_tokens=True)


class HyMT2Translator(BaseTranslator):
    name = "Hy-MT2 1.8B"
    model_id = "tencent/Hy-MT2-1.8B"
    batch_size = 8
    codes = HYMT_NAMES

    PROMPT = ("Translate the following text into {target_lang}. Note that you should "
              "only output the translated result without any additional explanation:\n\n{text}")

    def load(self, device: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.padding_side = "left"  # needed for batched generation
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=dtype).to(device)
        self.model.eval()

    def _translate_batch(self, lines, src, tgt):
        import torch

        tok = self.tokenizer
        convs = [[{"role": "user",
                   "content": self.PROMPT.format(target_lang=self.codes[tgt], text=t)}]
                 for t in lines]
        inputs = tok.apply_chat_template(convs, add_generation_prompt=True, padding=True,
                                         return_tensors="pt", return_dict=True).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.6,
                top_k=20,
                repetition_penalty=1.05,
                pad_token_id=tok.pad_token_id,
            )
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return tok.batch_decode(new_tokens, skip_special_tokens=True)


TRANSLATORS = {cls.name: cls for cls in (NLLBTranslator, HyMT2Translator)}
