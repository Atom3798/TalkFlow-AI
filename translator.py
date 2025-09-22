from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class M2MTranslator:
    """
    Multilingual translation using facebook/m2m100_418M.
    """
    def __init__(self, device: str = "auto"):
        model_name = "facebook/m2m100_418M"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
            self.model = self.model.to("cuda")
            self.device = "cuda"
        else:
            self.device = "cpu"

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        if not text:
            return text
        # Normalize language tag
        src = (src_lang or "en").split("-")[0].lower()
        tgt = (tgt_lang or "en").split("-")[0].lower()

        # m2m expects explicit source language
        self.tokenizer.src_lang = src

        encoded = self.tokenizer(text, return_tensors="pt")
        if self.device == "cuda":
            encoded = {k: v.to("cuda") for k, v in encoded.items()}

        generated_tokens = self.model.generate(
            **encoded,
            forced_bos_token_id=self.tokenizer.get_lang_id(tgt),
            max_new_tokens=256
        )
        out = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        return out
