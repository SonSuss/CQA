from transformers import AutoModel, AutoTokenizer


class VisionEncoder:
    def __init__(self, model_name):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs


# print(outputs)
vision_encoder_model = "google/siglip-so400m-patch14-384"
ViTEn = VisionEncoder(vision_encoder_model)