from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class SmallScaleLLM:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_text(self, prompt, max_length=50):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(inputs.input_ids, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
model_name = "microsoft/phi-2"  # Replace with the actual model name if different
small_scale_llm = SmallScaleLLM(model_name)

prompt = "Once upon a time"
generated_text = small_scale_llm.generate_text(prompt)
print(generated_text)