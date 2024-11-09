from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader

class SmallScaleLLM:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate_text(self, prompt, max_length=50):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(inputs.input_ids, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def get_num_parameters(self):
        return sum(p.numel() for p in self.model.parameters())

    def get_input_dimension(self):
        return self.model.config.hidden_size
    
    def train(self, train_dataset, epochs=3, batch_size=8, learning_rate=1e-4):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        self.model.train()
        for epoch in range(epochs):
            for batch in train_loader:
                inputs = self.tokenizer(batch['text'], return_tensors="pt", padding=True, truncation=True).to(self.device)
                outputs = self.model(**inputs, labels=inputs.input_ids)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Example usage
model_name = "microsoft/phi-2"  # Replace with the actual model name if different
small_scale_llm = SmallScaleLLM(model_name)
