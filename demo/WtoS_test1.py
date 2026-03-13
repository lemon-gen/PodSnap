import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class SimpleQwenSummarizer:
    def __init__(self, model_name="Qwen/Qwen2.5-3B-Instruct"):
        print(f"🤖 Loading model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.model.eval()
        print("✅ Model loaded")
    
    def read_txt(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def summarize(self, text, max_length=1024):
        #短文本完整摘要
        prompt = f"""You are a professional podcast summarization assistant. Please generate a clear summary based on the following transcript.

Requirements:
1. Summarize the core theme in one sentence
2. List 3-5 key points
3. Extract 2-3 golden quotes
4. Conclusion within 100 words
5. Respond in English

Transcript:
{text}

Summary:"""
        
        messages = [
            {"role": "system", "content": "You are a professional podcast summarization assistant."},
            {"role": "user", "content": prompt}
        ]
        
        text_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text_input], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response
    
    def extract_chunk_info(self, text, max_length=1024):
        prompt = f"""You are a podcast content extractor. Please extract the following information from this transcript segment:

1. KEY_POINTS: List 2-4 key points or arguments mentioned (bullet points)
2. CANDIDATE_QUOTES: Extract 2-3 notable quotes verbatim from the text (use quotation marks)
3. TOPICS: List 1-3 main topics discussed in this segment

Transcript segment:
{text}

Output format (use English):
KEY_POINTS:
- ...
- ...

CANDIDATE_QUOTES:
- "..."
- "..."

TOPICS:
- ...
- ..."""
        
        messages = [
            {"role": "system", "content": "You are a podcast content extractor."},
            {"role": "user", "content": prompt}
        ]
        
        text_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text_input], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.5,  # 降低温度，提取更准确
                top_p=0.8,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    def synthesize_final_summary(self, all_chunks_info, max_length=1500):
        prompt = f"""You are a professional podcast summarization assistant. Below is extracted information from all segments of a podcast transcript.

Please create a COMPLETE final summary with:

1. 🎯 CORE THEME: One sentence summarizing the entire podcast
2. 💡 KEY POINTS: 3-5 most important points from the entire discussion (synthesize from all segments)
3. 💬 GOLDEN QUOTES: 2-3 BEST quotes from the entire podcast (select the most impactful ones from all candidate quotes)
4. 📝 CONCLUSION: A 100-word final summary

Extracted information from all segments:
{all_chunks_info}

Final Summary (in English):"""
        
        messages = [
            {"role": "system", "content": "You are a professional podcast summarization assistant."},
            {"role": "user", "content": prompt}
        ]
        
        text_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text_input], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    def save_summary(self, summary, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# 🎧 Podcast Summary\n\n{summary}")
        print(f"✅ Summary saved to: {output_path}")
    
    def process(self, txt_path, output_path="summary.md", chunk_threshold=2000):
        print("=" * 50)
        print("📝 Starting Processing")
        print("=" * 50)
        
        text = self.read_txt(txt_path)
        print(f"📄 Text loaded, length: {len(text)} chars")
        
        if len(text) > chunk_threshold:
            print(f"⚠️  Text exceeds {chunk_threshold} chars, using TWO-STAGE chunk strategy")
            summary = self._chunk_summarize_v2(text)  # 使用新版本
        else:
            summary = self.summarize(text)
        
        self.save_summary(summary, output_path)
        
        print("=" * 50)
        print("🎉 Processing Complete!")
        print("=" * 50)
        
        return summary
    
    def _chunk_summarize_v2(self, text, chunk_size=3000):
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        all_chunks_info = []
        
        print(f"📦 Stage 1: Extracting info from {len(chunks)} chunks")
        
        # === 第一阶段：提取各片段素材 ===
        for i, chunk in enumerate(chunks):
            print(f"  Extracting chunk {i+1}/{len(chunks)}...")
            chunk_info = self.extract_chunk_info(chunk)
            all_chunks_info.append(f"=== Segment {i+1} ===\n{chunk_info}")
        
        # === 第二阶段：全局合成 ===
        print(f"📦 Stage 2: Synthesizing final summary from all segments")
        combined_info = "\n\n".join(all_chunks_info)
        final_summary = self.synthesize_final_summary(combined_info)
        
        return final_summary


if __name__ == "__main__":
    summarizer = SimpleQwenSummarizer(model_name="Qwen/Qwen2.5-3B-Instruct")
    summarizer.process(
        txt_path="txt1.txt",  
        output_path="summary.md",
        chunk_threshold=2000  # 低于此值直接摘要，高于此值分块
    )
