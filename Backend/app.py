import os
from datetime import datetime
import traceback
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset, concatenate_datasets

# Configuration
os.environ["HF_HOME"] = "/app/cache"
os.environ["XDG_CACHE_HOME"] = "/app/cache"
os.environ["OMP_NUM_THREADS"] = "1"  # Optimize for CPU
os.environ["MKL_NUM_THREADS"] = "1"
os.makedirs("/app/cache", exist_ok=True)
os.makedirs("/app/finetuned", exist_ok=True)
torch.set_num_threads(1)  # Better CPU utilization

app = FastAPI()

# Middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["mahdee987-financial-chatbot.hf.space", "localhost"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Model loading with warmup
try:
    model_name = "distilgpt2"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32
    )
    
    # Warmup the model
    @app.on_event("startup")
    async def warmup_model():
        try:
            print("Warming up model...")
            dummy_input = tokenizer("warmup", return_tensors="pt")
            model.generate(**dummy_input, max_length=1)
            print("Model warmup complete")
        except Exception as e:
            print(f"Warmup failed: {str(e)}")

except Exception as e:
    print(f"Model loading failed: {str(e)}")
    raise

# Request models
class Query(BaseModel):
    message: str = Field(..., max_length=500)

class FineTuneRequest(BaseModel):
    epochs: int = Field(1, gt=0, le=3)
    learning_rate: float = Field(5e-5, gt=0, le=1e-3)
    samples_per_dataset: int = Field(100, gt=10, le=500)  # Reduced max samples

# Middleware to validate requests
@app.middleware("http")
async def validate_requests(request: Request, call_next):
    if request.url.path == "/chat" and request.method != "POST":
        raise HTTPException(
            status_code=400,
            detail="Only POST requests are allowed for this endpoint"
        )
    return await call_next(request)

# Endpoints
@app.post("/chat")
async def chat(query: Query):
    try:
        current_model = model
        if os.path.exists("/app/finetuned/adapter_config.json"):
            current_model = PeftModel.from_pretrained(model, "/app/finetuned")
        
        prompt = f"Question: {query.message}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        outputs = current_model.generate(
            **inputs,
            max_new_tokens=150,  # Reasonable upper limit
            temperature=0.7,
            do_sample=True,
            no_repeat_ngram_size=2,
            eos_token_id=tokenizer.eos_token_id
        )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response.split("Answer:")[-1].strip()
        
        # Simple sentence completion
        for end in ['.', '!', '?']:
            if end in response:
                response = response[:response.rfind(end)+1]
                break
                
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat")
async def chat_get():
    """Helper endpoint to guide users"""
    return {
        "error": "Invalid method",
        "solution": "Send a POST request with JSON body: {'message':'your question'}"
    }

@app.post("/fine-tune")
async def fine_tune(params: FineTuneRequest):
    """Fine-tuning endpoint"""
    try:
        # Load datasets
        print("Starting fine tuning..")
        alpaca = load_dataset("gbharti/finance-alpaca", split=f"train[:{params.samples_per_dataset}]")
        fiqa = load_dataset("bilalRahib/fiqa-personal-finance-dataset", "full", split=f"train[:{params.samples_per_dataset}]")

        # Process datasets
        def process_example(ex):
            return {
                "text": (
                    f"Instruction: {ex['instruction']}\nOutput: {ex['output']}" 
                    if 'instruction' in ex else 
                    f"Question: {ex['question']}\nAnswer: {ex['answer']}"
                )
            }

        dataset = concatenate_datasets([
            alpaca.map(process_example),
            fiqa.map(process_example)
        ]).shuffle(seed=42)
        
        # Tokenize
        dataset = dataset.map(
            lambda x: tokenizer(
                x["text"],
                truncation=True,
                max_length=96,
                padding="max_length"
            ),
            batched=True,
            batch_size=8
        )

        # LoRA config
        peft_config = LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["c_attn"],
            lora_dropout=0.05,
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, peft_config)
        
        # Training
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir="/app/finetuned",
                per_device_train_batch_size=1,
                num_train_epochs=params.epochs,
                learning_rate=params.learning_rate,
                logging_steps=10,
                save_strategy="epoch",
                optim="adamw_torch",
                gradient_checkpointing=True,
                gradient_accumulation_steps=8,
                fp16=False
            ),
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
        )
        
        trainer.train()
        print("Training completed, saving model...")
        model.save_pretrained("/app/finetuned")
        print("Model saved!")  # Debug log
        return {
            "status": "success",
            "trained_samples": len(dataset),
            "training_time": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Fine-tuning failed: {e}")  # Debug log
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "model": model_name,
        "device": "cpu",
        "fine_tuned": os.path.exists("/app/finetuned/adapter_config.json"),
        "torch_version": torch.__version__
    }

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Financial Chatbot",
        "endpoints": {
            "/chat": "POST for chat responses",
            "/fine-tune": "POST for model fine-tuning",
            "/health": "GET service health check"
        }
    }
