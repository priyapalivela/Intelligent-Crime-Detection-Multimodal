"""
main.py — FastAPI wrapper for Crime Severity Detection v1.4.0
Includes attention-based explainability from Cell 52 MultimodalExplainer

Run:  python main.py
Open: http://127.0.0.1:8000/docs
"""

import os, sys, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertModel, DistilBertTokenizer

try:    PROJECT_ROOT = Path(__file__).resolve().parent
except: PROJECT_ROOT = Path.cwd()
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pt"

# ══ EXACT MODEL — Crime_Detection.ipynb Cell 41-43 ════════════════════════════

class AudioEncoder(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128):
        super().__init__()
        self.conv1=nn.Conv1d(input_dim,64,3,padding=1); self.bn1=nn.BatchNorm1d(64)
        self.conv2=nn.Conv1d(64,128,3,padding=1);       self.bn2=nn.BatchNorm1d(128)
        self.pool=nn.MaxPool1d(2); self.dropout=nn.Dropout(0.3)
        self.lstm=nn.LSTM(128,hidden_dim,1,batch_first=True,bidirectional=True)
        self.layer_norm=nn.LayerNorm(hidden_dim*2); self.out_dim=hidden_dim*2
    def forward(self,x):
        if x.shape[1]!=40: x=x.transpose(1,2)
        x=F.relu(self.bn1(self.conv1(x))); x=self.pool(x)
        x=F.relu(self.bn2(self.conv2(x))); x=self.dropout(x)
        x=x.transpose(1,2); out,_=self.lstm(x)
        return self.layer_norm(out.mean(dim=1))

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        print("[API] Loading DistilBERT...")
        self.distilbert=DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.out_dim=768
    def forward(self,input_ids,output_attentions=False):
        mask=( input_ids!=0).long()
        out=self.distilbert(input_ids=input_ids,attention_mask=mask,output_attentions=output_attentions)
        pooled=out.last_hidden_state.mean(dim=1)
        return (pooled,out.attentions) if output_attentions else pooled

class MultimodalFusionModel(nn.Module):
    def __init__(self,embed_dim=256,num_classes=3):
        super().__init__()
        self.audio_encoder=AudioEncoder(); self.text_encoder=TextEncoder()
        self.audio_proj=nn.Linear(self.audio_encoder.out_dim,embed_dim)
        self.text_proj=nn.Linear(self.text_encoder.out_dim,embed_dim)
        self.classifier=nn.Linear(embed_dim*2,num_classes)
    def forward(self,audio,text,output_attentions=False):
        af=self.audio_encoder(audio)
        if output_attentions: tf,attn=self.text_encoder(text,True)
        else: tf=self.text_encoder(text); attn=None
        ae=F.relu(self.audio_proj(af)); te=F.relu(self.text_proj(tf))
        logits=self.classifier(torch.cat([ae,te],dim=1))
        return (logits,ae,te,attn) if output_attentions else (logits,ae,te)
    @staticmethod
    def final_severity(a,t):
        if a==2 or t==2: return 2
        if a==1 or t==1: return 1
        return 0

# ══ MULTIMODAL EXPLAINER — Cell 52 adapted for API ════════════════════════════

class MultimodalExplainer:
    """
    Attention-based explainability.
    Extracts DistilBERT attention weights (6 layers × 12 heads),
    averages them, and assigns importance score to each input word.
    """
    def __init__(self,tokenizer): self.tokenizer=tokenizer

    @torch.no_grad()
    def explain(self,model,audio_t,text_t):
        model.eval()
        logits,audio_emb,text_emb,attentions=model(audio_t,text_t,output_attentions=True)
        probs=F.softmax(logits,dim=1)[0]
        pred=probs.argmax().item()

        # Modality contribution (Cell 52 logic)
        a_imp=audio_emb.norm(dim=1).mean().item()
        t_imp=text_emb.norm(dim=1).mean().item()
        tot=a_imp+t_imp+1e-8

        # Attention averaging: 6 layers × 12 heads → per-token score
        stacked=torch.stack(attentions,dim=0)          # [6,1,12,seq,seq]
        avg=stacked.mean(dim=[0,1,2])                   # [seq,seq]
        token_imp=avg.mean(dim=0)                       # [seq]

        # Token → word mapping
        ids=text_t[0].tolist()
        tokens=self.tokenizer.convert_ids_to_tokens(ids)
        word_scores=[]
        for tok,sc in zip(tokens,token_imp.tolist()):
            if tok in ('[CLS]','[SEP]','[PAD]'): continue
            word_scores.append({"word":tok.replace('##',''),"score":round(sc,4),"pct":round(sc*100,1)})

        top5=sorted(word_scores,key=lambda x:x["score"],reverse=True)[:5]

        return {
            "predicted_class": pred,
            "predicted_label": ["Low","Medium","High"][pred],
            "confidence":      round(probs[pred].item(),4),
            "probabilities":   {"Low":round(probs[0].item(),4),"Medium":round(probs[1].item(),4),"High":round(probs[2].item(),4)},
            "modality_contribution": {
                "audio_pct": round(a_imp/tot*100,1),
                "text_pct":  round(t_imp/tot*100,1),
                "dominant":  "Audio" if a_imp>t_imp else "Text"
            },
            "word_attention_scores": word_scores,
            "top_influential_words": top5,
            "method": "Attention-based — averaged across 6 DistilBERT layers × 12 heads",
        }

# ══ CONSTANTS ════════════════════════════════════════════════════════════════

TEXT_SEVERITY_MAPPING={
    "HOMICIDE":2,"CRIMINAL SEXUAL ASSAULT":2,"ROBBERY":2,"BATTERY":2,"ASSAULT":2,
    "STALKING":2,"BURGLARY":2,"MOTOR VEHICLE THEFT":2,"ARSON":2,"HUMAN TRAFFICKING":2,
    "KIDNAPPING":2,"WEAPONS VIOLATION":2,"DECEPTIVE PRACTICE":1,"CRIMINAL DAMAGE":1,
    "CRIMINAL TRESPASS":1,"PROSTITUTION":1,"OFFENSE INVOLVING CHILDREN":1,"SEX OFFENSE":1,
    "GAMBLING":1,"NARCOTICS":1,"OTHER NARCOTIC VIOLATION":1,"LIQUOR LAW VIOLATION":1,
    "INTERFERENCE WITH PUBLIC OFFICER":1,"INTIMIDATION":1,"PUBLIC PEACE VIOLATION":0,
    "NON-CRIMINAL":0,"OBSCENITY":0,"PUBLIC INDECENCY":0,"OTHER OFFENSE":0,
}
AUDIO_SEVERITY={"gun_shot":2,"siren":2,"drilling":2,"engine_idling":2,"car_horn":1,"dog_bark":1,"jackhammer":1,"children_playing":0,"street_music":0,"air_conditioner":0}
SEVERITY_LABELS={0:"Low",1:"Medium",2:"High"}
SEVERITY_COLORS={0:"#4CAF50",1:"#FF9800",2:"#F44336"}
MEASURES={0:["Log incident","Continue monitoring","Routine patrol check"],1:["Increase surveillance","Alert security team","Notify local authorities","Document evidence"],2:["Immediate response required","Alert police/emergency services","Evacuate area if safe","Activate full emergency protocol","Preserve crime scene"]}
MFCC_PATTERNS={"gun_shot":{"mean":2.5,"std":1.8},"siren":{"mean":1.8,"std":1.2},"drilling":{"mean":1.5,"std":1.0},"engine_idling":{"mean":1.2,"std":0.8},"car_horn":{"mean":0.8,"std":0.6},"dog_bark":{"mean":0.6,"std":0.5},"jackhammer":{"mean":1.0,"std":0.7},"children_playing":{"mean":-0.5,"std":0.4},"street_music":{"mean":-0.3,"std":0.3},"air_conditioner":{"mean":-0.8,"std":0.2}}

# ══ LOAD MODEL ════════════════════════════════════════════════════════════════
print("[API] Initializing...")
device=torch.device("cpu")
tokenizer=DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model=MultimodalFusionModel(256,3).to(device)
explainer=MultimodalExplainer(tokenizer)
if MODEL_PATH.exists():
    model.load_state_dict(torch.load(MODEL_PATH,map_location="cpu",weights_only=True))
    print(f"[API] ✅ Loaded {MODEL_PATH}")
else:
    print("[API] ⚠️  No weights found — demo mode")
model.eval()
print("[API] ✅ Ready! v1.4.0")

# ══ FASTAPI ═══════════════════════════════════════════════════════════════════
app=FastAPI(
    title="Crime Severity Detection API",
    description="""
## Multimodal Crime Severity Detection API v1.4.0

**NEW:** `POST /explain` — Attention-based explainability showing which words drove the prediction.

### Architecture
- CNN-BiLSTM AudioEncoder + DistilBERT TextEncoder
- Conservative late fusion: max(audio_pred, text_pred)
- **Accuracy: 88.29% | F1: 0.86+ | F2: ~0.88**

### Explainability Method
Attention Weight Visualization — averages DistilBERT's internal attention matrices
across 6 transformer layers × 12 heads to score each input word's influence.
""",version="1.4.0")

# ══ SCHEMAS ═══════════════════════════════════════════════════════════════════
class PredictionRequest(BaseModel):
    audio_class: str
    description: str
    class Config:
        json_schema_extra={"example":{"audio_class":"gun_shot","description":"ROBBERY"}}

class ExplainRequest(BaseModel):
    audio_class: str
    description: str
    class Config:
        json_schema_extra={"example":{"audio_class":"gun_shot","description":"armed robbery suspect near ATM weapons violation"}}

class WordScore(BaseModel):
    word:str; score:float; pct:float

class ModalityDetail(BaseModel):
    severity_label:str; severity_code:int; confidence:float; probabilities:dict

class PredictionResponse(BaseModel):
    audio_modality:ModalityDetail; text_modality:ModalityDetail
    final_severity:str; final_severity_code:int; fusion_rule:str
    probabilities:dict; recommended_actions:List[str]; color:str; model_info:dict

class ExplainResponse(BaseModel):
    predicted_label:str; predicted_class:int; confidence:float; probabilities:dict
    modality_contribution:dict; word_attention_scores:List[WordScore]
    top_influential_words:List[WordScore]; fusion_rule:str
    recommended_actions:List[str]; color:str; method:str

# ══ HELPER ════════════════════════════════════════════════════════════════════
def build_tensors(audio_class,description):
    p=MFCC_PATTERNS.get(audio_class,{"mean":0.0,"std":0.5})
    np.random.seed(hash(audio_class)%2**32)
    mfcc=np.random.normal(p["mean"],p["std"],(40,100)).astype(np.float32)
    audio_t=torch.tensor(mfcc).unsqueeze(0)
    tok=tokenizer(description.upper().strip(),padding="max_length",truncation=True,max_length=64,return_tensors="pt")
    return audio_t,tok["input_ids"].long()

# ══ ENDPOINTS ═════════════════════════════════════════════════════════════════
@app.get("/")
def root():
    return {"message":"Crime Severity Detection API v1.4.0","accuracy":"88.29%",
            "new":"POST /explain — attention-based word importance","docs":"http://127.0.0.1:8000/docs",
            "dashboard":"https://priyapalivela-crime-detection-dashboard.hf.space","author":"Bhanu Priya Palivela"}

@app.get("/health")
def health():
    return {"status":"healthy","model_loaded":MODEL_PATH.exists(),"version":"1.4.0"}

@app.get("/audio-classes")
def audio_classes():
    return {"total":len(AUDIO_SEVERITY),"classes":{"High":["gun_shot","siren","drilling","engine_idling"],"Medium":["car_horn","dog_bark","jackhammer"],"Low":["children_playing","street_music","air_conditioner"]}}

@app.get("/text-categories")
def text_categories():
    return {"total":len(TEXT_SEVERITY_MAPPING),"categories":TEXT_SEVERITY_MAPPING}

@app.post("/predict",response_model=PredictionResponse)
@torch.no_grad()
def predict(request:PredictionRequest):
    ac=request.audio_class.lower().strip()
    if ac not in AUDIO_SEVERITY: raise HTTPException(400,f"Invalid audio_class. Valid: {list(AUDIO_SEVERITY.keys())}")
    if not request.description.strip(): raise HTTPException(400,"description cannot be empty")
    at,tt=build_tensors(ac,request.description)
    logits,ae,te=model(at,tt)
    probs=F.softmax(logits,dim=1)[0]
    pd_={k:round(probs[i].item(),4) for i,k in enumerate(["Low","Medium","High"])}
    af=model.audio_encoder(at); aemb=F.relu(model.audio_proj(af))
    al=model.classifier(torch.cat([aemb,torch.zeros_like(aemb)],dim=1)); apr=F.softmax(al,dim=1)[0]; apred=al.argmax(dim=1).item()
    tf=model.text_encoder(tt); temb=F.relu(model.text_proj(tf))
    tl=model.classifier(torch.cat([torch.zeros_like(temb),temb],dim=1)); tpr=F.softmax(tl,dim=1)[0]; tpred=tl.argmax(dim=1).item()
    final=MultimodalFusionModel.final_severity(apred,tpred)
    return PredictionResponse(
        audio_modality=ModalityDetail(severity_label=SEVERITY_LABELS[apred],severity_code=apred,confidence=round(apr[apred].item(),4),probabilities={k:round(apr[i].item(),4) for i,k in enumerate(["Low","Medium","High"])}),
        text_modality=ModalityDetail(severity_label=SEVERITY_LABELS[tpred],severity_code=tpred,confidence=round(tpr[tpred].item(),4),probabilities={k:round(tpr[i].item(),4) for i,k in enumerate(["Low","Medium","High"])}),
        final_severity=SEVERITY_LABELS[final],final_severity_code=final,
        fusion_rule="conservative_max — max(audio_pred, text_pred)",
        probabilities=pd_,recommended_actions=MEASURES[final],color=SEVERITY_COLORS[final],
        model_info={"architecture":"CNN-BiLSTM + DistilBERT","accuracy":"88.29%","dataset_audio":"UrbanSound8K","dataset_text":"Chicago PD IUCR"})

@app.post("/explain",response_model=ExplainResponse)
def explain(request:ExplainRequest):
    """
    Attention-based explainability — shows WHICH words drove the prediction.

    Extracts DistilBERT's internal attention weights averaged across:
    - 6 transformer layers  
    - 12 attention heads per layer

    Returns word-level importance scores showing what the model focused on.
    Example: 'armed[0.8] robbery[0.9] near[0.1] ATM[0.3]'
    """
    ac=request.audio_class.lower().strip()
    if ac not in AUDIO_SEVERITY: raise HTTPException(400,f"Invalid audio_class. Valid: {list(AUDIO_SEVERITY.keys())}")
    if not request.description.strip(): raise HTTPException(400,"description cannot be empty")
    at,tt=build_tensors(ac,request.description)
    result=explainer.explain(model,at,tt)
    audio_sev=AUDIO_SEVERITY[ac]
    final=MultimodalFusionModel.final_severity(audio_sev,result["predicted_class"])
    return ExplainResponse(
        predicted_label=result["predicted_label"],predicted_class=result["predicted_class"],
        confidence=result["confidence"],probabilities=result["probabilities"],
        modality_contribution=result["modality_contribution"],
        word_attention_scores=[WordScore(**w) for w in result["word_attention_scores"]],
        top_influential_words=[WordScore(**w) for w in result["top_influential_words"]],
        fusion_rule="conservative_max — max(audio_pred, text_pred)",
        recommended_actions=MEASURES[final],color=SEVERITY_COLORS[final],method=result["method"])

@app.post("/predict/batch")
@torch.no_grad()
def predict_batch(requests:List[PredictionRequest]):
    if len(requests)>10: raise HTTPException(400,"Max 10 per batch")
    results=[]
    for req in requests:
        try:
            at,tt=build_tensors(req.audio_class.lower(),req.description)
            logits,_,_=model(at,tt); probs=F.softmax(logits,dim=1)[0]; pred=probs.argmax().item()
            results.append({"audio_class":req.audio_class,"description":req.description,"final_severity":SEVERITY_LABELS[pred],"confidence":round(probs[pred].item(),4),"color":SEVERITY_COLORS[pred]})
        except Exception as e:
            results.append({"error":str(e),"audio_class":req.audio_class})
    return {"total":len(results),"predictions":results}

if __name__=="__main__":
    import uvicorn
    uvicorn.run("main:app",host="0.0.0.0",port=8000,reload=False)