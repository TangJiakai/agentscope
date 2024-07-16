# The following code is adopted from RETA-LLM: https://github.com/RUC-GSAI/YuLan-IR/blob/main/RETA-LLM/dense_model.py

import torch
from torch import Tensor
import torch.nn.functional as F
from adapters import BertAdapterModel
from transformers import AutoConfig, AutoTokenizer
from tqdm import tqdm
from loguru import logger
import faiss
import numpy as np


SIMILARITY_METRIC_IP = "ip"
SIMILARITY_METRIC_COS = "cos"
SIMILARITY_METRICS = [SIMILARITY_METRIC_IP, SIMILARITY_METRIC_COS]

POOLING_AVERAGE = "average"
POOLING_CLS = "cls"
BATCH_SIZE = 4
POOLING_METHODS = [POOLING_AVERAGE, POOLING_CLS]

def extract_text_embed(
        last_hidden_state: Tensor, 
        attention_mask: Tensor, 
        similarity_metric: str, 
        pooling: str
    ):
    if pooling == POOLING_CLS:
        text_embeds = last_hidden_state[:, 0]
    elif pooling == POOLING_AVERAGE:
        masked_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.)
        text_embeds = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    else:
        raise NotImplementedError('pooling method not implemented')
    if similarity_metric == SIMILARITY_METRIC_IP:
        pass
    elif similarity_metric == SIMILARITY_METRIC_COS:
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
    else:
        raise NotImplementedError('similarity metric not implemented')
    return text_embeds

class BertDense(BertAdapterModel):
    def forward(self, input_ids, attention_mask, token_type_ids=None, position_ids=None, return_dict=False):
        outputs = self.bert(
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            token_type_ids = token_type_ids, 
            position_ids = position_ids,
            return_dict = True,
        )
        pooling = getattr(self.config, "pooling")
        similarity_metric = getattr(self.config, "similarity_metric")
        text_embeds = extract_text_embed(
            last_hidden_state = outputs.last_hidden_state, 
            attention_mask = attention_mask,
            similarity_metric = similarity_metric, 
            pooling = pooling,
        )
        if return_dict:
            outputs.embedding = text_embeds
            return outputs
        else:
            return text_embeds
        
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = self.texts[index]
        tokenized_text = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length)
        input_ids = torch.tensor(tokenized_text["input_ids"])
        attention_mask = torch.tensor(tokenized_text["attention_mask"])
        return input_ids, attention_mask

    @staticmethod
    def emb_collate_fn(batch):
        input_ids = [item[0] for item in batch]
        attention_masks = [item[1] for item in batch]
        input_ids = torch.stack(input_ids, dim=0)
        attention_masks = torch.stack(attention_masks, dim=0)
        return input_ids, attention_masks


def build_embedding_model(configs):
    config = AutoConfig.from_pretrained(configs["emb_model_path"])
    config.similarity_metric, config.pooling = "ip", "average"
    tokenizer = AutoTokenizer.from_pretrained(configs["emb_model_path"], config=config)
    model = BertDense.from_pretrained(configs["emb_model_path"], config=config)
    adapter_name = model.load_adapter(configs["adapter_path"])
    model.set_active_adapters(adapter_name)
    logger.info("Successfully build the model")
    return model, tokenizer

# calculate text embeddings for seekers and jobs
def calculate_embeddings(configs, seeker_agents, job_agents):
    # only calculate text embeddings for new agents
    need_embs_seeker_agents, need_embs_job_agents = [], []
    for seeker_agent in seeker_agents:
        if seeker_agent.seeker.emb is None:
            need_embs_seeker_agents.append(seeker_agent)
    for job_agent in job_agents:
        if job_agent.job.emb is None:
            need_embs_job_agents.append(job_agent)

    seeker_cv_texts, job_texts = [], []
    for seeker_agent in need_embs_seeker_agents:
        text = str(seeker_agent.seeker.cv) + "\n" + str(seeker_agent.seeker.trait)
        seeker_cv_texts.append(text)
    for job_agent in need_embs_job_agents:
        text = job_agent.job.jd + "\n" + str(job_agent.job.jr)
        job_texts.append(text)
    
    # init embedding model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_model, tokenizer = build_embedding_model(configs)
    emb_model.to(device)
    emb_model.eval()
    seeker_embs, job_embs = [], []
    # calculate seekers' embeddings
    if len(seeker_cv_texts) > 0:
        seeker_dataset = TextDataset(seeker_cv_texts, tokenizer)
        seeker_loader = torch.utils.data.DataLoader(seeker_dataset, batch_size=BATCH_SIZE, collate_fn=TextDataset.emb_collate_fn)
        for batch in tqdm(seeker_loader):
            with torch.no_grad():
                output = emb_model(input_ids=batch[0].to(device), attention_mask = batch[1].to(device))
            seeker_embs.append(output)
        seeker_embs = torch.cat(seeker_embs, dim=0).cpu().numpy()
        for i in range(len(need_embs_seeker_agents)):
            need_embs_seeker_agents[i].seeker.emb = seeker_embs[i].tolist()

    # calculate jobs' embeddings 
    if len(job_texts) > 0:
        job_dataset = TextDataset(job_texts, tokenizer)
        job_loader = torch.utils.data.DataLoader(job_dataset, batch_size=BATCH_SIZE, collate_fn=TextDataset.emb_collate_fn)
        for batch in tqdm(job_loader):
            with torch.no_grad():
                output = emb_model(input_ids = batch[0].to(device), attention_mask = batch[1].to(device))
            job_embs.append(output)
        job_embs = torch.cat(job_embs, dim=0).cpu().numpy()
        for i in range(len(need_embs_job_agents)):
            need_embs_job_agents[i].job.emb = job_embs[i].tolist()
        
    logger.info("Finish calculating text embeddings.")


def build_dense_index(configs, seeker_agents, job_agents):
    calculate_embeddings(configs, seeker_agents, job_agents)
    # build faiss index
    hidden_dim = len(seeker_agents[0].seeker.emb)
    job_embs = []
    for job_agent in job_agents:
        job_embs.append(job_agent.job.emb)
    job_embs = np.array(job_embs)   
    job_dense_index = faiss.IndexFlatL2(hidden_dim)
    job_dense_index.add(job_embs)
    logger.info("Finish building faiss index.")
    return job_dense_index