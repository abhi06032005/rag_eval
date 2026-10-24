"""Measure query retrieval including on-demand neural encoding; never includes generation."""
import argparse,gzip,json,pickle,time
import numpy as np
import psutil
import torch
from sentence_transformers import SentenceTransformer
from scripts.full_beir_reevaluation import OUT,dump,load_data,metrics
from ragbench.retrieval.policies import BM25Policy,DensePolicy,FixedHybridPolicy,AdaptiveRouterPolicy,AdaptiveThresholds

def main(name):
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(42);np.random.seed(42)
    root=OUT/name; model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41',device='cpu',local_files_only=True)
    vectors=np.load(root/'corpus_embeddings.npy');ids=json.loads((root/'corpus_ids.json').read_text());qids=json.loads((root/'query_ids.json').read_text());_,queries,splits=load_data(name)
    with (root/'bm25_index.pkl').open('rb') as f:bm=pickle.load(f)
    calls=0
    def embed(text):
        nonlocal calls
        calls+=1
        return model.encode(text,normalize_embeddings=True,show_progress_bar=False)
    def dense(vec,k):
        scores=vectors@vec
        order=np.argsort(-scores,kind='stable')[:k]
        return [(ids[i],float(scores[i])) for i in order]
    policies={'BM25':BM25Policy(bm),'Dense_Exact':DensePolicy(dense,embed_query_fn=embed),'Fixed_Hybrid':FixedHybridPolicy(bm,dense,embed_query_fn=embed,candidate_depth=250),'Adaptive_Router':AdaptiveRouterPolicy(bm,dense,embed_query_fn=embed,candidate_depth=250)}
    for p in policies.values():p.retrieve(queries[qids[0]],top_k=250)
    records=[]; labels=list(policies)
    for i,q in enumerate(qids):
        order=labels[i%4:]+labels[:i%4]
        for label in order:
            before=calls;start=time.perf_counter();r=policies[label].retrieve(queries[q],top_k=250);ms=(time.perf_counter()-start)*1000
            score=metrics([d for d,s in r.retrieved_docs],splits['test'][q])
            records.append({'query_id':q,'system':label,'rankings':r.retrieved_docs[:10],'metrics':{k:v for k,v in score.items() if k!='MRR@250'},'latency_ms':ms,'actual_neural_calls':calls-before,'route':r.route_taken})
    with gzip.open(root/'query_latency_per_query.jsonl.gz','wt',encoding='utf-8') as f:
        for r in records:f.write(json.dumps(r)+'\n')
    summary=[]
    for label in labels:
        rows=[r for r in records if r['system']==label];lat=[r['latency_ms'] for r in rows]
        summary.append({'dataset':name,'system':label,'candidate_depth':250,'n':len(rows),'p50_ms':float(np.percentile(lat,50)),'p95_ms':float(np.percentile(lat,95)),'p99_ms':float(np.percentile(lat,99)),'serial_query_qps':1000/float(np.mean(lat)),'actual_neural_calls':sum(r['actual_neural_calls'] for r in rows),'process_rss_bytes':psutil.Process().memory_info().rss,'metrics':{k:float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']},'scope':'NEW_EXECUTION; query embedding + local retrieval/fusion; no HTTP or generation; one warm serial pass, rotating policy order, no embedding cache'})
    dump(root/'query_latency.json',summary);print(name,'query latency pass complete',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);main(p.parse_args().dataset)
