from pandas import DataFrame

from dataset_provider import search
from tqdm import tqdm
import clearml_poc
import pandas as pd



def get_next_batch():
    after_key = None
    query = {
        "size": 0,
        "query": {
            "script_score": {
                "query": {
                    "match_all" : {}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding.llama_3_17_v_proj.ne_embedding') + 1.0",
                    "params": {
                        "query_vector":  [
                -0.21911706030368805,
                0.06572527438402176,
                0.27068382501602173,
                0.09131937474012375,
                0.14489173889160156,
                0.10690806061029434,
                0.02492649480700493,
                -0.2336108535528183,
                0.008711825124919415,
                0.09875797480344772,
                -0.054535768926143646,
                0.1428990364074707,
                0.25854480266571045,
                -0.2371070832014084,
                -0.0834948867559433,
                0.05490997061133385,
                0.03845496103167534,
                -0.05195046216249466,
                -0.08760710805654526,
                0.15429720282554626,
                0.11065942794084549,
                0.10722954571247101,
                -0.09524950385093689,
                0.15610608458518982,
                -0.12422475963830948,
                0.13242237269878387,
                -0.2110743671655655,
                -0.04206418618559837,
                0.08451423048973083,
                -0.2514849007129669,
                -0.19141964614391327,
                0.14886336028575897,
                -0.026451505720615387,
                -0.17146261036396027,
                -0.16952785849571228,
                -0.020440462976694107,
                0.09287111461162567,
                -0.022727331146597862,
                0.0047651417553424835,
                0.2736967206001282,
                -0.10234192758798599,
                0.02350013703107834,
                0.045262034982442856,
                -0.04395941272377968,
                -0.055646806955337524,
                0.08459554612636566,
                0.2200339287519455,
                -0.2800924479961395,
                0.2502889633178711,
                -0.17819148302078247
              ]
                    }
                }
            }
        },
        "aggs": {
            "grouped_results": {
                "composite": {
                    "size": 1000,
                    "sources": [
                        {
                            "text_id": {
                                "terms": {
                                    "field": "text_id"
                                }
                            }
                        }
                    ]

                },
                "aggs": {
                    "top_hit": {
                        "top_hits": {
                            "size": 1,
                            "_source": {
                                "includes": ["text_id",
                                             "doc_id",
                                             "all_text", "coarse_type", "fine_type", "phrase", "_score"]
                            },
                            "sort": [
                                {
                                    "_score": {
                                        "order": "desc"
                                    }
                                }
                            ]
                        }
                    }
                }
            }
        }
    }
    if after_key:
        query["aggs"]["grouped_results"]["composite"]["after"] = after_key
    index = "fewnerd_v4_*"
    response = search(index=index, query=query)

    while "after_key" in response["aggregations"]["grouped_results"]:
        after_key = response["aggregations"]["grouped_results"]["after_key"]
        for key in response["aggregations"]["grouped_results"]["buckets"]:
            yield key["top_hit"]["hits"]["hits"][0]
        query["aggs"]["grouped_results"]["composite"]["after"] = after_key
        response = search(index=index, query=query)

    # after_key = response["aggregations"]["grouped_results"]["after_key"]\
    #     if "after_key" in response["aggregations"]["grouped_results"] else None
    #
    # for key in response["aggregations"]["grouped_results"]["buckets"]:
    #     yield key["top_hit"]["hits"]["hits"][0]["_source"]
    #





def main():
    data = {
        "score": [],
        "text_id": [],
        "doc_id": [],
        "fine_type": [],
        "coarse_type": [],
        "all_text": [],
        "phrase": [],
    }
    for item in get_next_batch():
        pbar.update(1)
        data["score"].append(item["_score"])
        data["text_id"].append(item["_source"]["text_id"])
        data["doc_id"].append(item["_source"]["doc_id"])
        data["fine_type"].append(item["_source"]["fine_type"])
        data["coarse_type"].append(item["_source"]["coarse_type"])
        data["all_text"].append(item["_source"]["all_text"])
        data["phrase"].append(item["_source"]["phrase"])


    df = pd.DataFrame(data)
    DataFrame.to_csv(df, "resullts.csv")
    clearml_poc.register_artifact(artifact=df, name="resullts")




if __name__ == "__main__":
    pbar = tqdm()
    clearml_poc.clearml_init()
    main()
