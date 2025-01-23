import asyncio
from collections import defaultdict

from pandas import DataFrame

import dataset_provider
from tqdm import tqdm
import clearml_poc
import pandas as pd


async def get_next_batch():
    query = {
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding.llama_3_17_v_proj') + 1.0",
                    "params": {
                        "query_vector": [
              0.0361328125,
              0.046630859375,
              -0.30859375,
              0.2294921875,
              -0.35546875,
              -1.03125,
              -0.1005859375,
              -0.078125,
              0.28125,
              0.48046875,
              -0.275390625,
              -0.1376953125,
              -0.087890625,
              0.40625,
              -0.171875,
              -0.126953125,
              0.265625,
              -0.185546875,
              0.58984375,
              -0.341796875,
              0.341796875,
              0.431640625,
              0.50390625,
              -0.7734375,
              -0.60546875,
              -0.107421875,
              0.390625,
              -0.34375,
              -0.087890625,
              0.78125,
              0.2314453125,
              -0.1689453125,
              -0.353515625,
              0.31640625,
              -0.828125,
              -0.55078125,
              -0.0223388671875,
              -0.41796875,
              0.51171875,
              0.2578125,
              0.0076904296875,
              0.486328125,
              -0.107421875,
              -0.006378173828125,
              0.1328125,
              -0.019287109375,
              -0.08935546875,
              -0.259765625,
              -0.640625,
              -0.3046875,
              0.02392578125,
              0.69921875,
              0.39453125,
              0.58984375,
              0.3828125,
              0.11474609375,
              0.1220703125,
              0.2236328125,
              0.193359375,
              -0.458984375,
              -0.25,
              -0.150390625,
              0.01422119140625,
              -0.40625,
              0.765625,
              -0.01019287109375,
              0.36328125,
              -0.1806640625,
              0.0791015625,
              -0.0703125,
              0.138671875,
              -0.294921875,
              0.36328125,
              -0.0576171875,
              -0.208984375,
              -0.19140625,
              0.1298828125,
              -0.51171875,
              0.171875,
              -0.56640625,
              -0.00927734375,
              -0.6875,
              -0.1953125,
              -0.47265625,
              0.1142578125,
              0.1328125,
              -0.2734375,
              -0.2177734375,
              -0.1806640625,
              -0.2041015625,
              -0.06005859375,
              0.1875,
              -0.2158203125,
              -0.68359375,
              0.23046875,
              0.3828125,
              -0.48046875,
              0.2353515625,
              -0.287109375,
              0.369140625,
              0.26953125,
              -0.4453125,
              0.126953125,
              -0.11474609375,
              -0.21484375,
              0.197265625,
              0.0927734375,
              -0.11669921875,
              -0.224609375,
              0.49609375,
              -0.6953125,
              1.078125,
              0.34375,
              -0.44140625,
              0,
              0.427734375,
              -0.421875,
              -0.2265625,
              0.18359375,
              0.390625,
              0.41796875,
              0.12451171875,
              -0.349609375,
              0.470703125,
              -0.26953125,
              0.10693359375,
              0.212890625,
              0.1826171875,
              0.3125,
              0.279296875,
              -0.022705078125,
              -0.173828125,
              -0.55859375,
              -0.09521484375,
              -0.2734375,
              0.015625,
              0.01214599609375,
              -0.326171875,
              0.052734375,
              -0.11279296875,
              -0.234375,
              -0.060546875,
              0.224609375,
              -0.09716796875,
              -0.2001953125,
              0.1279296875,
              0.375,
              -0.11865234375,
              0.043701171875,
              0.0040283203125,
              -0.6328125,
              0.0703125,
              -0.00927734375,
              -0.04931640625,
              0.140625,
              0.34375,
              0.010498046875,
              0.001708984375,
              0.08056640625,
              0.06298828125,
              0.09033203125,
              -0.28125,
              -0.1533203125,
              0.462890625,
              0.056640625,
              0.28515625,
              0.07568359375,
              0.1103515625,
              -0.220703125,
              -0.369140625,
              -0.216796875,
              -0.06787109375,
              -0.1591796875,
              -0.1015625,
              0.94921875,
              -0.036376953125,
              -0.298828125,
              0.2216796875,
              -0.197265625,
              0.263671875,
              -0.498046875,
              -0.1953125,
              -0.09375,
              0.028076171875,
              -0.21484375,
              -0.013427734375,
              -0.322265625,
              0.56640625,
              0.3359375,
              0.265625,
              -0.96484375,
              -0.05859375,
              0.212890625,
              0.375,
              -0.11181640625,
              -0.12890625,
              0.1689453125,
              -0.380859375,
              -0.5859375,
              0.08642578125,
              0.0966796875,
              0.1484375,
              0.0390625,
              0.10791015625,
              0.2265625,
              -0.00244140625,
              -0.201171875,
              0.28515625,
              -0.038330078125,
              0.68359375,
              0.0185546875,
              0.13671875,
              -0.0546875,
              -0.54296875,
              0.26171875,
              0.412109375,
              -0.06494140625,
              0.345703125,
              0.39453125,
              -0.020263671875,
              -0.2890625,
              0.296875,
              -0.04052734375,
              0.041015625,
              -0.06103515625,
              -0.2373046875,
              0.32421875,
              -0.0615234375,
              -0.046875,
              -0.041015625,
              -0.2353515625,
              -0.1552734375,
              -0.0673828125,
              -0.154296875,
              -0.1376953125,
              0.38671875,
              0.0146484375,
              -0.265625,
              -0.03125,
              0.04345703125,
              0.042236328125,
              0.376953125,
              0.33984375,
              -0.00970458984375,
              0.07421875,
              0.578125,
              0.376953125,
              -0.171875,
              0.04345703125,
              -0.474609375,
              0.322265625,
              0.02099609375,
              -0.1494140625,
              -0.478515625,
              0.00067138671875,
              -0.447265625,
              0.27734375,
              0.041015625,
              -0.09619140625,
              0.228515625,
              -0.2109375,
              0.1123046875,
              0.1298828125,
              -0.00634765625,
              0.37109375,
              0.1689453125,
              0.3359375,
              0.2333984375,
              0.130859375,
              0.11767578125,
              0.0498046875,
              -0.193359375,
              0.125,
              0.1923828125,
              0.05224609375,
              -0.19921875,
              -0.27734375,
              0.119140625,
              -0.06689453125,
              0.08349609375,
              0.259765625,
              0.05078125,
              -0.02490234375,
              -0.1826171875,
              0.41015625,
              -0.283203125,
              -0.08544921875,
              0.267578125,
              -0.1650390625,
              -0.390625,
              0.095703125,
              0.07080078125,
              -0.048828125,
              -0.05224609375,
              -0.365234375,
              0.16015625,
              0.56640625,
              0.02294921875,
              -0.259765625,
              0.033935546875,
              0.33203125,
              -0.1826171875,
              -0.0634765625,
              -0.080078125,
              0.2119140625,
              -0.203125,
              0.0498046875,
              0.0224609375,
              0.1767578125,
              0.078125,
              0.099609375,
              0.10009765625,
              -0.0673828125,
              -0.03076171875,
              0.201171875,
              0.1728515625,
              -0.275390625,
              0.0751953125,
              -0.2080078125,
              -0.337890625,
              0.36328125,
              -0.26953125,
              0.07373046875,
              0.059814453125,
              0.1796875,
              -0.2578125,
              0.234375,
              -0.208984375,
              -0.34375,
              -0.0625,
              0.002410888671875,
              0.24609375,
              0.263671875,
              -0.2353515625,
              -0.2890625,
              0.0079345703125,
              -0.07763671875,
              0.10009765625,
              0.150390625,
              -0.0286865234375,
              0.283203125,
              0.0693359375,
              -0.0016326904296875,
              0.328125,
              0.05908203125,
              -0.40234375,
              0.322265625,
              -0.12890625,
              0.36328125,
              -0.04736328125,
              -0.365234375,
              0.1865234375,
              0.30078125,
              -0.26953125,
              0.15625,
              0.1796875,
              0.0069580078125,
              -0.5390625,
              0.1591796875,
              -0.7421875,
              0.0162353515625,
              -0.1328125,
              -0.3046875,
              0.044921875,
              0.15234375,
              0.0211181640625,
              -0.2314453125,
              0.03759765625,
              -0.146484375,
              -0.1572265625,
              0.154296875,
              0.037109375,
              -0.07373046875,
              -0.228515625,
              -0.10595703125,
              0.01483154296875,
              0.01513671875,
              0.28515625,
              0.318359375,
              0.146484375,
              -0.025634765625,
              -0.05712890625,
              -0.2890625,
              -0.03759765625,
              0.08984375,
              -0.0537109375,
              -0.205078125,
              -0.32421875,
              0.0927734375,
              0.0869140625,
              -0.09033203125,
              -0.14453125,
              0.0712890625,
              -0.09912109375,
              -0.12451171875,
              0.0111083984375,
              -0.1240234375,
              -0.166015625,
              0.1123046875,
              -0.04833984375,
              -0.042236328125,
              -0.07080078125,
              -0.1416015625,
              0.29296875,
              -0.345703125,
              0.1044921875,
              0.09423828125,
              0.0098876953125,
              -0.4375,
              -0.06494140625,
              0.146484375,
              -0.048828125,
              0.154296875,
              0.1083984375,
              -0.34765625,
              0.29296875,
              0.474609375,
              -0.66015625,
              -0.5390625,
              -0.18359375,
              -0.1259765625,
              -0.107421875,
              0.36328125,
              -0.0400390625,
              -0.1650390625,
              0.054931640625,
              0.6953125,
              0.03759765625,
              0.056884765625,
              0.1611328125,
              0.326171875,
              -0.1767578125,
              0.208984375,
              -0.0810546875,
              -0.423828125,
              0.28515625,
              0.061767578125,
              -0.67578125,
              0.240234375,
              0.1123046875,
              0.2177734375,
              -0.01300048828125,
              0.177734375,
              -0.328125,
              -0.1142578125,
              -0.349609375,
              0.2421875,
              -0.08251953125,
              -0.1171875,
              0.1591796875,
              0.326171875,
              0.31640625,
              0.0927734375,
              0.392578125,
              -0.396484375,
              0.12890625,
              0.08984375,
              -0.02099609375,
              0.361328125,
              -0.5625,
              -0.185546875,
              -0.19140625,
              -0.0189208984375,
              0.0096435546875,
              0.2275390625,
              -0.1650390625,
              -0.2890625,
              0.19140625,
              -0.130859375,
              -0.036865234375,
              0.1611328125,
              -0.08837890625,
              0.1484375,
              -0.1533203125,
              -0.06298828125,
              -0.1298828125,
              -0.14453125,
              0.2578125,
              0.01336669921875,
              0.302734375,
              -0.2041015625,
              -0.030517578125,
              0.30078125,
              0.1064453125,
              0.28125,
              0.045166015625,
              -0.08935546875,
              -0.17578125,
              0.1025390625,
              -0.46484375,
              0.294921875,
              0.06396484375,
              0.33203125,
              -0.298828125,
              0.039306640625,
              -0.010009765625,
              0.36328125,
              0.024169921875,
              -0.30859375,
              -0.5,
              -0.50390625,
              0.373046875,
              0.0091552734375,
              0.1328125,
              -0.0986328125,
              0.5,
              0.2392578125,
              0.16015625,
              0.357421875,
              0.333984375,
              -0.1484375,
              0.1455078125,
              0.00787353515625,
              -0.37109375,
              0.40625,
              -0.1650390625,
              0.134765625,
              -0.5078125,
              0.33203125,
              -0.0023193359375,
              -0.072265625,
              0.0247802734375,
              0.275390625,
              -0.251953125,
              -0.1953125,
              -0.0771484375,
              0.150390625,
              -0.173828125,
              -0.1748046875,
              -0.189453125,
              -0.25390625,
              -0.030029296875,
              0.1259765625,
              -0.283203125,
              -0.06591796875,
              0.0693359375,
              0.3828125,
              0.796875,
              -0.072265625,
              -0.17578125,
              -0.166015625,
              0.365234375,
              0.16015625,
              0.515625,
              -0.051513671875,
              0.236328125,
              -0.25390625,
              -0.5703125,
              0.345703125,
              0.08837890625,
              0.5859375,
              -0.65234375,
              0.05029296875,
              -0.2060546875,
              0.28515625,
              0.0888671875,
              0.005401611328125,
              -0.1025390625,
              0.0703125,
              0.53515625,
              -0.4921875,
              0.466796875,
              -0.1552734375,
              0.0634765625,
              0.28125,
              0.31640625,
              0.07080078125,
              -0.48828125,
              -0.421875,
              0.0078125,
              0.048828125,
              -0.357421875,
              -0.318359375,
              0.23828125,
              0.427734375,
              -0.482421875,
              0.287109375,
              -0.302734375,
              0.578125,
              -0.072265625,
              0.0869140625,
              -0.0625,
              -0.11474609375,
              0.291015625,
              -0.400390625,
              -0.33203125,
              0.1474609375,
              -0.2255859375,
              0.28125,
              -0.375,
              0.67578125,
              0.08740234375,
              0.0224609375,
              0.259765625,
              0.025634765625,
              -0.2470703125,
              -0.130859375,
              0.203125,
              0.03271484375,
              0.023193359375,
              0.2392578125,
              -0.2099609375,
              -0.1220703125,
              -0.1728515625,
              0.064453125,
              -0.126953125,
              0.1328125,
              -0.56640625,
              0.1064453125,
              -0.2158203125,
              -0.197265625,
              -0.3125,
              0.1328125,
              -0.140625,
              0.019775390625,
              -0.09228515625,
              0.3125,
              -0.35546875,
              -0.3125,
              0.0009765625,
              -0.359375,
              0.427734375,
              0.154296875,
              -0.48046875,
              0.1279296875,
              0.2265625,
              0.5390625,
              0.388671875,
              -0.0869140625,
              -0.048828125,
              -0.1826171875,
              0.4921875,
              0.171875,
              1.09375,
              0.85546875,
              0.2734375,
              -0.201171875,
              0.142578125,
              -0.1865234375,
              -0.083984375,
              -0.029296875,
              -0.2578125,
              -0.2431640625,
              -0.062255859375,
              -0.3359375,
              -0.03466796875,
              -0.142578125,
              -0.244140625,
              0.056640625,
              0.052490234375,
              -0.1708984375,
              0.263671875,
              -0.11181640625,
              -0.123046875,
              -0.052490234375,
              -0.15625,
              -0.1796875,
              -0.3046875,
              0.00860595703125,
              0.0025634765625,
              0.291015625,
              0.134765625,
              0.09765625,
              -0.2421875,
              0.154296875,
              -0.107421875,
              -0.1484375,
              0.1083984375,
              0.1015625,
              -0.16796875,
              0.0732421875,
              -0.365234375,
              0.061279296875,
              -0.328125,
              0.06689453125,
              -0.046142578125,
              -0.134765625,
              -0.0181884765625,
              0.2001953125,
              -0.08203125,
              -0.009521484375,
              -0.10595703125,
              0.2578125,
              0.09033203125,
              0.310546875,
              -0.314453125,
              -0.1982421875,
              0.044921875,
              0.240234375,
              -0.181640625,
              -0.283203125,
              0.076171875,
              -0.1884765625,
              -0.0712890625,
              -0.228515625,
              0.07861328125,
              0.06298828125,
              -0.263671875,
              -0.04052734375,
              -0.1826171875,
              0.15625,
              -0.142578125,
              -0.0966796875,
              -0.056884765625,
              -0.1396484375,
              -0.08203125,
              -0.03857421875,
              -0.10546875,
              0.484375,
              0.080078125,
              0.042724609375,
              0.462890625,
              -0.08349609375,
              -0.1376953125,
              -0.28125,
              0.2578125,
              -0.1376953125,
              -0.328125,
              -0.045654296875,
              -0.0908203125,
              0.1650390625,
              -0.228515625,
              -0.232421875,
              -0.00537109375,
              0.28515625,
              0.07763671875,
              0.052001953125,
              0.0267333984375,
              -0.11328125,
              0.03515625,
              -0.000885009765625,
              -0.00225830078125,
              0.11083984375,
              0.015869140625,
              -0.06396484375,
              0.37890625,
              -0.173828125,
              0.1845703125,
              -0.15234375,
              -0.173828125,
              -0.08642578125,
              0.236328125,
              -0.0888671875,
              -0.1337890625,
              0.04931640625,
              -0.0703125,
              0.1044921875,
              0.06787109375,
              0.1416015625,
              -0.55859375,
              0.20703125,
              -0.205078125,
              0.306640625,
              0.2080078125,
              0.19140625,
              -0.3359375,
              -0.166015625,
              0.162109375,
              -0.1787109375,
              -0.00048828125,
              0.298828125,
              0.216796875,
              0.02490234375,
              0.1376953125,
              0.01507568359375,
              -0.265625,
              -0.248046875,
              0.10302734375,
              -0.26171875,
              -0.98046875,
              -0.5078125,
              -0.2138671875,
              0.96875,
              -0.2314453125,
              0.73828125,
              0.2109375,
              -0.431640625,
              -0.20703125,
              0.28125,
              -0.10693359375,
              -0.6796875,
              0.375,
              0.10498046875,
              -0.0810546875,
              0.33984375,
              0.255859375,
              -0.474609375,
              0.142578125,
              0.181640625,
              -0.6328125,
              0.193359375,
              -0.34765625,
              0.1435546875,
              -0.416015625,
              0.306640625,
              0.359375,
              0.34765625,
              -0.4921875,
              0.013916015625,
              -0.140625,
              -0.05712890625,
              0.484375,
              0.25390625,
              -0.00897216796875,
              -0.330078125,
              -0.3125,
              -0.091796875,
              -0.02685546875,
              0.1904296875,
              0.365234375,
              0.5546875,
              0.57421875,
              0.470703125,
              -0.056640625,
              -0.2734375,
              -0.06591796875,
              -0.6796875,
              -0.04150390625,
              0.78125,
              -0.357421875,
              -0.181640625,
              1.15625,
              0.24609375,
              -0.1943359375,
              0.0810546875,
              -0.3671875,
              0.1259765625,
              0.57421875,
              -0.3203125,
              -0.26171875,
              0.466796875,
              1.203125,
              0.263671875,
              -0.2265625,
              -0.12060546875,
              -0.9765625,
              -0.0283203125,
              0.388671875,
              -0.7265625,
              -0.97265625,
              0.4609375,
              -0.306640625,
              0.345703125,
              -0.2109375,
              0.158203125,
              0.443359375,
              -0.8828125,
              0.09814453125,
              -0.04833984375,
              0.64453125,
              0.2490234375,
              0.52734375,
              0.51953125,
              0.04736328125,
              0.48046875,
              -0.466796875,
              0.10302734375,
              -0.396484375,
              -0.201171875,
              -0.8125,
              -0.453125,
              -0.90625,
              -0.189453125,
              0.1982421875,
              -0.2177734375,
              0.6328125,
              -0.01953125,
              0.4296875,
              -0.68359375,
              -0.56640625,
              0.64453125,
              -0.2373046875,
              -0.484375,
              -0.66796875,
              -0.408203125,
              -0.6484375,
              0.8984375,
              -0.275390625,
              0.34375,
              0.86328125,
              0.71484375,
              0.32421875,
              -0.0712890625,
              0.03466796875,
              1.1640625,
              0.328125,
              0.08544921875,
              0.57421875,
              0.5,
              0.38671875,
              -0.828125,
              0.5703125,
              0.8125,
              -0.16015625,
              -1.0390625,
              -0.1318359375,
              -0.0303955078125,
              -0.061767578125,
              0.1865234375,
              0.3203125,
              0.02734375,
              -0.240234375,
              -0.0615234375,
              0.00714111328125,
              -0.05078125,
              0.04150390625,
              -0.030029296875,
              -0.0751953125,
              -0.302734375,
              0.2734375,
              0.1435546875,
              -0.3984375,
              0.2216796875,
              0.08984375,
              -0.0419921875,
              -0.1982421875,
              -0.0908203125,
              0.00531005859375,
              0.087890625,
              0.099609375,
              0.3828125,
              0.1669921875,
              -0.0634765625,
              -0.1689453125,
              0.390625,
              0.11572265625,
              0.045166015625,
              0.0986328125,
              0.049560546875,
              -0.1357421875,
              0.078125,
              -0.310546875,
              -0.01318359375,
              -0.14453125,
              0.3125,
              -0.1708984375,
              -0.1064453125,
              0.2138671875,
              -0.099609375,
              0.1943359375,
              -0.2119140625,
              0.0537109375,
              0.78515625,
              -0.0010986328125,
              0.10546875,
              0.279296875,
              0.1796875,
              -0.2314453125,
              -0.003173828125,
              0.12158203125,
              0.1123046875,
              -0.2265625,
              0.09228515625,
              -0.41015625,
              -0.37890625,
              0.1416015625,
              -0.0167236328125,
              -0.2294921875,
              -0.16015625,
              -0.1025390625,
              -0.2216796875,
              -0.2275390625,
              -0.1640625,
              -0.0751953125,
              0.251953125,
              -0.052978515625,
              -0.2353515625,
              0.0091552734375,
              -0.3046875,
              0.3203125,
              -0.03955078125,
              0.0693359375,
              0.12255859375,
              0.150390625,
              0.02880859375,
              -0.25390625,
              -0.146484375,
              -0.09912109375,
              0.1455078125,
              0.41796875,
              -0.1474609375,
              0.0181884765625,
              0.140625,
              0.046875,
              -0.263671875,
              0.01220703125,
              0.029296875,
              -0.11279296875,
              0.2490234375,
              -0.08642578125,
              0.050048828125,
              0.035888671875,
              0.09130859375,
              0.1279296875,
              0.0830078125,
              0.0267333984375,
              0.11767578125,
              -0.10205078125,
              -0.3046875,
              -0.045654296875,
              0.1484375,
              0.16796875,
              0.10595703125,
              0.1064453125,
              0.06591796875,
              0.028076171875,
              -0.490234375,
              0.1796875,
              0.020263671875,
              -0.014404296875,
              0.158203125,
              0.043212890625,
              -0.01422119140625,
              -0.041015625,
              0.248046875,
              -0.169921875,
              0.185546875,
              -0.07958984375,
              0.08544921875,
              0.076171875,
              -0.10302734375,
              -0.01953125,
              0.2373046875,
              -0.2109375
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
    index = "fewnerd_tests"
    async for batch in dataset_provider.consume_big_aggregation(query, agg_key="grouped_results", index=index):
        yield batch


async def main():
    data = defaultdict(list)
    async for item in get_next_batch():
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
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
