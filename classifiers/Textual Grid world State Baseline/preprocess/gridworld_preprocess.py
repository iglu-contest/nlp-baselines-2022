
import pandas as pd
import os
from pathlib import Path
import numpy as np
import json
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
import warnings



from .tools import block_tools


path = Path(__file__).parent / "../public_data"

class LocalEvalConfig:
        CLASSIFIER_RESULTS_FILE = '/home/felipe//minerl2022/iglu/nlp/iglu-2022-clariq-nlp-starter-kit//local-eval-classifier-results.json'
        RANKER_RESULTS_FILE = '/home/felipe//minerl2022/iglu/nlp/iglu-2022-clariq-nlp-starter-kit//local-eval-ranker-results.json'
        DATA_FOLDER = path #'/home/felipe//minerl2022/iglu/nlp/iglu-2022-clariq-nlp-starter-kit/public_data'



def read_json_file(fname):
    with open(fname, 'r') as fp:
        d = json.load(fp)
    return d





def get_gridworld_state(row, datafolder=LocalEvalConfig.DATA_FOLDER):
    
    state_file_path = row.InitializedWorldPath
    state_file_path = os.path.join(datafolder, state_file_path)
    state = read_json_file(state_file_path)
    for drop_key in ['gameId', 'stepId', 'tape', 'clarification_question']:
        state.pop(drop_key)
    return state['avatarInfo']['pos'],state['avatarInfo']['look'],state['worldEndingState']['blocks']

def complete_df_with_grid_state(df):

    df['InputInstructionWithGameID'] = df.InputInstruction + df.GameId
    
    res = df.apply(get_gridworld_state,axis=1)

    df['pos']=res.apply(lambda x: x[0])
    df['look']=res.apply(lambda x: x[1])
    df['blocks']=res.apply(lambda x: x[2])

    return df





if __name__=='__main__':

    df = pd.read_csv(os.path.join(LocalEvalConfig.DATA_FOLDER, 'clarifying_questions_train.csv'))
    df = complete_df_with_grid_state()

    import multiprocess

    with multiprocess.Pool(processes=4) as pool:
       contexts = pool.map( block_tools.create_context_colour_count(blocks))


    # we should add the context to the question and run the ranker again


   