import os
import json
import numpy as np
import torch
from transformers import BertTokenizer
from models.model import Builder
from models.utils import transform_block, convert_blocks_to_tensor_repr


class TextGridCrossModularity:
    def __init__(self, saved_model_path=None, tokenizer=None, config=None, device=None):
        self.saved_model_path = "Text-Grid Cross Modularity" if not saved_model_path else saved_model_path
        if not config:
            with open(os.path.join(self.saved_model_path, "config.json"), "r") as f:
                self.config = json.load(f)
        else:
            self.config = config
        # self.config['saved_model_path'] = os.path.join(self.saved_model_path, "classifier_model.pt")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if not device else device
        self.tokenizer = BertTokenizer.from_pretrained("./bert-base-uncased-local") if not tokenizer else tokenizer
        # self.model = Builder.from_pretrained(self.config).to(self.device)
        self.model = Builder(self.config).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(self.saved_model_path, "classifier_model.pt")))

    def raise_aicrowd_error(self, msg):
        """ Will be used by the evaluator to provide logs, DO NOT CHANGE """
        raise NameError(msg)

    def clarification_required(self, instruction, gridworld_state):
        # return np.random.choice([0, 1])
        """
        Implements classifier for given instuction - whether a clarifying question is required or not
        Inputs:
            instruction - Single instruction string

            gridworld_state - Internal state from the iglu-gridworld simulator corresponding to the instuction
                              NOTE: The state will only contain the "avatarInfo" and "worldEndingState"

        Outputs:
            0 or 1 - 0 if clarification is not required, 1 if clarification is required 

        """

        batch_size = 1
        dialogue_history = '[CLS] ' + instruction + ' [SEP]'
        inputs = self.tokenizer(dialogue_history,
                                max_length=100,
                                truncation=True,
                                padding='max_length',
                                return_tensors="pt")

        initial_world_blocks = [
            transform_block(block)
            for block in gridworld_state['worldEndingState']['blocks']
        ]
        world_repr = convert_blocks_to_tensor_repr(initial_world_blocks)

        inputs = {k: v.view(batch_size, -1).to(self.device) for k, v in inputs.items()}
        world_repr = world_repr.unsqueeze(0).to(self.device)
        logits, _ = self.model(inputs, world_repr)
        prediction = torch.argmax(logits, dim=-1)

        return prediction.cpu().numpy()
