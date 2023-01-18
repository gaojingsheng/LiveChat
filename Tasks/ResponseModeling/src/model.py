
import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, BertTokenizer
from transformers.models.bert.modeling_bert import (BertEmbeddings,
                                                    BertEncoder, BertPooler)
import torch.nn.functional as F


class BertModel2(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings


class ThreeBert(BertPreTrainedModel):
    def __init__(self, config, train_from_scratch=False):
        super(ThreeBert, self).__init__(config)

        self.bert_model_config = config 
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")  

        if not train_from_scratch:
            self.bert_model1 = BertModel.from_pretrained("bert-base-chinese", config=self.bert_model_config)
            self.bert_model2 = BertModel2.from_pretrained("bert-base-chinese", config=self.bert_model_config)
            self.bert_model3 = BertModel.from_pretrained("bert-base-chinese", config=self.bert_model_config)  
        else:
            self.bert_model1 = BertModel(config=self.bert_model_config)
            self.bert_model2 = BertModel2(config=self.bert_model_config)
            self.bert_model3 = BertModel(config=self.bert_model_config)     

    def get_history(self, history):

        history_posts = []
        for i in range(history.size(0)):
            history_posts.append(self.history_post_list[str(int(history[i]))])
        output = self.bert_tokenizer(history_posts, max_length=512, padding=True, truncation=True, return_tensors="pt")
        for index in output:
            output[index] = output[index].to(self.device)
            assert output[index].device != "cpu"
        return output

    def match(self, model, x, y, x_mask, y_mask):
        # Multi-hop Co-Attention
        # x: (batch_size, m, hidden_size)
        # y: (batch_size, n, hidden_size)
        # x_mask: (batch_size, m)
        # y_mask: (batch_size, n)
        assert x.dim() == 3 and y.dim() == 3
        assert x_mask.dim() == 2 and y_mask.dim() == 2
        assert x_mask.shape == x.shape[:2] and y_mask.shape == y.shape[:2]

        attn_mask = torch.bmm(x_mask.unsqueeze(-1), y_mask.unsqueeze(1)) # (batch_size, m, n)
        attn = torch.bmm(x, y.transpose(1,2)) # (batch_size, m, n)
        model.attn = attn
        model.attn_mask = attn_mask
        
        x_to_y = torch.softmax(attn * attn_mask + (-5e4) * (1-attn_mask), dim=2) # (batch_size, m, n)
        y_to_x = torch.softmax(attn * attn_mask + (-5e4) * (1-attn_mask), dim=1).transpose(1,2) # # (batch_size, n, m)
        
        # x_attended, y_attended = None, None # no hop-1
        x_attended = torch.bmm(x_to_y, y) # (batch_size, m, hidden_size)
        y_attended = torch.bmm(y_to_x, x) # (batch_size, n, hidden_size)

        # x_attended_2hop, y_attended_2hop = None, None # no hop-2
        y_attn = torch.bmm(y_to_x.mean(dim=1, keepdim=True), x_to_y) # (batch_size, 1, n) # true important attention over y
        x_attn = torch.bmm(x_to_y.mean(dim=1, keepdim=True), y_to_x) # (batch_size, 1, m) # true important attention over x

        # truly attended representation
        x_attended_2hop = torch.bmm(x_attn, x).squeeze(1) # (batch_size, hidden_size)
        y_attended_2hop = torch.bmm(y_attn, y).squeeze(1) # (batch_size, hidden_size)

        x_attended = x_attended, x_attended_2hop
        y_attended = y_attended, y_attended_2hop

        return x_attended, y_attended

    def aggregate(self, aggregation_method, x, x_mask):
        # x: (batch_size, seq_len, emb_size)
        # x_mask: (batch_size, seq_len)
        assert x.dim() == 3 and x_mask.dim() == 2
        assert x.shape[:2] == x_mask.shape
        # batch_size, seq_len, emb_size = x.shape

        if aggregation_method == "mean":
            return (x * x_mask.unsqueeze(-1)).sum(dim=1)/x_mask.sum(dim=-1, keepdim=True).clamp(min=1) # (batch_size, emb_size)

        if aggregation_method == "max":
            return x.masked_fill(x_mask.unsqueeze(-1)==0, -5e4).max(dim=1)[0] # (batch_size, emb_size)

        if aggregation_method == "mean_max":
            return torch.cat([(x * x_mask.unsqueeze(-1)).sum(dim=1)/x_mask.sum(dim=-1, keepdim=True).clamp(min=1), \
                x.masked_fill(x_mask.unsqueeze(-1)==0, -5e4).max(dim=1)[0]], dim=-1) # (batch_size, 2*emb_size)


    def fuse(self, model, aggregation_method, batch_x_emb, batch_y_emb, batch_persona_emb, \
        batch_x_mask, batch_y_mask, batch_persona_mask, batch_size, num_candidates):
        
        batch_x_emb, batch_y_emb_context = self.match(model, batch_x_emb, batch_y_emb, batch_x_mask, batch_y_mask)
        # batch_x_emb: ((batch_size*num_candidates, m, emb_size), (batch_size*num_candidates, emb_size))
        # batch_y_emb_context: (batch_size*num_candidates, n, emb_size), (batch_size*num_candidates, emb_size)
        
        # hop 2 results
        batch_x_emb_2hop = batch_x_emb[1]
        batch_y_emb_context_2hop = batch_y_emb_context[1]
        
        # mean_max aggregation for the 1st hop result
        batch_x_emb = self.aggregate(aggregation_method, batch_x_emb[0], batch_x_mask) # batch_x_emb: (batch_size*num_candidates, 2*emb_size)
        batch_y_emb_context = self.aggregate(aggregation_method, batch_y_emb_context[0], batch_y_mask) # batch_y_emb_context: (batch_size*num_candidates, 2*emb_size)

        if batch_persona_emb is not None:
            batch_persona_emb, batch_y_emb_persona = self.match(model, batch_persona_emb, batch_y_emb, batch_persona_mask, batch_y_mask)
            # batch_persona_emb: (batch_size*num_candidates, m, emb_size), (batch_size*num_candidates, emb_size)
            # batch_y_emb_persona: (batch_size*num_candidates, n, emb_size), (batch_size*num_candidates, emb_size)

            batch_persona_emb_2hop = batch_persona_emb[1]
            batch_y_emb_persona_2hop = batch_y_emb_persona[1]

            # # no hop-1
            # return torch.bmm(torch.cat([batch_x_emb_2hop, batch_persona_emb_2hop], dim=-1).unsqueeze(1), \
            #             torch.cat([batch_y_emb_context_2hop, batch_y_emb_persona_2hop], dim=-1)\
            #                 .unsqueeze(-1)).reshape(batch_size, num_candidates)

            batch_persona_emb = self.aggregate(aggregation_method, batch_persona_emb[0], batch_persona_mask) # batch_persona_emb: (batch_size*num_candidates, 2*emb_size)
            batch_y_emb_persona = self.aggregate(aggregation_method, batch_y_emb_persona[0], batch_y_mask) # batch_y_emb_persona: (batch_size*num_candidates, 2*emb_size)

            # # no hop-2
            # return torch.bmm(torch.cat([batch_x_emb, batch_persona_emb], dim=-1).unsqueeze(1), \
            #             torch.cat([batch_y_emb_context, batch_y_emb_persona], dim=-1)\
            #                 .unsqueeze(-1)).reshape(batch_size, num_candidates)
            return torch.bmm(torch.cat([batch_x_emb, batch_x_emb_2hop, batch_persona_emb, batch_persona_emb_2hop], dim=-1).unsqueeze(1), \
                        torch.cat([batch_y_emb_context, batch_y_emb_context_2hop, batch_y_emb_persona, batch_y_emb_persona_2hop], dim=-1)\
                            .unsqueeze(-1)).reshape(batch_size, num_candidates)
        else:
            return torch.bmm(torch.cat([batch_x_emb, batch_x_emb_2hop], dim=-1).unsqueeze(1), \
                        torch.cat([batch_y_emb_context, batch_y_emb_context_2hop], dim=-1)\
                            .unsqueeze(-1)).reshape(batch_size, num_candidates)

    def forward(self, ids, mask, token_type_ids, apply_interaction=True, aggregation_method="max"):
        # version 3, fussion
        # Context
        input_ids1 = ids[0]
        attention_mask1 = mask[0]
        token_type_ids1 = token_type_ids[0]
        # Response
        input_ids2 = ids[1]
        attention_mask2 = mask[1]
        token_type_ids2 = token_type_ids[1]

        if len(ids)==3:
            # print("begin history post")
            input_ids3 = ids[2]
            attention_mask3 = mask[2]
            token_type_ids3 = token_type_ids[2]
            # last_hidden_state, bert_output = self.bert_model1(input_ids=input_ids3, attention_mask=attention_mask3, token_type_ids=token_type_ids3)
            outputs = self.bert_model1(input_ids=input_ids3, attention_mask=attention_mask3, token_type_ids=token_type_ids3)

            outputs1 = self.bert_model2(
                input_ids=input_ids1,
                attention_mask=attention_mask1,
                token_type_ids=token_type_ids1,
                )
            outputs2 = self.bert_model3(
                input_ids=input_ids2,
                attention_mask=attention_mask2,
                token_type_ids=token_type_ids2,
                )

            if apply_interaction:

                attention_mask1 = attention_mask1.float()
                attention_mask2 = attention_mask2.float()
                attention_mask3 = attention_mask3.float()

                batch_size, sent_len, emb_size = outputs2[0].shape
                history_output = outputs[0].repeat_interleave(batch_size, dim=0)
                attention_mask3 = attention_mask3.repeat_interleave(batch_size, dim=0)

                context_output = outputs1[0].repeat_interleave(batch_size, dim=0)
                attention_mask1 = attention_mask1.repeat_interleave(batch_size, dim=0)

                response_output = outputs2[0].unsqueeze(0).repeat(batch_size, 1, 1, 1).reshape(-1, sent_len, emb_size) 
                attention_mask2 = attention_mask2.unsqueeze(0).repeat(batch_size, 1, 1).reshape(-1, sent_len)

                logits = self.fuse(self.bert_model2, aggregation_method, \
                    context_output, response_output, history_output, attention_mask1, attention_mask2, attention_mask3, batch_size, batch_size)
                targets = torch.arange(batch_size, dtype=torch.long, device=self.device)
                # loss = F.cross_entropy(logits, targets)
                # num_ok = (targets.long() == logits.float().argmax(dim=1)).sum()
                # return loss, num_ok
                return logits, targets

            else:
                history_output = outputs[0].mean(dim=1)
                context_output = outputs1[0].mean(dim=1)
                response_output = outputs2[0].mean(dim=1)
                history_context_output = (history_output + context_output)/2
                
                return history_context_output, response_output

        else:
            outputs1 = self.bert_model2(
                input_ids=input_ids1,
                attention_mask=attention_mask1,
                token_type_ids=token_type_ids1,
                )
            outputs2 = self.bert_model3(
                input_ids=input_ids2,
                attention_mask=attention_mask2,
                token_type_ids=token_type_ids2,
                )
            if apply_interaction:
                attention_mask1 = attention_mask1.float()
                attention_mask2 = attention_mask2.float()

                attention_mask3 = None
                history_output = None

                batch_size, sent_len, emb_size = outputs2[0].shape

                context_output = outputs1[0].repeat_interleave(batch_size, dim=0)
                attention_mask1 = attention_mask1.repeat_interleave(batch_size, dim=0)

                response_output = outputs2[0].unsqueeze(0).repeat(batch_size, 1, 1, 1).reshape(-1, sent_len, emb_size) 
                attention_mask2 = attention_mask2.unsqueeze(0).repeat(batch_size, 1, 1).reshape(-1, sent_len)

                logits = self.fuse(self.bert_model2, aggregation_method, \
                    context_output, response_output, history_output, attention_mask1, attention_mask2, attention_mask3, batch_size, batch_size)
                targets = torch.arange(batch_size, dtype=torch.long, device=self.device)

                return logits, targets
            else:
                return outputs1[0].mean(dim=1), outputs2[0].mean(dim=1)
