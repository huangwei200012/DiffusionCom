import torch
import torch.nn as nn
import numpy as np
from .base import BaseLitModel
from transformers.optimization import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
from functools import partial
from .utils import LabelSmoothSoftmaxCEV1
from typing import Callable, Iterable, List
from torch_geometric.utils import subgraph
def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))

def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()

def decode(output_ids, tokenizer):
    return lmap(str.strip, tokenizer.batch_decode(output_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True))

class TransformerLitModel(BaseLitModel):
    def __init__(self, model, args, tokenizer=None, data_config={}):
        super().__init__(model, args)
        self.save_hyperparameters(args)
        self.args = args
        if args.bce:
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif args.label_smoothing != 0.0:
            self.loss_fn = LabelSmoothSoftmaxCEV1(lb_smooth=args.label_smoothing)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.best_acc = 0
        self.first = True
        self.tokenizer = tokenizer
        self.__dict__.update(data_config)

        # resize the word embedding layer
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.decode = partial(decode, tokenizer=self.tokenizer)
        # self.all_edge_index = torch.load('/nlp_group/huangwei12/Mul_graph/MKGformer_diiffusion/MKG_ori/dataset/FB15k-237/graph_data.pth').to("cuda:{}".format(self.args.gpus.replace(",","")))
        if args.pretrain:
            # when pretrain, only tune embedding layers
            self._freeze_attention()


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        label = batch.pop("label")
        en = batch.pop("en")
        input_ids = batch['input_ids']
        en_list = en.to("cpu").tolist()
        indices_relation = [torch.nonzero(row == 1, as_tuple=False).squeeze(1).to("cpu").tolist() for row in labels]
        indices_relation = [list(set(i) & set(en_list)) for i in indices_relation]
        indices_relation = [[x for x in indices_relation_sub if x != en_i] for indices_relation_sub,en_i in zip(indices_relation,en_list)]
        new_indices_relation = []
        for i in range(len(indices_relation)):
            indices_relation_sub = indices_relation[i]
            new_indices_relation_sub = []
            en_i = en_list[i]
            for x in indices_relation_sub:
                if x !=en_i and int(label[i].item())!=x:
                    new_indices_relation_sub.append(x)
            new_indices_relation.append(new_indices_relation_sub)
        indices_relation = new_indices_relation
        
        # filtered_tensor = en[en != value]
        mapping = {en[i].item(): i for i in range(en.size(0))}
        map_function = lambda x: mapping.get(x, x)
        # sub_edge_index, sub_edge_attr = subgraph(en, self.all_edge_index)
        # # 之差修改一下sub_edge_index
        # sub_edge_index = sub_edge_index.clone().cpu().apply_(map_function)
        # sub_edge_index = sub_edge_index.to(en.device)
        # 之后就是对repeated_indices进行一个遍历
        temp_tensor = torch.empty(2, 0,dtype=torch.int64).to(en.device)
        for i in range(len(indices_relation)):
            sub_indices_relation = indices_relation[i]
            for j in sub_indices_relation:
                new_column = torch.tensor([[en_list[i]], [j]]).to(temp_tensor.device)  # 替换成你实际要添加的列
                temp_tensor = torch.cat([temp_tensor,new_column],dim=-1)
        temp_tensor = temp_tensor.clone().cpu().apply_(map_function)
        temp_tensor = temp_tensor.to(en.device)
        # import pdb;pdb.set_trace()
        batch["edge_index"] = temp_tensor
        outputs = self.model(**batch, return_dict=True)
        mask_logits = outputs.logits
        kl_loss = outputs.hidden_states

        if self.args.bce:
            loss = self.loss_fn(mask_logits, labels)
        else:
            loss = self.loss_fn(mask_logits, label)
        
        # import pdb;pdb.set_trace()
        loss = loss + self.args.kl_loss*kl_loss
        # loss = self.args.kl_loss*kl_loss
        # print("使用kl_loss")

        if batch_idx == 0:
            print('\n'.join(self.decode(batch['input_ids'][:4])))
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        print(f"Step: {self.global_step}, Loss: {loss}, LR: {current_lr}")
        return loss

    def _eval(self, batch, batch_idx, ):
        labels = batch.pop("labels")
        input_ids = batch['input_ids']
        en = batch.pop("en")
        # single label
        label = batch.pop('label')  # bsz

        en_list = en.to("cpu").tolist()
        indices_relation = [torch.nonzero(row == 1, as_tuple=False).squeeze(1).to("cpu").tolist() for row in labels]
        indices_relation = [list(set(i) & set(en_list)) for i in indices_relation]
        indices_relation = [[x for x in indices_relation_sub if x != en_i] for indices_relation_sub,en_i in zip(indices_relation,en_list)]
        new_indices_relation = []
        for i in range(len(indices_relation)):
            indices_relation_sub = indices_relation[i]
            new_indices_relation_sub = []
            en_i = en_list[i]
            for x in indices_relation_sub:
                if x !=en_i and int(label[i].item())!=x:
                    new_indices_relation_sub.append(x)
            new_indices_relation.append(new_indices_relation_sub)
        indices_relation = new_indices_relation
        # filtered_tensor = en[en != value]
        mapping = {en[i].item(): i for i in range(en.size(0))}
        map_function = lambda x: mapping.get(x, x)
        # sub_edge_index, sub_edge_attr = subgraph(en, self.all_edge_index)
        # # 之差修改一下sub_edge_index
        # sub_edge_index = sub_edge_index.clone().cpu().apply_(map_function)
        # sub_edge_index = sub_edge_index.to(en.device)
        # 之后就是对repeated_indices进行一个遍历
        temp_tensor = torch.empty(2, 0,dtype=torch.int64).to(en.device)
        for i in range(len(indices_relation)):
            sub_indices_relation = indices_relation[i]
            for j in sub_indices_relation:
                new_column = torch.tensor([[en_list[i]], [j]]).to(temp_tensor.device)  # 替换成你实际要添加的列
                temp_tensor = torch.cat([temp_tensor,new_column],dim=-1)
        temp_tensor = temp_tensor.clone().cpu().apply_(map_function)
        temp_tensor = temp_tensor.to(en.device)
        batch["edge_index"] = temp_tensor

        bsz = input_ids.shape[0]
        assert labels[0][label[0]], "correct ids must in filiter!"
        labels[torch.arange(bsz), label] = 0
        result_ranks = torch.empty(bsz, 0,dtype=torch.int64)
        for i in range(20):
            logits = self.model(**batch, return_dict=True).logits
            assert logits.shape == labels.shape
            logits += labels * -100 # mask entity

            _, outputs = torch.sort(logits, dim=1, descending=True) # bsz, entities   index
            _, outputs = torch.sort(outputs, dim=1)
            ranks = outputs[torch.arange(bsz), label].detach().cpu() + 1
            ranks = ranks.unsqueeze(1)
            result_ranks = torch.cat((result_ranks,ranks),dim=-1)
        result_ranks_values = result_ranks.min(dim=-1).values
        return dict(ranks = np.array(result_ranks_values))

    def validation_step(self, batch, batch_idx):
        result = self._eval(batch, batch_idx)
        return result

    def validation_epoch_end(self, outputs) -> None:
        ranks = np.concatenate([_['ranks'] for _ in outputs])
        total_ranks = ranks.shape[0]

        if not self.args.pretrain:
            l_ranks = ranks[np.array(list(np.arange(0, total_ranks, 2)))]
            r_ranks = ranks[np.array(list(np.arange(0, total_ranks, 2))) + 1]
            self.log("Eval/lhits10", (l_ranks<=10).mean())
            self.log("Eval/rhits10", (r_ranks<=10).mean())

        hits20 = (ranks<=20).mean()
        hits10 = (ranks<=10).mean()
        hits3 = (ranks<=3).mean()
        hits1 = (ranks<=1).mean()

        self.log("Eval/hits10", hits10)
        self.log("Eval/hits20", hits20)
        self.log("Eval/hits3", hits3)
        self.log("Eval/hits1", hits1)
        self.log("Eval/mean_rank", ranks.mean())
        self.log("Eval/mrr", (1. / ranks).mean())
        self.log("hits10", hits10, prog_bar=True)
        self.log("hits1", hits1, prog_bar=True)
        self.log("mean_rank", ranks.mean(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        result = self._eval(batch, batch_idx)
        # self.log("Test/ranks", np.mean(ranks))
        return result

    def test_epoch_end(self, outputs) -> None:
        ranks = np.concatenate([_['ranks'] for _ in outputs])

        hits20 = (ranks<=20).mean()
        hits10 = (ranks<=10).mean()
        hits3 = (ranks<=3).mean()
        hits1 = (ranks<=1).mean()
        self.log("Test/hits10", hits10)
        self.log("Test/hits20", hits20)
        self.log("Test/hits3", hits3)
        self.log("Test/hits1", hits1)
        self.log("Test/mean_rank", ranks.mean())
        self.log("Test/mrr", (1. / ranks).mean())

    def configure_optimizers(self):
        no_decay_param = ["bias", "LayerNorm.weight"]

        optimizer_group_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay_param)], "weight_decay": 0}
        ]

        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.lr, eps=1e-8)
        if self.args.bce:
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.num_training_steps * self.args.warm_up_radio, num_training_steps=self.num_training_steps)
        else:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_training_steps * self.args.warm_up_radio, num_training_steps=self.num_training_steps)
        return {
            "optimizer": optimizer, 
            "lr_scheduler":{
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }
    
    def _freeze_attention(self):
        for k, v in self.model.named_parameters():
            if "word" not in k:
                v.requires_grad = False
            else:
                print(k)
    
    def _freaze_word_embedding(self):
        for k, v in self.model.named_parameters():
            if "word" in k:
                print(k)
                v.requires_grad = False

    @staticmethod
    def add_to_argparse(parser):
        parser = BaseLitModel.add_to_argparse(parser)

        parser.add_argument("--label_smoothing", type=float, default=0.1, help="")
        parser.add_argument("--bce", type=int, default=0, help="")
        return parser
