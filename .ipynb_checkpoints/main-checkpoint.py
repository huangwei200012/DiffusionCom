import os
import torch
import argparse
import importlib
import numpy as np
import pytorch_lightning as pl
from models.modeling_clip import CLIPModel
from transformers import CLIPConfig, BertConfig, BertModel
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# In order to ensure reproducible experiments, we must set random seeds.

def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.models.MLP'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_



def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic arguments
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--litmodel_class", type=str, default="TransformerLitModel")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data_class", type=str, default="KGC")
    parser.add_argument("--chunk", type=str, default="")
    parser.add_argument("--model_class", type=str, default="RobertaUseLabelWord")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--data", type=str, default="d1015")
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=1536)
    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"data.{temp_args.data_class}")
    model_class = _import_class(f"models.{temp_args.model_class}")
    lit_model_class = _import_class(f"lit_models.{temp_args.litmodel_class}")

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    if hasattr(model_class, "add_to_argparse"):
        model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_model_class.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser


def main():
    parser = _setup_parser()
    args = parser.parse_args()
    print(args)
    vision_config = CLIPConfig.from_pretrained('clip').vision_config
    text_config = BertConfig.from_pretrained('bert')
    bert = BertModel.from_pretrained('bert')
    clip_model = CLIPModel.from_pretrained('clip')
    clip_vit = clip_model.vision_model
    
    vision_config.device = 'cpu'

    data_class = _import_class(f"data.{args.data_class}")               # Dataset
    model_class = _import_class(f"models_tu.{args.model_class}")           # Model
    litmodel_class = _import_class(f"lit_models.lit_models_tu_relation.{args.litmodel_class}") # Lit_model
    model = model_class(vision_config, text_config,args.num_layers)

 
    clip_model_dict = clip_vit.state_dict()
    text_model_dict = bert.state_dict()

    def load_state_dict():
        """Load bert and vit pretrained weights"""
        vision_names, text_names = [], []
        model_dict = model.state_dict()
        for name in model_dict:
            if 'vision' in name:
                clip_name = name.replace('vision_', '').replace('model.', '').replace('unimo.', '')
                if clip_name in clip_model_dict:
                    vision_names.append(clip_name)
                    model_dict[name] = clip_model_dict[clip_name]
            elif 'text' in name:
                text_name = name.replace('text_', '').replace('model.', '').replace('unimo.', '')
                if text_name in text_model_dict:
                    text_names.append(text_name)
                    model_dict[name] = text_model_dict[text_name]
        assert len(vision_names) == len(clip_model_dict) and len(text_names) == len(text_model_dict), \
                    (len(vision_names), len(text_names), len(clip_model_dict), len(text_model_dict))
        model.load_state_dict(model_dict)
        print('Load model state dict successful.')
    load_state_dict()

    data = data_class(args, model)
    tokenizer = data.tokenizer
    print("tokenizer.mask_token_id:",tokenizer.mask_token_id)
    lit_model = litmodel_class(args=args, model=model, tokenizer=tokenizer, data_config=data.get_config())
    if args.checkpoint:
        lit_model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["state_dict"])

    logger = pl.loggers.TensorBoardLogger("training/logs")


    metric_name = "Eval/hits10"

    # early_callback = pl.callbacks.EarlyStopping(monitor="Eval/mrr", mode="max", patience=5)
    # 保存4个checkpoint，同时保存最后一个，把早挺给去掉了。
    map_dict = {0:"no_pretrain",1:"pretrain"}



    model_checkpoint = pl.callbacks.ModelCheckpoint(monitor=metric_name, mode="max",
        filename=args.data_dir.split("/")[-1] + '/{epoch}-{Eval/hits10:.2f}-{Eval/hits1:.2f}' if not args.pretrain else args.data_dir.split("/")[-1] + '/{epoch}-{step}-{Eval/hits10:.2f}',
        dirpath="output_stage1/"+args.mode+"_{}_{}_{}_{}_{}".format(map_dict[int(args.pretrain)],str(args.lr),str(args.data),str(args.num_layers),str(args.batch_size)),
        save_weights_only=True,
        save_top_k=2,
        save_last=True,
    )


    callbacks = [model_checkpoint]
    
    trainer = pl.Trainer.from_argparse_args(args, 
                                            callbacks=callbacks, 
                                            logger=logger, 
                                            default_root_dir="training/logs",)
    # result = trainer.test(lit_model, datamodule=data)
    # print(result)
    if "EntityEmbedding" not in lit_model.__class__.__name__:
        trainer.fit(lit_model, datamodule=data)
        path = model_checkpoint.best_model_path
        lit_model.load_state_dict(torch.load(path)["state_dict"])

    result = trainer.test(lit_model, datamodule=data)
    print(result)

    # _saved_pretrain(lit_model, tokenizer, path)
    if "EntityEmbedding" not in lit_model.__class__.__name__:
        print("*path"*30)
        print(path)





if __name__ == "__main__":

    main()
