import gradio as gr
import os
import pathlib
import shutil
import json
import gc
import torch
import time
from typing import *
import argparse
import library.train_util as util
import train_network
import pathlib
''' 
list of options to ask:

base_model: path required
img_folder: path required 
output_folder: path required
save_json_folder: path|None
reg_imgs:       path|None
batch_size:int|1
num_epochs:int|1
net_dim:int|128
Alpha: float|netdim/2
train_resolution: int|512
learning_rate: float|1e-4
text_encoder_lr: float|None
unet_lr:float|None
scheduler: string
cosine_restarts: int
scheduler_power: float
save_at_n_epochs: int
shuffle_captions: bool
keep_tokens: int
warmup_lr_ratio: float
change_output_name: string
training_comment:string 
unet_only: bool
text_only: bool

components mapping:
str->textbox
int/float->number
list->radio
bool->checkbox

optional args:
save_json_folder:will be the same as output folder
load_json_path: will add this.

text_encoder_lr & unet_lr: usually are not touched
gradient_acc_steps: usually not touched,but can be used to compensate low batch size
reg_img_folder: no one uses reg img
log_dir: not touched
max_steps: calculated in find_max_step()
change_output_name:
'''
class ArgStore:
    # Represents the entirety of all possible inputs for sd-scripts. they are ordered from most important to least
    def __init__(self):
        # self.required_configs = ["base_model","img_folder","output_folder"]
        # Important, these are the most likely things you will modify
        self.base_model: str = r""  # example path, r"E:\sd\stable-diffusion-webui\models\Stable-diffusion\nai.ckpt"
        self.img_folder: str = r""  # is the folder path to your img folder, make sure to follow the guide here for folder setup: https://rentry.org/2chAI_LoRA_Dreambooth_guide_english#for-kohyas-script
        self.output_folder: str = r""  # just the folder all epochs/safetensors are output
        self.change_output_name: Union[str, None] = ""  # changes the output name of the epochs
        self.save_json_folder: Union[str, None] = None  # OPTIONAL, saves a json folder of your config to whatever location you set here.
        self.load_json_path: Union[str, None] = None  # OPTIONAL, loads a json file partially changes the config to match. things like folder paths do not get modified.
        self.json_load_skip_list: Union[list[str], None] = ["base_model", "img_folder", "output_folder",
                                                            "save_json_folder", "reg_img_folder", "lora_model_for_resume",
                                                            "change_output_name", "training_comment", "json_load_skip_list"]  # OPTIONAL, allows the user to define what they skip when loading a json, by default it loads everything, including all paths, set it up like this ["base_model", "img_folder", "output_folder"]

        self.net_dim: int = 128  # network dimension, 128 is the most common, however you might be able to get lesser to work
        self.alpha: float = 128  # represents the scalar for training. the lower the alpha, the less gets learned per step. if you want the older way of training, set this to dim
        # list of schedulers: linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup
        # self.scheduler: str = "cosine_with_restarts"  # the scheduler for learning rate. Each does something specific
        self.scheduler = "cosine_with_restarts" #["cosine_with_restarts", "constant_with_warmup","linear", "cosine", "polynomial", "constant"]
        self.cosine_restarts: Union[int, None] = 1  # OPTIONAL, represents the number of times it restarts. Only matters if you are using cosine_with_restarts
        self.scheduler_power: Union[float, None] = 1  # OPTIONAL, represents the power of the polynomial. Only matters if you are using polynomial
        self.warmup_lr_ratio: Union[float, None] = None  # OPTIONAL, Calculates the number of warmup steps based on the ratio given. Make sure to set this if you are using constant_with_warmup, None to ignore
        self.learning_rate: Union[float, None] = 1e-4  # OPTIONAL, when not set, lr gets set to 1e-3 as per adamW. Personally, I suggest actually setting this as lower lr seems to be a small bit better.
        self.text_encoder_lr: Union[float, None] = None  # OPTIONAL, Sets a specific lr for the text encoder, this overwrites the base lr I believe, None to ignore
        self.unet_lr: Union[float, None] = None  # OPTIONAL, Sets a specific lr for the unet, this overwrites the base lr I believe, None to ignore
        self.num_workers: int = 1  # The number of threads that are being used to load images, lower speeds up the start of epochs, but slows down the loading of data. The assumption here is that it increases the training time as you reduce this value
        self.persistent_workers: bool = True  # makes workers persistent, further reduces/eliminates the lag in between epochs. however it may increase memory usage

        self.batch_size: int = 1  # The number of images that get processed at one time, this is directly proportional to your vram and resolution. with 12gb of vram, at 512 reso, you can get a maximum of 6 batch size
        self.num_epochs: int = 1  # The number of epochs, if you set max steps this value is ignored as it doesn't calculate steps.
        self.save_at_n_epochs: int = 2  # OPTIONAL, how often to save epochs, None to ignore
        self.shuffle_captions: bool = False  # OPTIONAL, False to ignore
        self.keep_tokens: int = 1  # OPTIONAL, None to ignore
        self.max_steps: Union[int, None] = None  # OPTIONAL, if you have specific steps you want to hit, this allows you to set it directly. None to ignore

        # These are the second most likely things you will modify
        self.train_resolution: int = 512
        self.min_bucket_resolution: int = 320
        self.max_bucket_resolution: int = 960
        self.lora_model_for_resume: Union[str, None] = None  # OPTIONAL, takes an input lora to continue training from, not exactly the way it *should* be, but it works, None to ignore
        self.save_state: bool = False  # OPTIONAL, is the intended way to save a training state to use for continuing training, False to ignore
        self.load_previous_save_state: Union[str, None] = None  # OPTIONAL, is the intended way to load a training state to use for continuing training, None to ignore
        self.training_comment: str = ""  # OPTIONAL, great way to put in things like activation tokens right into the metadata. seems to not work at this point and time
        self.unet_only: bool = False  # OPTIONAL, set it to only train the unet
        self.text_only: bool = False  # OPTIONAL, set it to only train the text encoder

        # These are the least likely things you will modify
        self.reg_img_folder: Union[str, None] = None  # OPTIONAL, None to ignore
        self.clip_skip: int = 2  # If you are training on a model that is anime based, keep this at 2 as most models are designed for that
        self.test_seed: int = 23  # this is the "reproducable seed", basically if you set the seed to this, you should be able to input a prompt from one of your training images and get a close representation of it
        self.prior_loss_weight: float = 1  # is the loss weight much like Dreambooth, is required for LoRA training
        self.gradient_checkpointing: bool = False  # OPTIONAL, enables gradient checkpointing
        self.gradient_acc_steps: Union[int, None] = 1  # OPTIONAL, not sure exactly what this means
        self.mixed_precision ="fp16"  # If you have the ability to use bf16, do it, it's better
        self.save_precision  ="fp16"  # You can also save in bf16, but because it's not universally supported, I suggest you keep saving at fp16
        self.save_as: str = "safetensors"  # list is pt, ckpt, safetensors
        self.caption_extension: str = ".txt"  # the other option is .captions, but since wd1.4 tagger outputs as txt files, this is the default
        self.max_clip_token_length:int = 150  # can be 75, 150, or 225 I believe, there is no reason to go higher than 150 though
        self.buckets: bool = True
        self.xformers: bool = True
        self.use_8bit_adam: bool = True
        self.cache_latents: bool = True
        self.color_aug: bool = False  # IMPORTANT: Clashes with cache_latents, only have one of the two on!
        self.flip_aug: bool = False
        self.vae: Union[str, None] = None  # Seems to only make results worse when not using that specific vae, should probably not use
        self.no_meta: bool = False  # This removes the metadata that now gets saved into safetensors, (you should keep this on)
        self.log_dir: Union[str, None] = None  # output of logs, not useful to most people.

    # Creates the dict that is used for the rest of the code, to facilitate easier json saving and loading
    @staticmethod
    def convert_args_to_dict():
        return ArgStore().__dict__
def add_misc_args(parser):
    parser.add_argument("--save_json_path", type=str, default=None,
                        help="Path to save a configuration json file to")
    parser.add_argument("--load_json_path", type=str, default=None,
                        help="Path to a json file to configure things from")
    parser.add_argument("--no_metadata", action='store_true',
                        help="do not save metadata in output model / メタデータを出力先モデルに保存しない")
    parser.add_argument("--save_model_as", type=str, default="safetensors", choices=[None, "ckpt", "pt", "safetensors"],
                        help="format to save the model (default is .safetensors) / モデル保存時の形式（デフォルトはsafetensors）")

    parser.add_argument("--unet_lr", type=float, default=None, help="learning rate for U-Net / U-Netの学習率")
    parser.add_argument("--text_encoder_lr", type=float, default=None,
                        help="learning rate for Text Encoder / Text Encoderの学習率")
    parser.add_argument("--lr_scheduler_num_cycles", type=int, default=1,
                        help="Number of restarts for cosine scheduler with restarts / cosine with restartsスケジューラでのリスタート回数")
    parser.add_argument("--lr_scheduler_power", type=float, default=1,
                        help="Polynomial power for polynomial scheduler / polynomialスケジューラでのpolynomial power")

    parser.add_argument("--network_weights", type=str, default=None,
                        help="pretrained weights for network / 学習するネットワークの初期重み")
    parser.add_argument("--network_module", type=str, default=None,
                        help='network module to train / 学習対象のネットワークのモジュール')
    parser.add_argument("--network_dim", type=int, default=None,
                        help='network dimensions (depends on each network) / モジュールの次元数（ネットワークにより定義は異なります）')
    parser.add_argument("--network_alpha", type=float, default=1,
                        help='alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version) / LoRaの重み調整のalpha値、デフォルト1（旧バージョンと同じ動作をするにはnetwork_dimと同じ値を指定）')
    parser.add_argument("--network_args", type=str, default=None, nargs='*',
                        help='additional argmuments for network (key=value) / ネットワークへの追加の引数')
    parser.add_argument("--network_train_unet_only", action="store_true",
                        help="only training U-Net part / U-Net関連部分のみ学習する")
    parser.add_argument("--network_train_text_encoder_only", action="store_true",
                        help="only training Text Encoder part / Text Encoder関連部分のみ学習する")
    parser.add_argument("--training_comment", type=str, default=None,
                        help="arbitrary comment string stored in metadata / メタデータに記録する任意のコメント文字列")
def setup_args(parser):
    util.add_sd_models_arguments(parser)
    util.add_dataset_arguments(parser, True, True)
    util.add_training_arguments(parser, True)
    add_misc_args(parser)
def create_optional_args(args: dict, steps):
    output = []
    if args["reg_img_folder"]:
        output.append(f"--reg_data_dir={args['reg_img_folder']}")

    if args['lora_model_for_resume']:
        output.append(f"--network_weights={args['lora_model_for_resume']}")

    if args['save_at_n_epochs']:
        output.append(f"--save_every_n_epochs={args['save_at_n_epochs']}")
    else:
        output.append("--save_every_n_epochs=999999")

    if args['shuffle_captions']:
        output.append("--shuffle_caption")

    if args['keep_tokens'] and args['keep_tokens'] > 0:
        output.append(f"--keep_tokens={args['keep_tokens']}")

    if args['buckets']:
        output.append("--enable_bucket")
        output.append(f"--min_bucket_reso={args['min_bucket_resolution']}")
        output.append(f"--max_bucket_reso={args['max_bucket_resolution']}")

    if args['use_8bit_adam']:
        output.append("--use_8bit_adam")

    if args['xformers']:
        output.append("--xformers")

    if args['color_aug']:
        if args['cache_latents']:
            print("color_aug and cache_latents conflict with one another. Please select only one")
            quit(1)
        output.append("--color_aug")

    if args['flip_aug']:
        output.append("--flip_aug")

    if args['cache_latents']:
        output.append("--cache_latents")

    if args['warmup_lr_ratio'] and args['warmup_lr_ratio'] > 0:
        warmup_steps = int(steps * args['warmup_lr_ratio'])
        output.append(f"--lr_warmup_steps={warmup_steps}")

    if args['gradient_checkpointing']:
        output.append("--gradient_checkpointing")

    if args['gradient_acc_steps'] and args['gradient_acc_steps'] > 0 and args['gradient_checkpointing']:
        output.append(f"--gradient_accumulation_steps={args['gradient_acc_steps']}")

    if args['learning_rate'] and args['learning_rate'] > 0:
        output.append(f"--learning_rate={args['learning_rate']}")

    if args['text_encoder_lr'] and args['text_encoder_lr'] > 0:
        output.append(f"--text_encoder_lr={args['text_encoder_lr']}")

    if args['unet_lr'] and args['unet_lr'] > 0:
        output.append(f"--unet_lr={args['unet_lr']}")

    if args['vae']:
        output.append(f"--vae={args['vae']}")

    if args['no_meta']:
        output.append("--no_metadata")

    if args['save_state']:
        output.append("--save_state")

    if args['load_previous_save_state']:
        output.append(f"--resume={args['load_previous_save_state']}")

    if args['change_output_name']:
        output.append(f"--output_name={args['change_output_name']}")

    if args['training_comment']:
        output.append(f"--training_comment={args['training_comment']}")

    if args['cosine_restarts'] and args['scheduler'] == "cosine_with_restarts":
        output.append(f"--lr_scheduler_num_cycles={args['cosine_restarts']}")

    if args['scheduler_power'] and args['scheduler'] == "polynomial":
        output.append(f"--lr_scheduler_power={args['scheduler_power']}")

    if args['persistent_workers']:
        output.append(f"--persistent_data_loader_workers")

    if args['unet_only']:
        output.append("--network_train_unet_only")

    if args['text_only'] and not args['unet_only']:
        output.append("--network_train_text_encoder_only")
    return output
def find_max_steps(args: dict) -> int:
    total_steps = 0
    folders = os.listdir(args["img_folder"])
    for folder in folders:
        if not os.path.isdir(os.path.join(args["img_folder"], folder)):
            continue
        num_repeats = folder.split("_")
        if len(num_repeats) < 2:
            print(f"folder {folder} is not in the correct format. Format is x_name. skipping")
            continue
        try:
            num_repeats = int(num_repeats[0])
        except ValueError:
            print(f"folder {folder} is not in the correct format. Format is x_name. skipping")
            continue
        imgs = 0
        for file in os.listdir(os.path.join(args["img_folder"], folder)):
            if os.path.isdir(file):
                continue
            ext = file.split(".")
            if ext[-1].lower() in {"png", "bmp", "gif", "jpeg", "jpg", "webp"}:
                imgs += 1
        total_steps += (num_repeats * imgs)
    total_steps = int((total_steps / args["batch_size"]) * args["num_epochs"])
    return total_steps
def create_arg_space(args: dict) -> [str]:
    # This is the list of args that are to be used regardless of setup
    output = ["--network_module=networks.lora", f"--pretrained_model_name_or_path={args['base_model']}",
              f"--train_data_dir={args['img_folder']}", f"--output_dir={args['output_folder']}",
              f"--prior_loss_weight={args['prior_loss_weight']}", f"--caption_extension=" + args['caption_extension'],
              f"--resolution={args['train_resolution']}", f"--train_batch_size={args['batch_size']}",
              f"--mixed_precision={args['mixed_precision']}", f"--save_precision={args['save_precision']}",
              f"--network_dim={args['net_dim']}", f"--save_model_as={args['save_as']}",
              f"--clip_skip={args['clip_skip']}", f"--seed={args['test_seed']}",
              f"--max_token_length={args['max_clip_token_length']}", f"--lr_scheduler={args['scheduler']}",
              f"--network_alpha={args['alpha']}", f"--max_data_loader_n_workers={args['num_workers']}"]
    if not args['max_steps']:
        output.append(f"--max_train_epochs={args['num_epochs']}")
        output += create_optional_args(args, find_max_steps(args))
    else:
        output.append(f"--max_train_steps={args['max_steps']}")
        output += create_optional_args(args, args['max_steps'])
    return output

class gradio_ui(object):
    def __init__(self):

        argstore = ArgStore()
        ui_map = dict()
        hint_map = dict()
        alias_map = dict()
        self.argstore = argstore
        self.ui_map = ui_map
        self.hint_map = hint_map
        self.alias_map = alias_map
        self.customize_components = (
            'base_model',
            "img_folder","output_folder",
            "caption_extension","change_output_name",
            'net_dim','alpha',
            'scheduler',
            'cosine_restarts','scheduler_power',
            'learning_rate','num_workers','batch_size',
                             'persistent_workers',
            'num_epochs','save_at_n_epochs',
            'unet_only','text_only','gradient_checkpointing',
            'mixed_precision','save_precision',
            'clip_skip','max_clip_token_length'

        )
        self.hidden_components = (
            'min_bucket_resolution',
            'max_bucket_resolution',
            'test_seed','prior_loss_weight',
            'save_as',
            'xformers','buckets','use_8bit_adam','no_meta',
            'mixed_precision','save_precision',
            'json_load_skip_list'
        )
        self.config_path = 'sd_lora_ui_config.json'
        self.config = {
            'SD_dir':None,
            'json_path':None,
        }
        self.SD_models = ['None']
        self.load_config()

        self.load_hints()
        self.load_alias()
        self.register_components()




    def test(self):
        print('WTF')
    def load_alias(self):
        alias_map = self.alias_map
        alias_map = {
            'base_model':'Base model \n(required)',
            'img_folder':'Dataset folder \n(required)',
            'output_folder':'Output path \n(required)',
            'change_output_name': 'Output name',
            'net_dim':'Network dimension \n(128 is the default, but less can work)',
            'alpha':'Alpha \n(represents how much gets learned per step,usually equal to dim)',
            'cosine_restarts':'Cosine restarts \n(represents the number of times it restarts. Only matters if you are using cosine_with_restarts as scheduler)',
            'scheduler_power':'Scheduler Power \n(represents the power of the polynomial. Only matters if you are using polynomial as scheduler)',
            'num_workers':'Num workers \n(depends on your CPU)',
            'batch_size':'batch size \n(depends on your GPU)',
            'unet_only':'unet_only \n(only train unet)',
            'text_only':'text_only \n(only train text encoder)',

        }
        self.alias_map = alias_map
    def load_hints(self):
        hint_map = self.hint_map
        hint_map = {
            'base_model':"Path to the base SD model(.ckpt or .safetensors)",
            'img_folder':"Path to the dataset folder",
            'output_folder':"Path to the folder where the lora will be saved",
            'training_comment':'Add your comment into the metadata of the lora',
            'caption_extension':'The extension of your training caption file, usually .txt',
            'change_output_name':'Name of your model'
        }
        self.hint_map = hint_map
    def register_components(self):
        argdict = self.argstore.__dict__
        options = self.argstore.__dict__.keys()
        ui_map = self.ui_map
        for key in options:
            arg = argdict[key]
            visibility = key not in self.hidden_components
            if type(arg)==bool:
                ui_map[key] = gr.Checkbox(value=arg,label=key,interactive=True,visible=visibility)
            elif type( arg )==int or type(arg)==float:
                ui_map[key] = gr.Number(value=arg, label=key,interactive=True,visible=visibility)
            elif type( arg )==str :
                ui_map[key] = gr.Textbox(value=arg, label=key,interactive=True,visible=visibility)
            elif type(arg) == list:
                ui_map[key] = gr.Radio(choices=arg, value=arg[0],label=key,interactive=True,visible=visibility)
        for key,hint in self.hint_map.items():
            ui_map[key].placeholder = hint
        for key,alias in self.alias_map.items():
            ui_map[key].label = alias
        print(len(argdict))
        print(len(ui_map))
        s1 = set(argdict.keys())
        s2 = set(ui_map.keys())
        print(s1-s2)
    # def save_json(path, obj: dict) -> None:
    #     fp = open(os.path.join(path, f"config-{time.time()}.json"), "w")
    #     json.dump(obj, fp=fp, indent=4)
    #     fp.close()
    def grab_models(self,dir):
        d = pathlib.Path(dir)
        if d.is_dir():
            suffixes = ('.safetensors','.ckpt')
            files = list(d.glob('**/*'))
            models = [x for x in files if x.suffix in suffixes]
            self.SD_models = self.SD_models+models
            self.config['SD_dir'] = str(d.resolve())
            self.save_config()
            hint = f"Successfully loaded {len(models)} models!"
        else:
            hint = f"Failed to load model from directory {dir}!"
        return gr.Markdown.update(value=hint),gr.Dropdown.update(choices=self.SD_models)
    def select_model(self,model):
        return model
    def clear_parameters(self):
        print("Restore default values of variables")
        self.argstore = ArgStore()
        argdict = self.argstore.__dict__
        ret = []
        for key in self.ui_map.keys():
            ret.append(argdict[key])
            # self.ui_map[key].update(value = argdict[key])
        print("Reset values:",self.ui_map.keys())
        return ret
    def save_config(self):
        config = {
            'config':self.config,
            'argdict':self.argstore.__dict__
        }
        with open(self.config_path,'w') as f:
            json.dump(config,f)
    def load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path,'r') as f:
                try:
                    config = json.load(f)
                    self.config = config['config']
                    self.argstore.__dict__.update(config['argdict'])
                    print(config)
                except:
                    print("Failed to load config")
    def update_argstore(self,*args):
        print(args)
        argdict = self.argstore.__dict__
        keys = list(self.ui_map.keys())
        for i in range(len(args)):
            key = keys[i]
            arg = args[i]
            print(keys, args)
            # gradio Number defaulted to float, need to cast it back to int
            if type(argdict[key])==int:
                arg = int(arg)
            argdict[ key ]=arg
        self.argstore.__dict__.update(argdict)
    def save_params(self,*args):
        self.update_argstore(*args)
        self.save_config()

    def start_training(self,*args):
        self.update_argstore(*args)
        argdict = self.argstore.__dict__
        print(argdict)

        args = create_arg_space(argdict)
        parser = argparse.ArgumentParser()
        setup_args(parser)
        args = parser.parse_args(args)
        try:
            train_network.train(args)
        except Exception as e:
            print(e)
            print("Failed to launch training. ")
        gc.collect()
        torch.cuda.empty_cache()
        return self.clear_parameters()
    def interface(self):
        train_btn = gr.Button(variant="primary",value="Start training")
        clear_btn = gr.Button(value = "Clear parameters")
        save_para_btn = gr.Button(value="Save parameters")

        sd_dir = gr.Textbox(label="SD Folder",value=self.config['SD_dir'], placeholder="Load your SD base models here",interactive=True)
        sd_dir_submit = gr.Button(value="Load",interactive=True)
        sd_dir_hint = gr.Markdown()
        sd_selection = gr.Dropdown(choices=self.SD_models,value='None',label="Base model")

        with gr.Blocks() as demo:

            with gr.Row():
                train_btn.render()
                clear_btn.render()
            with gr.Row():
                save_para_btn.render()
            with gr.Row():
                sd_dir.render()
                sd_dir_submit.render()
            with gr.Row():
                with gr.Column():
                    sd_dir_hint.render()
                    sd_selection.render()

            # with gr.Row():
            #     self.ui_map['base_model'].render()
            # with gr.Row():
            #     with gr.Column():
            #         self.ui_map['img_folder'].render()
            #         self.ui_map['caption_extension'].render()
            #     with gr.Column():
            #         self.ui_map['output_folder'].render()
            #         self.ui_map['change_output_name'].render()
            # with gr.Row():
            #     self.ui_map['net_dim'].render()
            #     self.ui_map['alpha'].render()
            # with gr.Row():
            #     self.ui_map['scheduler'].render()
            # with gr.Row():
            #     self.ui_map['cosine_restarts'].render()
            #     self.ui_map['scheduler_power'].render()
            # with gr.Row():
            #     with gr.Box():
            #         self.ui_map['learning_rate'].render()
            #     with gr.Box():
            #         self.ui_map['num_workers'].render()
            #         self.ui_map['persistent_workers'].render()
            #     with gr.Box():
            #         self.ui_map['batch_size'].render()
            # with gr.Row():
            #     self.ui_map['clip_skip'].render()
            #     self.ui_map['max_clip_token_length'].render()
            # with gr.Row():
            #     self.ui_map['unet_only'].render()
            #     self.ui_map['text_only'].render()
            #     self.ui_map['gradient_checkpointing'].render()
            # with gr.Row():
            #     self.ui_map['mixed_precision'].render()
            #     self.ui_map['save_precision'].render()
            """
            gradio kinda fucked up here. fuck gradio.
            """
            # keys_rem = set(self.ui_map.keys())-set(self.customize_components)
            # for key in keys_rem:
            #     self.ui_map[key].render()

            keys = [x for x in self.ui_map.keys()]
            # keys = sorted(keys)
            #         # if x not in self.customize_components and x not in self.hidden_components]
            for i in range(len(keys)):
                    self.ui_map[keys[i]].render()

            clear_btn.click(fn = self.clear_parameters ,outputs=list(self.ui_map.values()))
            train_btn.click(fn = self.start_training ,inputs = list(self.ui_map.values()))
            save_para_btn.click(fn = self.save_params,inputs = list(self.ui_map.values()))
            sd_dir_submit.click(fn=self.grab_models,inputs=sd_dir,outputs=[sd_dir_hint,sd_selection])
            sd_selection.change(fn = self.select_model,inputs = sd_selection,outputs=self.ui_map['base_model'])
        demo.launch(server_port=2222)
if __name__ == "__main__":
    demo = gradio_ui()
    demo.interface()

# def train():
#     parser = argparse.ArgumentParser()
#     setup_args(parser)
#     args = parser.parse_args()
#     try:
#         train_network.train(args)
#     except Exception as e:
#         print(f"Failed to train this set of args.\nSkipping this training session.\nError is: {e}")
#     gc.collect()
#     torch.cuda.empty_cache()

