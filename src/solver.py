import os
import math
import numpy as np
import json
import torch
from pathlib import Path
from tensorboardX import SummaryWriter
from .dataset import getDataLoader
from .module import Tacotron
from .utils import AudioProcessor, make_spec_figure, make_attn_figure
import shutil

class Solver(object):
    """Super class Solver for all kinds of tasks (train, test)"""
    def __init__(self, config, args):
        self.use_gpu = args.gpu and torch.cuda.is_available()
        self.device = torch.device('cuda') if self.use_gpu else torch.device('cpu')
        self.config = config
        self.args = args

    def verbose(self, msg):
        print(' '*100, end='\r')
        if self.args.verbose:
            print("[INFO]", msg)

    def progress(self, msg):
        if self.args.verbose:
            print(msg + ' '*40, end='\r')


class Trainer(Solver):
    """Handle training task"""
    def __init__(self, config, args):
        super(Trainer, self).__init__(config, args)
        # Best validation error, initialize it with a large number
        self.best_val_err = 1e10
        # Logger Settings
        name = Path(args.checkpoint_dir).stem
        self.log_dir = str(Path(args.log_dir, name))
        self.log_writer = SummaryWriter(self.log_dir)
        self.checkpoint_path = args.checkpoint_path
        self.checkpoint_dir = args.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.config = config
        self.audio_processor = AudioProcessor(**config['audio'])
        # Training detail
        self.step = 0
        self.max_step = config['solver']['total_steps']

    def load_data(self):
        """Load data for train and validation"""
        self.verbose("Load data")

        _config = self.config['solver']
        # Training dataset
        self.data_tr = getDataLoader(
                mode='train',
                meta_path=_config['meta_path']['train'],
                data_dir=_config['data_dir'],
                batch_size=_config['batch_size'],
                r=self.config['model']['tacotron']['r'],
                n_jobs=_config['n_jobs'],
                use_gpu=self.use_gpu)

        # Validation dataset
        self.data_va = getDataLoader(
                mode='test',
                meta_path=_config['meta_path']['test'],
                data_dir=_config['data_dir'],
                batch_size=_config['batch_size'],
                r=self.config['model']['tacotron']['r'],
                n_jobs=_config['n_jobs'],
                use_gpu=self.use_gpu)

    def build_model(self):
        """Build model"""
        self.verbose("Build model")

        self.model = Tacotron(**self.config['model']['tacotron']).to(device=self.device)
        self.criterion = torch.nn.L1Loss()

        # Optimizer
        _config = self.config['model']
        lr = _config['optimizer']['lr']
        optim_type = _config['optimizer'].pop('type', 'Adam')
        self.optim = getattr(torch.optim, optim_type)
        self.optim = self.optim(self.model.parameters(), **_config['optimizer'])
        # Load checkpoint if specify
        if self.checkpoint_path is not None:
            self.load_ckpt()

    def update_optimizer(self):
        warmup_steps = 4000.0
        step = self.step + 1.
        init_lr = self.config['model']['optimizer']['lr']
        current_lr = init_lr * warmup_steps**0.5 * np.minimum(
            step * warmup_steps**-1.5, step**-0.5)
        for param_group in self.optim.param_groups:
            param_group['lr'] = current_lr
        return current_lr

    def exec(self):
        """Train"""
        local_step = 0
        fs = self.config['audio']['sample_rate']
        linear_dim = self.model.linear_size
        n_priority_freq = int(3000 / (fs * 0.5) * linear_dim)
        self.model.train()
        self.verbose('Start training: {} batches'.format(len(self.data_tr)))
        while self.step < self.max_step:
            for curr_b, (txt, text_lengths, mel, spec) in enumerate(self.data_tr):
                # Sort data by length
                sorted_lengths, indices = torch.sort(text_lengths.view(-1), dim=0, descending=True)
                sorted_lengths = sorted_lengths.long().numpy()
                txt, mel, spec = txt[indices], mel[indices], spec[indices]

                txt = txt.to(device=self.device)
                mel = mel.to(device=self.device)
                spec = spec.to(device=self.device)

                # Decay learning rate
                current_lr = self.update_optimizer()

                # Forwarding
                self.optim.zero_grad()
                mel_outputs, linear_outputs, attn = self.model(
                        txt, mel, text_lengths=sorted_lengths)

                mel_loss = self.criterion(mel_outputs, mel)
                # Count linear loss
                linear_loss = 0.5 * self.criterion(linear_outputs, spec) \
                            + 0.5 * self.criterion(linear_outputs[:, :, :n_priority_freq], spec[:, :, :n_priority_freq])

                loss = mel_loss + linear_loss
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['solver']['grad_clip'])

                # Skip this update if grad is NaN
                if math.isnan(grad_norm):
                    self.verbose('Error : grad norm is NaN @ step ' + str(self.step))
                else:
                    self.optim.step()

                # Add to tensorboard
                if self.step % 5 == 0:
                    self.write_log('Loss', {
                        'total_loss' : loss.item(),
                        'mel_loss'   : mel_loss.item(),
                        'linear_loss': linear_loss.item()
                        })
                    self.write_log('grad_norm', grad_norm)
                    self.write_log('learning_rate', current_lr)

                if self.step % self.config['solver']['log_interval'] == 0:
                    log = '[{}] total_loss: {:.3f}. mel_loss: {:.3f}, linear_loss: {:.3f}, grad_norm: {:.3f}, lr: {:.5f}'.format(self.step, loss.item(), mel_loss.item(), linear_loss.item(), grad_norm, current_lr)
                    self.progress(log)

                if self.step % self.config['solver']['validation_interval'] == 0 and local_step != 0:
                    with torch.no_grad():
                        val_err = self.validate()
                    if val_err < self.best_val_err:
                        # Save checkpoint
                        self.save_ckpt()
                        self.best_val_err = val_err
                    self.model.train()

                if self.step % self.config['solver']['save_checkpoint_interval'] == 0 and local_step != 0:
                    self.save_ckpt()

                # Global step += 1
                self.step += 1
                local_step += 1

    def save_ckpt(self):
        ckpt_path = os.path.join(self.checkpoint_dir, "checkpoint_step{}.pth".format(self.step))
        torch.save({
            "state_dict": self.model.state_dict(),
            "optimizer": self.optim.state_dict(),
            "global_step": self.step
        }, ckpt_path)
        self.verbose("@ step {} => saved checkpoint: {}".format(self.step, ckpt_path))

    def load_ckpt(self):
        self.verbose("Load checkpoint from: {}".format(self.checkpoint_path))
        ckpt = torch.load(self.checkpoint_path)
        self.model.load_state_dict(ckpt['state_dict'])
        self.optim.load_state_dict(ckpt['optimizer'])
        self.step = ckpt['global_step']

    def write_log(self, val_name, val_dict):
        if type(val_dict) == dict:
            self.log_writer.add_scalars(val_name, val_dict, self.step)
        else:
            self.log_writer.add_scalar(val_name, val_dict, self.step)

    def validate(self):
        # (r9y9's comment) Turning off dropout of decoder's prenet causes serious performance
        # drop, not sure why.
        self.model.encoder.eval()
        self.model.postnet.eval()

        fs = self.config['audio']['sample_rate']
        linear_dim = self.model.linear_size
        n_priority_freq = int(3000 / (fs * 0.5) * linear_dim)

        mel_loss_avg = 0.0
        linear_loss_avg = 0.0
        total_loss_avg = 0.0

        for curr_b, (txt, text_lengths, mel, spec) in enumerate(self.data_va):
            # Sort data by length
            sorted_lengths, indices = torch.sort(text_lengths.view(-1), dim=0, descending=True)
            sorted_lengths = sorted_lengths.long().numpy()
            txt, mel, spec = txt[indices], mel[indices], spec[indices]

            txt = txt.to(device=self.device)
            mel = mel.to(device=self.device)
            spec = spec.to(device=self.device)

            # Forwarding
            mel_outputs, linear_outputs, attn = self.model(
                    txt, mel, text_lengths=sorted_lengths)

            mel_loss = self.criterion(mel_outputs, mel)
            # Count linear loss
            linear_loss = 0.5 * self.criterion(linear_outputs, spec) \
                        + 0.5 * self.criterion(linear_outputs[:, :, :n_priority_freq], spec[:, :, :n_priority_freq])
            loss = mel_loss + linear_loss

            mel_loss_avg += mel_loss.item()
            linear_loss_avg += linear_loss.item()
            total_loss_avg += loss.item()

            NUM_GL = 5
            if curr_b < NUM_GL:
                fig_spec = make_spec_figure(linear_outputs[0].cpu().numpy(), self.audio_processor)
                fig_attn = make_attn_figure(attn[0].cpu().numpy())
                # Predicted audio signal
                waveform = self.audio_processor.inv_spectrogram(linear_outputs[0].cpu().numpy().T)
                waveform = np.clip(waveform, -1, 1)
                # Tensorboard
                self.log_writer.add_figure('spectrogram-%d' % curr_b, fig_spec, self.step)
                self.log_writer.add_figure('attn-%d' % curr_b, fig_attn, self.step)
                self.log_writer.add_audio('wav-%d' % curr_b, waveform, self.step, sample_rate=fs)
            # Perform Griffin-Lim to generate waveform: "GL"
            header = '[GL-{}/{}]'.format(curr_b + 1, NUM_GL) if curr_b < NUM_GL else '[VAL-{}/{}]'.format(curr_b + 1, len(self.data_va))
            # Terminal log
            log = header + ' total_loss: {:.3f}. mel_loss: {:.3f}, linear_loss: {:.3f}'.format(
                    loss.item(), mel_loss.item(), linear_loss.item())

            self.progress(log)

        mel_loss_avg /= len(self.data_va)
        linear_loss_avg /= len(self.data_va)
        total_loss_avg /= len(self.data_va)

        self.verbose('@ step {} => total_loss: {:.3f}, mel_loss: {:.3f}, linear_loss: {:.3f}'.format(self.step, total_loss_avg, mel_loss_avg, linear_loss_avg))

        self.write_log('Loss', {
            'total_loss_val' : total_loss_avg,
            'mel_loss_val'   : mel_loss_avg,
            'linear_loss_val': linear_loss_avg
            })
        return linear_loss_avg



