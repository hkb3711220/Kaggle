import os
import time
import warnings
from glob import glob

import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.data import DataLoader

from data import *
from utils import *

warnings.filterwarnings("ignore")


class Fitter:

    def __init__(self, encoder_model, decoder_model, device, tokenizer, config):
        self.config = config
        self.epoch = 0
        self.tokenizer = tokenizer
        self.device = device
        self.valid_labels = config.valid_labels
        self.base_dir = f'./{config.folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.log_path = f'{self.base_dir}/log.txt'
        self.best_score = 10 ** 5

        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.encoder_model.to(self.device)
        self.decoder_model.to(self.device)

        self.use_mix_precision = config.use_mix_precision
        self.accumulate_steps = config.accumulate_steps

        self.clip_grad = config.clip_grad
        self.max_norm = 5.0

        self.criterion = nn.CrossEntropyLoss(ignore_index=config.pad_index)
        self.encoder_optimizer = Adam(self.encoder_model.parameters(),
                                      lr=config.encoder_lr, weight_decay=config.weight_decay)
        self.decoder_optimizer = Adam(self.decoder_model.parameters(),
                                      lr=config.decoder_lr, weight_decay=config.weight_decay)

        self.encoder_scheduler = config.SchedulerClass(self.encoder_optimizer, **config.scheduler_params)
        self.decoder_scheduler = config.SchedulerClass(self.decoder_optimizer, **config.scheduler_params)

        if self.use_mix_precision:
            self.scaler = GradScaler()

        self.log(f'Fitter prepared. Device is {self.device}')

    def fit(self, train_loader, validation_loader):
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                encoder_lr = self.encoder_optimizer.param_groups[0]['lr']
                decoder_lr = self.decoder_optimizer.param_groups[0]['lr']
                self.log(f'EPOCH: {e}, ENCODER LR: {encoder_lr}, DECODER LR: {decoder_lr}')

            t = time.time()
            summary_loss = self.train_one_epoch(train_loader)

            self.log(
                f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, '
                f'time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            predictions = self.validation(validation_loader)
            self.log(f"ground truth: {self.valid_labels[:5]}")
            self.log(f"predictions: {predictions[:5]}")
            score = get_score(self.valid_labels, predictions)
            self.log(
                f'[RESULT]: Val. Epoch: {self.epoch}, score: {score:.5f}, time: {(time.time() - t):.5f}')
            if score < self.best_score:
                self.best_score = score
                self.encoder_model.eval()
                self.decoder_model.eval()
                self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                    os.remove(path)

            if self.config.validation_scheduler:
                self.encoder_scheduler.step()
                self.decoder_scheduler.step()

            self.epoch += 1

    def validation(self, val_loader):
        self.encoder_model.eval()
        self.decoder_model.eval()
        t = time.time()
        all_text_predictions = []
        for step, (images, labels, decode_lengths) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                images = images.to(self.device).float()
                encoder_output = self.encoder_model(images)
                predictions = self.decoder_model.predict(encoder_output)
            predictions = predictions.detach().cpu().numpy()
            text_predictions = self.tokenizer.predict_captions(predictions)
            all_text_predictions.append(text_predictions)
        all_text_predictions = np.concatenate(all_text_predictions)
        all_text_predictions = [f"InChI=1S/{text}" for text in all_text_predictions]

        return all_text_predictions

    def train_one_epoch(self, train_loader):
        self.encoder_model.train()
        self.decoder_model.train()
        summary_loss = AverageMeter()
        t = time.time()
        self.encoder_optimizer.zero_grad()  # very important
        self.decoder_optimizer.zero_grad()
        for step, (images, labels, decode_lengths) in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )

            images = images.to(self.device).float()
            labels = labels.to(self.device)
            decode_lengths = decode_lengths.to(self.device)
            # batch_size = images.shape[0]
            if self.use_mix_precision:
                with autocast():
                    encoder_output = self.encoder_model(images)
                    train_output, sorted_labels, sorted_decode_lengths = self.decoder_model(encoder_output,
                                                                                            labels=labels,
                                                                                            caption_lengths=decode_lengths)

                    # max_decode_lengths = train_output.size(1)
                    # vocab_size = train_output.size(-1)
                    # loss = self.criterion(train_output.reshape(batch_size * max_decode_lengths, vocab_size),
                    #                       sorted_labels.reshape(-1))
                    loss = seq_cross_entropy_loss(train_output, sorted_labels, sorted_decode_lengths)
                    loss /= self.accumulate_steps
                self.scaler.scale(loss).backward()
                if (step + 1) % self.accumulate_steps == 0:  # Wait for several backward steps
                    self.scaler.step(self.encoder_optimizer)
                    self.scaler.step(self.decoder_optimizer)
                    self.scaler.update()
                    self.encoder_optimizer.zero_grad()
                    self.decoder_optimizer.zero_grad()
            else:
                encoder_output = self.encoder_model(images)
                train_output, sorted_labels = self.decoder_model(encoder_output, labels=labels,
                                                                 caption_lengths=decode_lengths)

                # max_decode_lengths = train_output.size(1)
                # vocab_size = train_output.size(-1)
                # loss = self.criterion(train_output.reshape(batch_size * max_decode_lengths, vocab_size),
                #                       sorted_labels.reshape(-1))
                # loss /= self.accumulate_steps
                loss = seq_cross_entropy_loss(train_output, sorted_labels, sorted_decode_lengths)
                loss /= self.accumulate_steps
                loss.backward()
                if (step + 1) % self.accumulate_steps == 0:  # Wait for several backward steps
                    self.encoder_optimizer.step()
                    self.decoder_optimizer.step()
                    self.encoder_optimizer.zero_grad()
                    self.decoder_optimizer.zero_grad()

            summary_loss.update(loss.detach().item() * self.accumulate_steps)
            if self.config.step_scheduler and (step + 1) % self.accumulate_steps == 0:
                self.encoder_scheduler.step()
                self.decoder_scheduler.step()

        return summary_loss

    def save(self, path):
        torch.save({'encoder': self.encoder_model.state_dict(),
                    'encoder_optimizer': self.encoder_optimizer.state_dict(),
                    'encoder_scheduler': self.encoder_scheduler.state_dict(),
                    'decoder': self.decoder_model.state_dict(),
                    'decoder_optimizer': self.decoder_optimizer.state_dict(),
                    'decoder_scheduler': self.decoder_scheduler.state_dict(),
                    }, path)

    # def load(self, path):
    #     checkpoint = torch.load(path)
    #     self.model.model.load_state_dict(checkpoint['model_state_dict'])

    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')


class FitterV2:

    def __init__(self, model, device, tokenizer, config):
        self.config = config
        self.epoch = 0
        self.tokenizer = tokenizer
        self.device = device
        self.valid_labels = config.valid_labels
        self.base_dir = f'./{config.folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.log_path = f'{self.base_dir}/log.txt'
        self.best_score = 10 ** 5

        self.model = model
        # self.decoder_model = decoder_model
        self.model.to(self.device)
        # self.decoder_model.to(self.device)

        self.use_mix_precision = config.use_mix_precision
        self.accumulate_steps = config.accumulate_steps

        self.clip_grad = config.clip_grad
        self.max_norm = 10.0

        # self.criterion = nn.CrossEntropyLoss(ignore_index=config.pad_index)
        self.optimizer = Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        # self.decoder_optimizer = Adam(self.decoder_model.parameters(),
        #                               lr=config.decoder_lr, weight_decay=config.weight_decay)

        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
        # self.decoder_scheduler = config.SchedulerClass(self.decoder_optimizer, **config.scheduler_params)

        if self.use_mix_precision:
            self.scaler = GradScaler()

        self.log(f'Fitter prepared. Device is {self.device}')

    def fit(self, train_loader, validation_loader):
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                # decoder_lr = self.decoder_optimizer.param_groups[0]['lr']
                self.log(f'EPOCH: {e}, LR: {lr}')

            t = time.time()
            summary_loss = self.train_one_epoch(train_loader)

            self.log(
                f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            predictions = self.validation(self.model, validation_loader)
            self.log(f"ground truth: {self.valid_labels[:5]}")
            self.log(f"predictions: {predictions[:5]}")
            score = get_score(self.valid_labels, predictions)
            self.log(
                f'[RESULT]: Val. Epoch: {self.epoch}, score: {score:.5f}, time: {(time.time() - t):.5f}')
            if score < self.best_score:
                self.best_score = score
                self.model.eval()
                # self.decoder_model.eval()
                self.save(f'{self.base_dir}/{self.config.fold_num}-best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.base_dir}/{self.config.fold_num}-best-checkpoint-*epoch.bin'))[:-3]:
                    os.remove(path)

            if self.config.validation_scheduler:
                self.scheduler.step()
                # self.decoder_scheduler.step()

            self.epoch += 1

    def validation(self, model, val_loader):
        model.eval()
        # self.decoder_model.eval()
        t = time.time()
        all_text_predictions = []
        for step, (images) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                images = images.to(self.device).float()
                predictions = self.model.predict(images)
                # predictions = self.decoder_model.predict(encoder_output)
            predictions = predictions.detach().cpu().numpy()
            text_predictions = self.tokenizer.predict_captions(predictions)
            all_text_predictions.append(text_predictions)
        all_text_predictions = np.concatenate(all_text_predictions)
        all_text_predictions = [f"InChI=1S/{text}" for text in all_text_predictions]

        return all_text_predictions

    def train_one_epoch(self, train_loader):
        self.model.train()
        # self.decoder_model.train()
        summary_loss = AverageMeter()
        t = time.time()
        self.optimizer.zero_grad()  # very important
        # self.decoder_optimizer.zero_grad()
        for step, (images, labels, decode_lengths) in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )

            images = images.to(self.device).float()
            labels = labels.to(self.device)
            decode_lengths = decode_lengths.to(self.device)
            # batch_size = images.shape[0]
            if self.use_mix_precision:
                with autocast():
                    # encoder_output = self.encoder_model(images)
                    train_output, sorted_labels, sorted_decode_lengths = self.model(images,
                                                                                    labels=labels,
                                                                                    decode_lengths=decode_lengths)

                    # max_decode_lengths = train_output.size(1)
                    # vocab_size = train_output.size(-1)
                    # loss = self.criterion(train_output.reshape(batch_size * max_decode_lengths, vocab_size),
                    #                       sorted_labels.reshape(-1))
                    loss = seq_cross_entropy_loss(train_output, sorted_labels, sorted_decode_lengths)
                    loss /= self.accumulate_steps
                self.scaler.scale(loss).backward()
                if (step + 1) % self.accumulate_steps == 0:  # Wait for several backward steps
                    # if self.clip_grad:
                    #     self.scaler.unscale_(self.optimizer)
                    #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                    self.scaler.step(self.optimizer)
                    # self.scaler.step(self.decoder_optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    # self.decoder_optimizer.zero_grad()
            else:
                train_output, sorted_labels, sorted_decode_lengths = self.model(images,
                                                                                labels=labels,
                                                                                decode_lengths=decode_lengths)
                loss = seq_cross_entropy_loss(train_output, sorted_labels, sorted_decode_lengths)
                loss /= self.accumulate_steps
                # max_decode_lengths = train_output.size(1)
                # vocab_size = train_output.size(-1)
                # loss = self.criterion(train_output.reshape(batch_size * max_decode_lengths, vocab_size),
                #                       sorted_labels.reshape(-1))
                loss /= self.accumulate_steps
                loss.backward()
                if (step + 1) % self.accumulate_steps == 0:  # Wait for several backward steps
                    self.optimizer.step()
                    # self.decoder_optimizer.step()
                    self.optimizer.zero_grad()
                    # self.decoder_optimizer.zero_grad()

            summary_loss.update(loss.detach().item() * self.accumulate_steps)
            if self.config.step_scheduler and (step + 1) % self.accumulate_steps == 0:
                self.scheduler.step()
                # self.scheduler.step()

        return summary_loss

    def save(self, path):
        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    # 'decoder': self.decoder_model.state_dict(),
                    # 'decoder_optimizer': self.decoder_optimizer.state_dict(),
                    # 'decoder_scheduler': self.decoder_scheduler.state_dict(),
                    }, path)

    # def load(self, path):
    #     checkpoint = torch.load(path)
    #     self.model.model.load_state_dict(checkpoint['model_state_dict'])

    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')

class FitterV3:

    def __init__(self, model, device, tokenizer, config):
        self.config = config
        self.epoch = 0
        self.tokenizer = tokenizer
        self.device = device
        self.valid_labels = config.valid_labels
        self.base_dir = f'./{config.folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.log_path = f'{self.base_dir}/log.txt'
        self.best_score = 10 ** 5

        self.model = model
        self.model.to(self.device)

        self.use_mix_precision = config.use_mix_precision
        self.accumulate_steps = config.accumulate_steps

        self.clip_grad = config.clip_grad
        self.max_norm = 10.0
        self.optimizer = Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)

        if self.use_mix_precision:
            self.scaler = GradScaler()

        self.log(f'Fitter prepared. Device is {self.device}')

    def fit(self, train_loader, validation_loader):
        summary_loss = AverageMeter()
        t = time.time()
        step = 0
        while step < self.config.num_steps:
            for images, labels, decode_lengths in train_loader:
                if self.config.verbose:
                    if step % self.config.verbose_step == 0:
                        print(
                            f'Train Step {step}/{self.config.num_steps}, ' + \
                            f'summary_loss: {summary_loss.avg:.5f}, ' + \
                            f'time: {(time.time() - t):.5f}', end='\r'
                        )

                self.model.train()
                images = images.to(self.device).float()
                labels = labels.to(self.device)
                decode_lengths = decode_lengths.to(self.device)
                if self.use_mix_precision:
                    with autocast():
                        train_output, sorted_labels, sorted_decode_lengths = self.model(images,
                                                                                        labels=labels,
                                                                                        decode_lengths=decode_lengths)
                        loss = seq_cross_entropy_loss(train_output, sorted_labels, sorted_decode_lengths)
                        loss /= self.accumulate_steps
                    self.scaler.scale(loss).backward()
                    if (step + 1) % self.accumulate_steps == 0:  # Wait for several backward steps
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                else:
                    train_output, sorted_labels, sorted_decode_lengths = self.model(images,
                                                                                    labels=labels,
                                                                                    decode_lengths=decode_lengths)
                    loss = seq_cross_entropy_loss(train_output, sorted_labels, sorted_decode_lengths)
                    loss /= self.accumulate_steps
                    loss.backward()
                    if (step + 1) % self.accumulate_steps == 0:  # Wait for several backward steps
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                summary_loss.update(loss.detach().item() * self.accumulate_steps)
                if (step + 1) % 10000 == 0:
                    self.log(
                        f'[RESULT]: Train. Step: {step}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
                    self.save(f'{self.base_dir}/last-checkpoint.bin')
                if (step + 1) % 10000 == 0:
                    t = time.time()
                    predictions = self.validation(self.model, validation_loader)
                    if (step + 1) % 100000 == 0:
                        self.log(f"ground truth: {self.valid_labels[:5]}")
                        self.log(f"predictions: {predictions[:5]}")
                    score = get_score(self.valid_labels[:len(predictions)], predictions)
                    self.log(
                        f'[RESULT]: Val. Step: {step}, score: {score:.5f}, time: {(time.time() - t):.5f}')
                    if score < self.best_score:
                        self.best_score = score
                        self.model.eval()
                        self.save(f'{self.base_dir}/best-checkpoint-{step}.bin')
                        # for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*.bin'))[:-3]:
                        #     os.remove(path)
                    if self.config.validation_scheduler:
                        self.scheduler.step()
                        lr = self.optimizer.param_groups[0]['lr']
                        self.log(f'Step: {step}, LR: {lr}')

                if self.config.step_scheduler:
                    self.scheduler.step()

                step += 1

    def validation(self, model, val_loader):
        model.eval()
        t = time.time()
        all_text_predictions = []
        for step, (images) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                images = images.to(self.device).float()
                predictions = model.predict(images)
            predictions = predictions.detach().cpu().numpy()
            text_predictions = self.tokenizer.predict_captions(predictions)
            all_text_predictions.append(text_predictions)
        all_text_predictions = np.concatenate(all_text_predictions)
        all_text_predictions = [f"InChI=1S/{text}" for text in all_text_predictions]

        return all_text_predictions

    def save(self, path):
        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    }, path)

    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')

def run_training(encoder_net, decoder_net, train_dataset,
                 validation_dataset, tokenizer, collate_fn, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=False
    )

    fitter = Fitter(encoder_model=encoder_net,
                    decoder_model=decoder_net,
                    tokenizer=tokenizer,
                    device=device,
                    config=config)
    fitter.fit(train_loader, val_loader)




def run_training_v2(model, train_dataset, validation_dataset, tokenizer, collate_fn, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        sampler=FixNumSampler(validation_dataset, 20000),
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=False
    )

    fitter = FitterV3(model=model,
                      tokenizer=tokenizer,
                      device=device,
                      config=config)
    fitter.fit(train_loader, val_loader)
