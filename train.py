import torch
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from tqdm.notebook import tqdm
from itertools import repeat
from wandb_logger import WanDBWriter
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn



def save_checkpoint(model, optimizer, epoch, scheduler=None, checkpoint_name="checkpoint"):
    state = {
        "arch": type(model).__name__,
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else ""
    }

    filename = f"{checkpoint_name}.pth"
    torch.save(state, filename)
        

def evaluate(model, dataloader, device, metric=None):
    model.eval()
    losses = 0
    metrics = 0

    for inp in tqdm(dataloader):
        inp['input_ids'] = inp['input_ids'].to(device)
        inp['attention_mask'] = inp['attention_mask'].to(device)
        inp['labels'] = inp['labels'].to(device)
        
        out = model(**inp)

        losses += out.loss.item()
        if metric is not None:
            metrics += metric(out.logits.argmax(2).tolist(), inp['labels'].tolist())

    return losses / len(dataloader), metrics / len(dataloader)


def train(n_epochs, model, optimizer, train_loader, val_loader, device, scheduler=None, log_step=500, metric=None, run_name=None, clip=100, checkpoint_name="checkpoint"):
    writer = WanDBWriter(run_name)
    len_epoch = len(train_loader)
    best_loss = 1e6
    for epoch in range(1, n_epochs + 1):
        losses = 0
        metrics = 0

        for i, inp in enumerate(tqdm(train_loader, desc="train")):
            model.train()
            
            optimizer.zero_grad()
            
            inp['input_ids'] = inp['input_ids'].to(device)
            inp['attention_mask'] = inp['attention_mask'].to(device)
            inp['labels'] = inp['labels'].to(device)
            
            out = model(**inp)

            out.loss.backward()
            
            clip_grad_norm_(model.parameters(), clip)

            optimizer.step()
            losses += out.loss.item()
            
            if metric is not None:
                metrics += metric(out.logits.argmax(2).tolist(), inp['labels'].tolist())

            if (i + 1) % log_step == 0:
                writer.set_step((epoch - 1) * len_epoch + i + 1)
                if scheduler is not None:
                    writer.add_scalar("lr", scheduler.get_last_lr()[0])
                writer.add_scalar("train_loss", losses / log_step)
                writer.add_scalar("train_metric", metrics / log_step)
                print(f"Epoch: {epoch}, Train loss: {(losses / log_step):.3f}, Train metric: {(metrics / log_step):.3f}")
                losses = 0
                metrics = 0

            if scheduler is not None:
                scheduler.step()
                
        writer.set_step((epoch - 1) * len_epoch + i + 1)
        val_loss, val_metric = evaluate(model, val_loader, device, metric)
        if val_loss < best_loss:
            print("Saving checkpoint...")
            save_checkpoint(model, optimizer, epoch, scheduler, checkpoint_name)
            best_loss = val_loss

        a = (i + 1) % log_step
        if a == 0:
            a = log_step

        writer.add_scalar("train_loss", losses / a)
        writer.add_scalar("val_loss", val_loss)
        writer.add_scalar("val_metric", val_metric)
        if scheduler is not None:
            writer.add_scalar("lr", scheduler.get_last_lr()[0])
        print((f"Epoch: {epoch}, Train loss: {(losses / a):.3f}, Val loss: {val_loss:.3f}, Train metric: {(metrics / a):.3f}, Val metric: {val_metric:.3f}"))


def train_distillation(n_epochs, 
                       model_teacher,
                       model_student,
                       optimizer, 
                       train_loader, 
                       val_loader, 
                       device, 
                       scheduler=None, 
                       log_step=500, 
                       metric=None, 
                       run_name=None, 
                       clip=100, 
                       checkpoint_name="checkpoint",
                       t=2):
    writer = WanDBWriter(run_name)
    
    distillation_loss_fn = nn.KLDivLoss(reduction="batchmean")
    model_teacher.eval()
    
    len_epoch = len(train_loader)
    best_loss = 1e6
    for epoch in range(1, n_epochs + 1):
        losses = 0
        losses_d = 0
        metrics = 0

        for i, inp in enumerate(tqdm(train_loader, desc="train")):
            model_student.train()
            
            optimizer.zero_grad()
            
            inp['input_ids'] = inp['input_ids'].to(device)
            inp['attention_mask'] = inp['attention_mask'].to(device)
            inp['labels'] = inp['labels'].to(device)
            
            out_teacher = model_teacher(**inp)
            p_teacher = torch.nn.functional.softmax(out_teacher.logits / t, dim=-1)
            
            out = model_student(**inp)
            p_student = torch.nn.functional.log_softmax(out.logits / t, dim=-1)
            
            loss_d = distillation_loss_fn(p_student, p_teacher)
            
            global_loss = 0.5 * loss_d + out.loss
            global_loss.backward()
            
            clip_grad_norm_(model_student.parameters(), clip)
            optimizer.step()
            
            losses += out.loss.item()
            losses_d += loss_d.item()
            
            if metric is not None:
                metrics += metric(out.logits.argmax(2).tolist(), inp['labels'].tolist())

            if (i + 1) % log_step == 0:
                writer.set_step((epoch - 1) * len_epoch + i + 1)
                if scheduler is not None:
                    writer.add_scalar("lr", scheduler.get_last_lr()[0])
                writer.add_scalar("train_loss", losses / log_step)
                writer.add_scalar("train_loss_d", losses_d / log_step)
                writer.add_scalar("train_metric", metrics / log_step)
                print(f"Epoch: {epoch}, Train loss: {(losses / log_step):.3f}, Train loss d: {(losses_d / log_step):.3f}, Train metric: {(metrics / log_step):.3f}")
                losses = 0
                losses_d = 0
                metrics = 0

            if scheduler is not None:
                scheduler.step()
                
        writer.set_step((epoch - 1) * len_epoch + i + 1)
        val_loss, val_metric = evaluate(model_student, val_loader, device, metric)
        if val_loss < best_loss:
            print("Saving checkpoint...")
            save_checkpoint(model_student, optimizer, epoch, scheduler, checkpoint_name)
            best_loss = val_loss

        a = (i + 1) % log_step
        if a == 0:
            a = log_step

        writer.add_scalar("train_loss", losses / a)
        writer.add_scalar("train_loss_d", losses_d / a)
        writer.add_scalar("val_loss", val_loss)
        writer.add_scalar("val_metric", val_metric)
        if scheduler is not None:
            writer.add_scalar("lr", scheduler.get_last_lr()[0])
        print((f"Epoch: {epoch}, Train loss: {(losses / a):.3f}, Train loss d: {(losses_d / a):.3f}, Val loss: {val_loss:.3f}, Train metric: {(metrics / a):.3f}, Val metric: {val_metric:.3f}"))