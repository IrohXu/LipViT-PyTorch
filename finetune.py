import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter

from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Average, Loss
from ignite.contrib.handlers import ProgressBar
from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup

from timm.data import create_loader, FastCollateMixup

from datasets import get_dataset
from lib.ood_evaluator import OODEvaluator
from lib.calibration_error import ECE
from lib.utils import get_results_directory, Hyperparameters, set_seed

from models import vit_backbone, tgp
import torch_optimizer as optim


# torch.backends.cudnn.benchmark = True


def main(hparams):
    results_dir = get_results_directory(hparams.output_dir, stamp=not hparams.no_stamp)
    writer = SummaryWriter(log_dir=str(results_dir))
    ds = get_dataset(hparams.dataset, root=hparams.data_root)
    train_dataset, test_dataset, data_config = ds
    hparams.seed = set_seed(hparams.seed)
    print(f"Fine-tuning with {hparams}")
    hparams.save(results_dir / "hparams.json")

    if not hparams.gaussian_process:
        if hparams.load_model:
            model = vit_backbone(hparams.pretrain_model_num_classes,
                                 hparams.vit_variant,
                                 attention=hparams.attention_module,
                                 alpha=hparams.alpha)
            model.load_state_dict(torch.load(hparams.load_model))
            print("Pretrained model loaded from {:s}".format(hparams.load_model))
            model.reset_classifier(data_config['num_classes'])
        else:
            model = vit_backbone(data_config['num_classes'],
                                 hparams.vit_variant,
                                 attention=hparams.attention_module,
                                 alpha=hparams.alpha)
    else:
        if hparams.load_model:
            vit = vit_backbone(hparams.pretrain_model_num_classes,
                               hparams.vit_variant,
                               attention=hparams.attention_module,
                               alpha=hparams.alpha)
            vit.load_state_dict(torch.load(hparams.load_model))
            print("Pretrained model loaded from {:s}".format(hparams.load_model))
            vit.reset_classifier(0)
            model = tgp(
                num_classes=data_config['num_classes'],
                num_data=len(train_dataset),
                batch_size=hparams.batch_size,
                variant=hparams.vit_variant,
                vit=vit
            )
        else:
            model = tgp(num_classes=data_config['num_classes'],
                        num_data=len(train_dataset),
                        batch_size=hparams.batch_size,
                        variant=hparams.vit_variant,
                        attention=hparams.attention_module,
                        alpha=hparams.alpha)

    param_frozen_list = nn.ParameterList()
    param_active_list = nn.ParameterList()

    hparams.active_modules = hparams.active_modules.split(',')
    # print("Finetune parameters:")
    print("Frozen parameters:")
    for name, param in model.named_parameters():
        if any([m in name for m in hparams.active_modules]):
            param.requires_grad = True
            param_active_list.append(param)
            # print(name)
        else:
            param.requires_grad = False
            param_frozen_list.append(param)
            print(name)

    model = model.cuda()
    optimizer = optim.Lamb(
        param_active_list,
        lr=hparams.learning_rate,
        weight_decay=hparams.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    torch_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=hparams.num_epochs,
    )
    scheduler = create_lr_scheduler_with_warmup(torch_lr_scheduler,
                                                warmup_start_value=hparams.learning_rate,
                                                warmup_duration=hparams.warmup_epochs)
    loss_fn = F.binary_cross_entropy_with_logits

    ood_datasets = ood_evaluators = None
    if hparams.evaluate_ood:
        ood_datasets = hparams.ood_datasets.split(',')
        ood_evaluators = [OODEvaluator(hparams.dataset, ood_dataset, model, hparams.data_root, hparams.num_workers,
                                       hparams.batch_size) for ood_dataset in ood_datasets]

    def step(engine, batch):
        model.train()
        optimizer.zero_grad()

        x, y = batch
        x, y = x.cuda(), y.cuda()

        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        loss.backward()
        optimizer.step()

        return loss.item()

    def eval_step(engine, batch):
        model.eval()

        x, y = batch
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            y_pred = model(x)

        return y_pred, y

    trainer = Engine(step)
    evaluator = Engine(eval_step)

    metric = Average()
    metric.attach(trainer, "loss")

    metric = Accuracy()
    metric.attach(evaluator, "accuracy")

    metric = Loss(F.cross_entropy)
    metric.attach(evaluator, "loss")

    if hparams.evaluate_ece:
        metric = ECE()
        metric.attach(evaluator, "ece")

    mixup_args = dict(
        mixup_alpha=0.1,
        cutmix_alpha=1.0,
        prob=1.0,  # Probability of performing mixup or cutmix when either/both is enabled
        switch_prob=0.5,  # Probability of switching to cutmix when both mixup and cutmix enabled
        mode='batch',  # 'How to apply mixup/cutmix params. Per "batch", "pair", or "elem"'
        label_smoothing=hparams.label_smoothing,
        num_classes=data_config['num_classes'],
    )

    collate_fn = FastCollateMixup(**mixup_args)
    # mixup_fn = Mixup(**mixup_args)

    train_args = dict(
        batch_size=hparams.batch_size,
        is_training=True,  # H.flip & RRC
        use_prefetcher=True,
        no_aug=False,
        auto_augment='rand-m6-mstd0.5',  # Rand Augment 7/0.5
        num_workers=hparams.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        mean=data_config['mean'],
        std=data_config['std'],
    )
    test_args = dict(
        batch_size=hparams.batch_size,
        is_training=False,
        use_prefetcher=True,
        num_workers=hparams.num_workers,
        pin_memory=True,
        mean=data_config['mean'],
        std=data_config['std'],
        crop_pct=0.95
    )

    train_loader = create_loader(
        train_dataset,
        data_config['input_size'],
        **train_args
    )

    test_loader = create_loader(
        test_dataset,
        data_config['input_size'],
        **test_args
    )

    @trainer.on(Events.EPOCH_STARTED)
    def reset_precision_matrix(trainer):
        if hparams.gaussian_process:
            model.reset_precision_matrix()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(trainer):
        metrics = trainer.state.metrics
        train_loss = metrics["loss"]
        result = f"Train - Epoch: {trainer.state.epoch} "
        result += f"Loss: {train_loss:.4f} "
        print(result)
        writer.add_scalar("Loss/train", train_loss, trainer.state.epoch)

        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        test_acc = metrics["accuracy"]
        test_loss = metrics["loss"]
        result = f"Test - Epoch: {trainer.state.epoch} "
        result += f"Loss: {test_loss:.4f} "
        result += f"Accuracy: {test_acc:.4f} "
        writer.add_scalar("Loss/test", test_loss, trainer.state.epoch)
        writer.add_scalar("Accuracy/test", test_acc, trainer.state.epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], trainer.state.epoch)
        if hparams.evaluate_ece:
            test_ece = metrics["ece"]
            result += f"ECE: {test_ece:.4f} "
            writer.add_scalar("ECE/test", test_ece, trainer.state.epoch)
        print(result)

        if hparams.evaluate_ood:
            if trainer.state.epoch >= int(0.75 * hparams.num_epochs) and trainer.state.epoch % 2 == 0:
                for ood_evaluator in ood_evaluators:
                    auroc, aupr = ood_evaluator.get_ood_metrics()
                    print(
                        f"OoD {ood_evaluator.in_ds_name} vs {ood_evaluator.out_ds_name} - AUROC: {auroc}, AUPR: {aupr}")
                    writer.add_scalar(f"OoD/{ood_evaluator.out_ds_name}/auroc", auroc, trainer.state.epoch)
                    writer.add_scalar(f"OoD/{ood_evaluator.out_ds_name}/aupr", aupr, trainer.state.epoch)

        torch.save(
            {'epoch': trainer.state.epoch,
             'model': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict()},
            results_dir / "train.pt"
        )

    pbar = ProgressBar(dynamic_ncols=True)
    pbar.attach(trainer)

    trainer.add_event_handler(Events.EPOCH_STARTED, scheduler)
    trainer.run(train_loader, max_epochs=hparams.num_epochs)

    # Done training - time to evaluate
    results = {}
    evaluator.run(test_loader)
    test_acc = evaluator.state.metrics["accuracy"]
    test_loss = evaluator.state.metrics["loss"]
    results["test_accuracy"] = test_acc
    results["test_loss"] = test_loss
    if hparams.evaluate_ece:
        test_ece = evaluator.state.metrics["ece"]
        results["test_ece"] = test_ece

    if hparams.evaluate_ood:
        for ood_evaluator in ood_evaluators:
            auroc, aupr = ood_evaluator.get_ood_metrics()
            results[f"ood_{ood_evaluator.out_ds_name}_auroc"] = auroc
            results[f"ood_{ood_evaluator.out_ds_name}_aupr"] = aupr

    print(f"Final accuracy {results['test_accuracy']:.4f}")

    results_json = json.dumps(results, indent=4, sort_keys=True)
    (results_dir / "results.json").write_text(results_json)

    torch.save(model.state_dict(), results_dir / "final_model.pt")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=512, help="Batch size to use for training")
    parser.add_argument("--learning_rate", type=float, default=6e-3)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="./default/runs/finetune", type=str)
    parser.add_argument("--data_root", default="./default/datasets", type=str)
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument("--vit_variant", default='tiny', choices=['tiny', 'small', 'base', 'large'])
    parser.add_argument("--label_smoothing", default=0.0, type=float, help="0.0 means no smoothing")
    parser.add_argument("--active_modules", default="attn,head", help="'' empty means all modules are active")
    parser.add_argument("--no_stamp", action="store_true", help="not to add timestamp to the result dir")
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--attention_module", default="COBILIR", type=str, choices=['DPSA', 'L2SA', 'SNSA', 'COBILIR'])
    parser.add_argument("--warmup_epochs", default=5, type=int)
    parser.add_argument("--gaussian_process", action="store_true", help="Whether to use GP output layer")
    parser.add_argument("--dataset", default="CIFAR10", choices=["CIFAR10", "CIFAR100", "ImageNet1K"])
    parser.add_argument("--evaluate_ood", action="store_true")
    parser.add_argument("--ood_datasets", default="CIFAR100,SVHN")
    parser.add_argument("--load_model", default="", type=str)
    parser.add_argument("--pretrain_model_num_classes", default=1000, type=int, help="num classes when pretrain model")
    parser.add_argument("--alpha", default=100, type=int, help='Hyper-parameter in CoBiLiR module')
    parser.add_argument("--evaluate_ece", action="store_true")

    args = parser.parse_args()
    hparams = Hyperparameters(**vars(args))

    main(hparams)
