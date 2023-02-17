import argparse
import json

import torch
import torch.nn.functional as F

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine
from ignite.metrics import Accuracy, Loss

from timm.data import create_loader

from datasets import get_dataset
from archive.evaluate_ood import get_ood_metrics
from lib.ood_evaluator import OODEvaluator
from lib.calibration_error import ECE
from lib.utils import get_results_directory, Hyperparameters, set_seed

from models import vit_backbone


# torch.backends.cudnn.benchmark = True


def main(hparams):
    results_dir = get_results_directory(hparams.output_dir, stamp=not hparams.no_stamp)
    ds = get_dataset(hparams.dataset, root=hparams.data_root)
    _, test_dataset, data_config = ds
    hparams.seed = set_seed(hparams.seed)
    print(f"Evaluate with {hparams}")
    hparams.save(results_dir / "hparams.json")
    model = vit_backbone(data_config['num_classes'],
                         hparams.vit_variant,
                         attention=hparams.attention_module,
                         alpha=hparams.alpha)
    model.load_state_dict(torch.load(hparams.load_model))
    print("Pretrained model loaded from {:s}".format(hparams.load_model))
    model = model.cuda()

    ood_datasets = ood_evaluators = None
    if hparams.evaluate_ood:
        ood_datasets = hparams.ood_datasets.split(',')
        ood_evaluators = [OODEvaluator(hparams.dataset, ood_dataset, model, hparams.data_root, hparams.num_workers,
                                       hparams.batch_size) for ood_dataset in ood_datasets]

    def eval_step(engine, batch):
        model.eval()

        x, y = batch
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            y_pred = model(x)

        return y_pred, y

    evaluator = Engine(eval_step)

    metric = Accuracy()
    metric.attach(evaluator, "accuracy")
    metric = Loss(F.cross_entropy)
    metric.attach(evaluator, "loss")
    if hparams.evaluate_ece:
        metric = ECE()
        metric.attach(evaluator, "ece")

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

    test_loader = create_loader(
        test_dataset,
        data_config['input_size'],
        **test_args
    )

    results = {}

    pbar = ProgressBar(dynamic_ncols=True)
    pbar.attach(evaluator)

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

    print(f"Accuracy {results['test_accuracy']:.4f}")

    results_json = json.dumps(results, indent=4, sort_keys=True)
    (results_dir / "results.json").write_text(results_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size to use for training")
    parser.add_argument("--dataset", default="CIFAR10", choices=["CIFAR10", "CIFAR100", "ImageNet1K"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="./default/runs/evaluate", type=str)
    parser.add_argument("--data_root", default="./default/datasets", type=str)
    parser.add_argument("--vit_variant", default='tiny', choices=['tiny', 'small', 'base', 'large'])
    parser.add_argument("--evaluate_ood", action="store_true")
    parser.add_argument("--ood_datasets", default="CIFAR100,SVHN")
    parser.add_argument("--load_model", default="", type=str)
    parser.add_argument("--no_stamp", action="store_true", help="not to add timestamp to the result dir")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--attention_module", default="DPSA", type=str, choices=['DPSA', 'L2SA', 'SNSA', 'COBILIR'])
    parser.add_argument("--alpha", default=100, type=int, help='Hyper-parameter in CoBiLiR module')
    parser.add_argument("--evaluate_ece", action="store_true")

    args = parser.parse_args()
    hparams = Hyperparameters(**vars(args))

    main(hparams)
