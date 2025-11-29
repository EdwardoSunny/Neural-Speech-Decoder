import math
import os
import pickle
import time

from edit_distance import SequenceMatcher
import hydra
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import wandb

from .model import GRUDecoder
from .transformer_ctc import NeuralTransformerCTCModel
from .dataset import SpeechDataset


def getDatasetLoaders(
    datasetName,
    batchSize,
):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)

        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )

    train_ds = SpeechDataset(loadedData["train"], transform=None)
    test_ds = SpeechDataset(loadedData["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData


def _compute_output_lengths(model, x_lens):
    if hasattr(model, "compute_output_lengths"):
        return model.compute_output_lengths(x_lens)
    else:
        return ((x_lens - model.kernelLen) / model.strideLen).to(torch.int32)


def trainModel(args):
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = "cuda"

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    # Initialize wandb
    wandb.init(
        project="neural-seq-decoder",
        config=args,
        name=os.path.basename(args["outputDir"]),
    )

    trainLoader, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
    )

    if args.get("model_type", "gru_baseline") == "transformer_ctc":
        model = NeuralTransformerCTCModel(
            n_channels=args["nInputFeatures"],
            n_classes=args["nClasses"] + 1,  # CTC blank already added in GRU; align dimensions
            n_days=len(loadedData["train"]),
            use_wavelets=args.get("use_wavelets", True),
            n_scales=args.get("wavelet_n_scales", 4),
            wavelet_window_bins=args.get("wavelet_window_bins", 5),
            wavelet_stride_bins=args.get("wavelet_stride_bins", 1),
            use_pac_features=args.get("use_pac_features", False),
            frontend_dim=args.get("frontend_dim", 512),
            latent_dim=args.get("latent_dim", 256),
            transformer_layers=args.get("transformer_num_layers", 4),
            transformer_heads=args.get("transformer_n_heads", 4),
            transformer_ff_dim=args.get("transformer_dim_ff", 1024),
            transformer_dropout=args.get("transformer_dropout", 0.1),
            device=device,
        ).to(device)
    else:
        model = GRUDecoder(
            neural_dim=args["nInputFeatures"],
            n_classes=args["nClasses"],
            hidden_dim=args["nUnits"],
            layer_dim=args["nLayers"],
            nDays=len(loadedData["train"]),
            dropout=args["dropout"],
            device=device,
            strideLen=args["strideLen"],
            kernelLen=args["kernelLen"],
            gaussianSmoothWidth=args["gaussianSmoothWidth"],
            bidirectional=args["bidirectional"],
        ).to(device)

    # Watch model with wandb
    wandb.watch(model, log="all", log_freq=100)

    blank_idx = 0
    n_classes = args["nClasses"] + 1  # +1 blank
    loss_ctc = torch.nn.CTCLoss(blank=blank_idx, reduction="mean", zero_infinity=True)

    if args.get("optimizer", "adam") == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args["lrStart"],
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.get("weight_decay", args.get("l2_decay", 0)),
        )
        warmup_steps = int(args.get("warmup_steps", 0))
        total_steps = args["nBatch"]

        def lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args["lrStart"],
            betas=(0.9, 0.999),
            eps=0.1,
            weight_decay=args["l2_decay"],
        )
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=args["lrEnd"] / args["lrStart"],
            total_iters=args["nBatch"],
        )

    # --train--
    testLoss = []
    testCER = []
    startTime = time.time()
    for batch in range(args["nBatch"]):
        model.train()

        X, y, X_len, y_len, dayIdx = next(iter(trainLoader))
        X, y, X_len, y_len, dayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            dayIdx.to(device),
        )

        # Noise augmentation is faster on GPU
        if args["whiteNoiseSD"] > 0:
            X += torch.randn(X.shape, device=device) * args["whiteNoiseSD"]

        if args["constantOffsetSD"] > 0:
            X += (
                torch.randn([X.shape[0], 1, X.shape[2]], device=device)
                * args["constantOffsetSD"]
            )

        # Compute prediction error
        if args.get("model_type", "gru_baseline") == "transformer_ctc":
            log_probs, out_lens = model(X, dayIdx, X_len)
        else:
            pred = model.forward(X, dayIdx)
            out_lens = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)
            log_probs = pred.log_softmax(2).permute(1, 0, 2)

        loss = loss_ctc(
            log_probs,
            y,
            out_lens,
            y_len,
        )
        loss = torch.sum(loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Log training metrics
        wandb.log({
            "train/loss": loss.item(),
            "train/learning_rate": optimizer.param_groups[0]['lr'],
            "train/batch": batch,
        }, step=batch)

        # Eval
        if batch % 100 == 0:
            with torch.no_grad():
                model.eval()
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0
                for X, y, X_len, y_len, testDayIdx in testLoader:
                    X, y, X_len, y_len, testDayIdx = (
                        X.to(device),
                        y.to(device),
                        X_len.to(device),
                        y_len.to(device),
                        testDayIdx.to(device),
                    )

                    if args.get("model_type", "gru_baseline") == "transformer_ctc":
                        pred, adjustedLens = model(X, testDayIdx, X_len)
                        # pred already [T, B, C]
                    else:
                        logits = model.forward(X, testDayIdx)
                        adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(
                            torch.int32
                        )
                        pred = logits.log_softmax(2).permute(1, 0, 2)

                    loss = loss_ctc(
                        pred,
                        y,
                        adjustedLens,
                        y_len,
                    )
                    loss = torch.sum(loss)
                    allLoss.append(loss.cpu().detach().numpy())

                    for iterIdx in range(pred.shape[1]):  # pred is [T, B, C]
                        decodedSeq = torch.argmax(
                            torch.tensor(pred[0 : adjustedLens[iterIdx], iterIdx, :]),
                            dim=-1,
                        )  # [num_seq,]
                        decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                        decodedSeq = decodedSeq.cpu().detach().numpy()
                        decodedSeq = np.array([i for i in decodedSeq if i != 0])

                        trueSeq = np.array(
                            y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
                        )

                        matcher = SequenceMatcher(
                            a=trueSeq.tolist(), b=decodedSeq.tolist()
                        )
                        total_edit_distance += matcher.distance()
                        total_seq_length += len(trueSeq)

                avgDayLoss = np.sum(allLoss) / len(testLoader)
                cer = total_edit_distance / total_seq_length

                endTime = time.time()
                time_per_batch = (endTime - startTime) / 100

                print(
                    f"batch {batch}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}, time/batch: {time_per_batch:>7.3f}"
                )

                # Log evaluation metrics
                wandb.log({
                    "eval/loss": avgDayLoss,
                    "eval/cer": cer,
                    "eval/time_per_batch": time_per_batch,
                }, step=batch)

                startTime = time.time()

            if len(testCER) > 0 and cer < np.min(testCER):
                torch.save(model.state_dict(), args["outputDir"] + "/modelWeights")
                wandb.log({"eval/best_cer": cer}, step=batch)
            testLoss.append(avgDayLoss)
            testCER.append(cer)

            tStats = {}
            tStats["testLoss"] = np.array(testLoss)
            tStats["testCER"] = np.array(testCER)

            with open(args["outputDir"] + "/trainingStats", "wb") as file:
                pickle.dump(tStats, file)

    # Finish wandb run
    wandb.finish()


def loadModel(modelDir, nInputLayers=24, device="cuda"):
    modelWeightPath = modelDir + "/modelWeights"
    with open(modelDir + "/args", "rb") as handle:
        args = pickle.load(handle)

    if args.get("model_type", "gru_baseline") == "transformer_ctc":
        model = NeuralTransformerCTCModel(
            n_channels=args["nInputFeatures"],
            n_classes=args["nClasses"] + 1,
            n_days=nInputLayers,
            use_wavelets=args.get("use_wavelets", True),
            n_scales=args.get("wavelet_n_scales", 4),
            wavelet_window_bins=args.get("wavelet_window_bins", 5),
            wavelet_stride_bins=args.get("wavelet_stride_bins", 1),
            use_pac_features=args.get("use_pac_features", False),
            frontend_dim=args.get("frontend_dim", 512),
            latent_dim=args.get("latent_dim", 256),
            transformer_layers=args.get("transformer_num_layers", 4),
            transformer_heads=args.get("transformer_n_heads", 4),
            transformer_ff_dim=args.get("transformer_dim_ff", 1024),
            transformer_dropout=args.get("transformer_dropout", 0.1),
            device=device,
        ).to(device)
    else:
        model = GRUDecoder(
            neural_dim=args["nInputFeatures"],
            n_classes=args["nClasses"],
            hidden_dim=args["nUnits"],
            layer_dim=args["nLayers"],
            nDays=nInputLayers,
            dropout=args["dropout"],
            device=device,
            strideLen=args["strideLen"],
            kernelLen=args["kernelLen"],
            gaussianSmoothWidth=args["gaussianSmoothWidth"],
            bidirectional=args["bidirectional"],
        ).to(device)

    model.load_state_dict(torch.load(modelWeightPath, map_location=device))
    return model


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg):
    cfg.outputDir = os.getcwd()
    trainModel(cfg)

if __name__ == "__main__":
    main()
