import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# import my libs
from dataset.dataset_train import TrainDataset
from model.U_Net import uNet
from config import cfg


def train(cfg):
    # define device
    device = cfg.TRAIN.DEVICE

    # 1.load data
    data = TrainDataset(cfg.DATASET.TRAIN_IMAGE_FOLDER, cfg.DATASET.TRAIN_LABEL_FOLDER, cfg.DATASET.BAND_LIST,
                        cfg.DATASET.WINDOE_SIZE, cfg.DATASET.AUGMENTATION)
    rs_dataloader = DataLoader(data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=cfg.TRAIN.SHUFFLE)

    # 2.define net model
    my_net = uNet(cfg.TRAIN.INPUT_CHANNEL, cfg.DATASET.CLASS_NUM).to(device)

    # 3.define loss function
    loss_function = nn.CrossEntropyLoss()

    # 4.define optimizer
    optimizer = torch.optim.SGD(my_net.parameters(), lr=cfg.TRAIN.LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM,
                                weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # 5.start training
    # define loss output path
    loss_txt = open(cfg.TRAIN.LOSS_SAVE_PATH)
    for e in range(1, cfg.TRAIN.EPOCH + 1):  # 1 to epoch
        for idx, images, labels in enumerate(rs_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = my_net(images)

            optimizer.zero_grad()
            loss = loss_function(outputs, labels.long())
            loss.backward()
            optimizer.step()

            if (idx + 1) % cfg.TRAIN.LOSS_OUT_IDX == 0:  # save loss every idx batch
                print('epoch is %d , %d times , loss is %f' % (e, idx + 1, loss.item()))
                loss_txt.write(
                    "epoch is " + str(e) + ", " + str(idx + 1) + " batch, loss is " + str(loss.item()) + "\n")

        if e == 1 or e % cfg.TRAIN.MODEL_SAVE_ITER == 0:  # save model every iter epoch
            torch.save(my_net, cfg.TRAIN.MODEL_SAVE_PATH + "//model" + str(e) + ".pkl")

    return


if __name__ == '__main__':
    cfg.merge_from_file("config//config.yaml")
    train(cfg)
    print("Congratulations, the training is done!\n")
