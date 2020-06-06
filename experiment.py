import os
from datetime import datetime, timedelta
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np

from dataset_factory import get_datasets
from file_utils import *
import matplotlib.pyplot as plt
from constants import ROOT_STATS_DIR
from model_factory import get_model


class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./config/', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        # Load Datasets
        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        ds_train, ds_val = get_datasets(config_data)
        self.__train_loader = DataLoader(ds_train, batch_size=config_data['experiment']['batch_size_train'],
                                         shuffle=True,
                                         num_workers=config_data['experiment']['num_workers'], pin_memory=True)
        self.__val_loader = DataLoader(ds_val, batch_size=config_data['experiment']['batch_size_val'], shuffle=True,
                                       num_workers=config_data['experiment']['num_workers'], pin_memory=True)
        # Setup Experiment Stats
        self.__epochs = config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []

        # Init Model
        self.__model = get_model(config_data)

        # These can be made configurable or changed, if required.
        self.__criterion = torch.nn.BCEWithLogitsLoss()
        self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr=config_data['experiment']['learning_rate'])

        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()

    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])

        else:
            os.makedirs(self.__experiment_dir)
            os.makedirs(os.path.join(self.__experiment_dir, 'models'))

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()
        # self.model = torch.nn.DataParallel(self.model)

    def run(self):
        start_epoch = self.__current_epoch
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss = self.__train()
            val_loss = self.__val()
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()

    def __train(self):
        self.__model.__train()
        train_loss_epoch = []
        for i, data in enumerate(self.__train_loader):
            inputs = data[0].cuda().float() if torch.cuda.is_available() else data[0].double()
            labels = data[1].cuda().float() if torch.cuda.is_available() else data[1].double()

            self.__optimizer.zero_grad()
            outputs = self.__model.forward(inputs).squeeze()
            loss = self.__criterion(outputs, labels)
            loss.backward()
            self.__optimizer.step()
            train_loss_epoch.append(loss.item())

            status_str = "Epoch: {}, Train, Batch {}/{}. Loss {}".format(self.__current_epoch + 1, i + 1,
                                                                         len(self.__train_loader),
                                                                         loss.item())
            self.__log(status_str)

        return np.mean(train_loss_epoch)

    def __val(self):
        self.__model.eval()
        val_loss_epoch = []
        for i, data in enumerate(self.__val_loader):
            inputs = data[0].cuda().float() if torch.cuda.is_available() else data[0].double()
            labels = data[1].cuda().float() if torch.cuda.is_available() else data[1].double()

            with torch.no_grad():
                outputs = self.__model.forward(inputs).squeeze()
                loss = self.__criterion(outputs, labels)
            val_loss_epoch.append(loss.item())

            status_str = "Epoch: {}, Val, Batch {}/{}. Loss {}".format(self.__current_epoch + 1, i + 1,
                                                                       len(self.__val_loader),
                                                                       loss.item())
            self.__log(status_str)

        return np.mean(val_loss_epoch)

    def __save_model(self):
        epoch_model_path = os.path.join(self.__experiment_dir, 'models', 'model_{}.pt'.format(self.__current_epoch))
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')

        if isinstance(self.__model, torch.nn.DataParallel):
            model_dict = self.__model.module.state_dict()
        else:
            model_dict = self.__model.state_dict()

        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(self.__model.state_dict(), epoch_model_path)
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss, val_dice):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()
