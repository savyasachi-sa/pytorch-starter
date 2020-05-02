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
        self.name = config_data['experiment_name']
        self.experiment_dir = os.path.join(ROOT_STATS_DIR, self.name)

        ds_train, ds_val = get_datasets(config_data)
        self.train_loader = DataLoader(ds_train, batch_size=config_data['experiment']['batch_size_train'], shuffle=True,
                                       num_workers=config_data['experiment']['num_workers'], pin_memory=True)
        self.val_loader = DataLoader(ds_val, batch_size=config_data['experiment']['batch_size_val'], shuffle=True,
                                     num_workers=config_data['experiment']['num_workers'], pin_memory=True)
        # Setup Experiment Stats
        self.epochs = config_data['experiment']['num_epochs']
        self.current_epoch = 0
        self.training_losses = []
        self.val_losses = []

        # Init Model
        self.model = get_model(config_data)

        # These can be made configurable or changed, if required.
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config_data['experiment']['learning_rate'])

        self.init_model()

        # Load Experiment Data if available
        self.load_experiment()

    def load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.experiment_dir):
            self.training_losses = read_file_in_dir(self.experiment_dir, 'training_losses.txt')
            self.val_losses = read_file_in_dir(self.experiment_dir, 'val_losses.txt')
            self.current_epoch = len(self.training_losses)

            state_dict = torch.load(os.path.join(self.experiment_dir, 'latest_model.pt'))
            self.model.load_state_dict(state_dict['model'])
            self.optimizer.load_state_dict(state_dict['optimizer'])

        else:
            os.makedirs(self.experiment_dir)
            os.makedirs(os.path.join(self.experiment_dir, 'models'))

    def init_model(self):
        if torch.cuda.is_available():
            self.model = self.model.cuda().float()
            self.criterion = self.criterion.cuda()
        # self.model = torch.nn.DataParallel(self.model)

    def run(self):
        start_epoch = self.current_epoch
        for epoch in range(start_epoch, self.epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.current_epoch = epoch
            train_loss = self.train()
            val_loss = self.val()
            self.record_stats(train_loss, val_loss)
            self.log_epoch_stats(start_time)
            self.save_model()

    def train(self):
        self.model.train()
        train_loss_epoch = []
        for i, data in enumerate(self.train_loader):
            inputs = data[0].cuda().float() if torch.cuda.is_available() else data[0].double()
            labels = data[1].cuda().float() if torch.cuda.is_available() else data[1].double()

            self.optimizer.zero_grad()
            outputs = self.model.forward(inputs).squeeze()
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            train_loss_epoch.append(loss.item())

            status_str = "Epoch: {}, Train, Batch {}/{}. Loss {}".format(self.current_epoch + 1, i + 1,
                                                                         len(self.train_loader),
                                                                         loss.item())
            self.log(status_str)

        return np.mean(train_loss_epoch)

    def val(self):
        self.model.eval()
        val_loss_epoch = []
        for i, data in enumerate(self.val_loader):
            inputs = data[0].cuda().float() if torch.cuda.is_available() else data[0].double()
            labels = data[1].cuda().float() if torch.cuda.is_available() else data[1].double()

            with torch.no_grad():
                outputs = self.model.forward(inputs).squeeze()
                loss = self.criterion(outputs, labels)
            val_loss_epoch.append(loss.item())

            status_str = "Epoch: {}, Val, Batch {}/{}. Loss {}".format(self.current_epoch + 1, i + 1,
                                                                       len(self.val_loader),
                                                                       loss.item())
            self.log(status_str)

        return np.mean(val_loss_epoch)

    def save_model(self):
        epoch_model_path = os.path.join(self.experiment_dir, 'models', 'model_{}.pt'.format(self.current_epoch))
        root_model_path = os.path.join(self.experiment_dir, 'latest_model.pt')

        if isinstance(self.model, torch.nn.DataParallel):
            model_dict = self.model.module.state_dict()
        else:
            model_dict = self.model.state_dict()

        state_dict = {'model': model_dict, 'optimizer': self.optimizer.state_dict()}
        torch.save(self.model.state_dict(), epoch_model_path)
        torch.save(state_dict, root_model_path)

    def record_stats(self, train_loss, val_loss, val_dice):
        self.training_losses.append(train_loss)
        self.val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.experiment_dir, 'training_losses.txt', self.training_losses)
        write_to_file_in_dir(self.experiment_dir, 'val_losses.txt', self.val_losses)

    def log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.experiment_dir, file_name, log_str)

    def log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.epochs - self.current_epoch - 1)
        train_loss = self.training_losses[self.current_epoch]
        val_loss = self.val_losses[self.current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.training_losses, label="Training Loss")
        plt.plot(x_axis, self.val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.name + " Stats Plot")
        plt.savefig(os.path.join(self.experiment_dir, "stat_plot.png"))
        plt.show()
