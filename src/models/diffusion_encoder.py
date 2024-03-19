import torch
from torch import nn


class DDPMEncoder(nn.Module):
    def __init__(
            self,
            in_dim,
            in_channel,
            time_step=1000,
            dec=None
    ):
        super().__init__()

        self.channel_list = [64, 128, 256, 512]

        # self.enc1 = nn.Sequential(nn.Conv1d(1, 10, 3, stride=1),nn.LeakyReLU(),nn.Conv1d(1, 10, 3, stride=1),)
        self.in_dim = in_dim
        self.fold_rate = 1
        self.kernal_size = 3
        self.dec = dec
        self.real_input_dim = in_dim
        # (
        #     int((in_dim+1000) / self.fold_rate**4 + 1) * self.fold_rate**4
        # )

        self.enc1 = nn.Sequential(
            nn.InstanceNorm1d(self.real_input_dim),
            nn.Conv1d(in_channel, self.channel_list[0], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim),
            nn.Conv1d(self.channel_list[0], self.channel_list[0], self.kernal_size, stride=self.fold_rate, padding=1),
            # nn.MaxPool1d(2),
        )
        self.enc2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list[0], self.channel_list[1], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list[1], self.channel_list[1], self.kernal_size, stride=self.fold_rate, padding=1),
            # nn.MaxPool1d(2),
        )
        self.enc3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.Conv1d(self.channel_list[1], self.channel_list[2], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.Conv1d(self.channel_list[2], self.channel_list[2], self.kernal_size, stride=self.fold_rate, padding=1),
            # nn.MaxPool1d(2),
        )
        self.enc4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.Conv1d(self.channel_list[2], self.channel_list[3], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.Conv1d(self.channel_list[3], self.channel_list[3], self.kernal_size, stride=self.fold_rate, padding=1),
            # nn.MaxPool1d(2),
        )

        self.dec1 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 4),
            # nn.ConvTranspose1d(
            nn.Conv1d(
                self.channel_list[3], self.channel_list[2], self.kernal_size, stride=self.fold_rate, padding=1
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 4),
            nn.Conv1d(self.channel_list[2], self.channel_list[2], self.kernal_size, stride=1, padding=1),
        )
        self.dec2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            # nn.ConvTranspose1d(
            nn.Conv1d(
                self.channel_list[2], self.channel_list[1], self.kernal_size, stride=self.fold_rate, padding=1
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.Conv1d(self.channel_list[1], self.channel_list[1], self.kernal_size, stride=1, padding=1),
        )
        self.dec3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            # nn.ConvTranspose1d(
            nn.Conv1d(
                self.channel_list[1], self.channel_list[0], self.kernal_size, stride=self.fold_rate, padding=1
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.Conv1d(self.channel_list[0], self.channel_list[0], self.kernal_size, stride=1, padding=1),
        )
        self.dec4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            # nn.ConvTranspose1d(
            nn.Conv1d(
                self.channel_list[0], self.channel_list[0], self.kernal_size, stride=self.fold_rate, padding=1
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list[0], in_channel, self.kernal_size, stride=1, padding=1),
        )

        self.time_encode = nn.Embedding(time_step, self.real_input_dim)

    def forward(self, input, time):
        assert input.shape[1] * input.shape[2] == self.in_dim, input.shape

        input_shape = input.shape
        input = input.reshape(input.shape[0], 1, -1)
        time_info = self.time_encode(time - 1).reshape(input.shape[0], 1, -1)  # [0, None, None]
        # time_info = time_info.repeat((input.shape[0], 1, 1))

        if len(input.size()) == 2:
            input = input.view(input.size(0), 1, -1)

        input = input

        emb_enc1 = self.enc1(input + time_info)
        emb_enc2 = self.enc2(emb_enc1 + time_info)
        emb_enc3 = self.enc3(emb_enc2 + time_info)
        emb_enc4 = self.enc4(emb_enc3 + time_info)
        # import pdb; pdb.set_trace()

        emb_dec1 = self.dec1(emb_enc4 + time_info) + emb_enc3
        emb_dec2 = self.dec2(emb_dec1 + time_info) + emb_enc2
        emb_dec3 = self.dec3(emb_dec2 + time_info) + emb_enc1
        emb_dec4 = self.dec4(emb_dec3 + time_info)
        # import pdb; pdb.set_trace()

        emb_dec4 = emb_dec4.reshape(input_shape)
        # if self.dec is not None:
        #     emb_dec4 = self.dec.Dec(emb_dec4)

        return emb_dec4
