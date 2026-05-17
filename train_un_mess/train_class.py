import torch
class train_class:
    """ The training class for the continuous SDE path, which will be used to calculate the training loss and test loss.
      It will be passed to the DiffusionTrainer, which will call the training_loss and test_loss methods during training and testing. """
    def __init__(self, sde,  loss_fn,  w_masked_pix=0.7):
        
        self._sde = sde
        self._device = None
        self._B, self._C, self._L = None, None, None
        self._std = None

        # loss function and masks for loss calculation
        self._loss_fn = loss_fn # loss function, e.g., nn.MSELoss()
        self._w_masked_pix = w_masked_pix # weighting for masked pixels in the loss calculation, between 0 and 1

        # Preset the noise and time steps for the forward process.
        self._t = None
        self._z = None

        # to be loaded from the dataloader
        self._parsed_data = None
        self._melspec = None
        self._masked_cond = None
        self._mask = None
        self._mask_mask = None
        self._masked_audio_time_mask = None


    def load_data(self, parsed_data):
        self._parsed_data = parsed_data

        # unpack the parsed data
        self._melspec = self._parsed_data['melspec_key']
        self._masked_cond = self._parsed_data['masked_cond_key']
        self._mask = self._parsed_data['mask_key']
        self._mask_mask = self._parsed_data['mask_mask_key']
        self._masked_audio_time_mask = self._parsed_data['masked_audio_time_mask_key']

        # set device and shape parameters
        self._device = self._melspec.device
        self._B, self._C, self._L = self._melspec.shape


    def prepare_step(self):
        """ Forward process of the SDE, used for sampling and computing the training loss
            args:
                None, as the necessary inputs are precomputed in the constructor
            returns:
                xt: the noisy data (mel-spectrogram) at time t
            update:
                self._t: the time steps for the forward process, shape=(B,)
                self._z: the noise for the forward process, shape=(B, C, L)
                self._std: the standard deviation of the noise at time t, shape=(B, 1, 1)
        """
        
        # Compute the noise and time steps for the forward process.
        eps = 1e-5
        self._t = torch.rand(self._B, device=self._device) * (self._sde.T - eps) + eps   # [B]
        self._z = torch.randn_like(self._melspec)                            # [B, C, L]


        mean, unbrodcast_std = self._sde.marginal_prob(self._melspec, None, self._t)
        self._std = unbrodcast_std.view(self._B, 1, 1) # [B, 1, 1]
        x_t = mean + self._std * self._z

        return x_t
    
    def apply_mask(self, x_t, mask):
        """ apply masking - in need of research
            args:
                x_t: the noisy data (mel-spectrogram) at time t, shape=(B, C, L)
                mask: the binary mask, shape=(B, 1, L)
            returns:
                masked_x_t: the masked noisy data, where the unmasked regions are replaced by the original melspec
        """
        return self._melspec * mask + x_t * (1 - mask)
    
    def calculate_loss(self, predicted_score):
        """ calculate the loss with shared weighting for masked and unmasked regions
            args:
                predicted_score: the predicted score by the network, shape=(B, C, L)
            returns:
                loss: the final loss value, a scalar
                """
        
         # denoising score matching loss
        loss = self._loss_fn(predicted_score * self._std, -self._z) # shape=(B, C, L)

        # ── Shared loss weighting ──────────────────
        loss = loss * self._mask_mask
        unmasked_loss = torch.sum(self._mask * loss) / (torch.sum(self._mask * self._mask_mask) * loss.shape[1])
        masked_loss   = torch.sum((1-self._mask) * loss) / (torch.sum((1-self._mask) * self._mask_mask) * loss.shape[1])
        return (1 - self._w_masked_pix) * unmasked_loss + self._w_masked_pix * masked_loss # shape=(), a scalar



    def training_loss(self, net, text, input_text, on_noisy_masked_melspec, parsed_data):
        """
        Compute the training loss of epsilon and epsilon_theta

        Parameters:
        net (torch network):            the wavenet model
        loss_fn (torch loss function):  the loss function, default is nn.MSELoss()
        X (torch.tensor):               training data, shape=(batchsize, 1, length of audio)
        diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by get_diffusion_hyperparams
                                        note, the tensors need to be cuda tensors

        Returns:
        training loss
        """ 

        # ── Continuous SDE (new path) ─────────────────────────────────────────

        self.load_data(parsed_data) # load the data from the dataloader, which will set the necessary attributes for the training step.

        x_t = self.prepare_step() # prepare the noisy data at time t, which will set the necessary attributes for the training step.

        if on_noisy_masked_melspec:
            x_t = self.apply_mask(x_t, self._mask)

        cond_drop_prob = 0.2
        predicted_score = net(x_t, self._masked_cond, self._t.view(self._B, 1), cond_drop_prob,
                                text=text, input_text=input_text,
                                mask_padding_time=self._masked_audio_time_mask,
                                mask_padding_frames=self._mask_mask)
        

        return self.calculate_loss(predicted_score) # return the final loss

    def test_loss(self, net, text, input_text, on_noisy_masked_melspec, parsed_data):
        """ compute the test loss, which is the same as the training loss """
        return self.training_loss(net, text, input_text, on_noisy_masked_melspec, parsed_data)
    


    def training_loss_phoneme(self, net, text, input_text, on_noisy_masked_melspec, parsed_data):
        """
        Compute the training loss of epsilon and epsilon_theta

        Parameters:
        net (torch network):            the wavenet model
        loss_fn (torch loss function):  the loss function, default is nn.MSELoss()
        X (torch.tensor):               training data, shape=(batchsize, 1, length of audio)
        diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by get_diffusion_hyperparams
                                        note, the tensors need to be cuda tensors

        Returns:
        training loss
        """ 

        # ── Continuous SDE (new path) ─────────────────────────────────────────

        self.load_data(parsed_data) # load the data from the dataloader, which will set the necessary attributes for the training step.

        x_t = self.prepare_step() # prepare the noisy data at time t, which will set the necessary attributes for the training step.

        if on_noisy_masked_melspec:
            x_t = self.apply_mask(x_t, self._mask)

        cond_drop_prob = 0.2
        predicted_score = net(x_t, self._masked_cond, self._t.view(self._B, 1), cond_drop_prob,
                                text=text, input_text=input_text,
                                mask_padding_time=self._masked_audio_time_mask,
                                mask_padding_frames=self._mask_mask)
        

        return self.calculate_loss(predicted_score) # return the final loss

    def test_loss_phoneme(self, net, text, input_text, on_noisy_masked_melspec, parsed_data):
        """ compute the test loss, which is the same as the training loss """
        return self.training_loss_phoneme(net, text, input_text, on_noisy_masked_melspec, parsed_data)